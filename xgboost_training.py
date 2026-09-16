"""
GPU-Accelerated SDM using XGBoost
Trains one model per species with spatial block cross-validation.

Overfitting controls: spatial CV (not random splits), early stopping,
shallower regularized trees, and a final model capped at the CV-chosen
number of boosting rounds.
"""

import os

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

N_SPATIAL_FOLDS = 5
N_SPATIAL_BLOCKS = 10  # geographic clusters; each CV fold holds some out
EARLY_STOPPING_ROUNDS = 30
MAX_BOOST_ROUNDS = 500  # ceiling; early stopping usually picks far fewer

# Regularized GPU histogram trees. WorldClim BIO vars are highly collinear,
# so we keep trees shallow and subsample features each round.
XGB_GPU_PARAMS = dict(
    objective="binary:logistic",
    tree_method="hist",
    device="cuda",  # correct for amd_xgboost / ROCm on MI300X
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=8,
    gamma=1.0,
    reg_lambda=5.0,
    reg_alpha=0.5,
    max_bin=256,
    eval_metric="auc",
    n_jobs=0,
    random_state=42,
    verbosity=0,
)


# ─────────────────────────────────────────────────────
# STEP 1: Load climate rasters once, sample many points
# ─────────────────────────────────────────────────────
class ClimateRasters:
    """Keep all BIO GeoTIFFs in memory and sample them with vectorized indexing.

    The old loop called src.read(1) once per occurrence, which re-read the
    whole raster for every point. That dwarfed GPU training time.
    """

    def __init__(self, climate_dir):
        files = sorted(
            f for f in os.listdir(climate_dir) if f.endswith(".tif")
        )
        if not files:
            raise FileNotFoundError(f"No .tif files in {climate_dir}")

        self.names = files
        self.template_path = os.path.join(climate_dir, files[0])
        bands = []
        with rasterio.open(self.template_path) as src:
            self.transform = src.transform
            self.height = src.height
            self.width = src.width

        for name in files:
            path = os.path.join(climate_dir, name)
            with rasterio.open(path) as src:
                band = src.read(1).astype(np.float32, copy=False)
                if src.nodata is not None:
                    band = np.where(band == src.nodata, np.nan, band)
                bands.append(band)

        # shape: [n_variables, height, width]
        self.stack = np.stack(bands, axis=0)

    def sample(self, coords):
        """Sample all climate layers at (lat, lon) rows. Out-of-bounds → NaN."""
        coords = np.asarray(coords, dtype=np.float64)
        rows, cols = rasterio.transform.rowcol(
            self.transform, coords[:, 1], coords[:, 0]
        )
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)

        n_pts = len(coords)
        n_vars = self.stack.shape[0]
        out = np.full((n_pts, n_vars), np.nan, dtype=np.float32)
        ok = (
            (rows >= 0)
            & (rows < self.height)
            & (cols >= 0)
            & (cols < self.width)
        )
        if not np.any(ok):
            return out
        out[ok] = self.stack[:, rows[ok], cols[ok]].T
        return out

    def unique_pixel_coords(self, coords):
        """Keep one occurrence per climate cell (GBIF often repeats a pixel)."""
        coords = np.asarray(coords, dtype=np.float64)
        rows, cols = rasterio.transform.rowcol(
            self.transform, coords[:, 1], coords[:, 0]
        )
        rc = np.column_stack([np.asarray(rows), np.asarray(cols)])
        _, idx = np.unique(rc, axis=0, return_index=True)
        return coords[np.sort(idx)]


def _presence_window_mask(valid, transform, presence_coords, pad_frac=0.25,
                          min_pad_deg=2.0):
    """True on land pixels inside a padded bounding box of the occurrences."""
    lats = presence_coords[:, 0]
    lons = presence_coords[:, 1]
    lat_pad = max((lats.max() - lats.min()) * pad_frac, min_pad_deg)
    lon_pad = max((lons.max() - lons.min()) * pad_frac, min_pad_deg)
    lat_min, lat_max = lats.min() - lat_pad, lats.max() + lat_pad
    lon_min, lon_max = lons.min() - lon_pad, lons.max() + lon_pad

    r0, c0 = rasterio.transform.rowcol(transform, lon_min, lat_max)
    r1, c1 = rasterio.transform.rowcol(transform, lon_max, lat_min)
    r0, r1 = np.clip([r0, r1], 0, valid.shape[0] - 1)
    c0, c1 = np.clip([c0, c1], 0, valid.shape[1] - 1)
    r_lo, r_hi = int(min(r0, r1)), int(max(r0, r1)) + 1
    c_lo, c_hi = int(min(c0, c1)), int(max(c0, c1)) + 1

    window = np.zeros_like(valid, dtype=bool)
    window[r_lo:r_hi, c_lo:c_hi] = True
    return valid & window


def generate_pseudo_absences(presence_coords, raster_path, n_absences,
                             rng=None):
    """Sample random background points on land inside the species' range.

    Global background puts almost every 0 in its own continent-sized KMeans
    block, so spatial CV folds become one-class and AUC is undefined (nan).
    Restricting to a padded bounding box keeps 0s and 1s in the same region.

    Endemic species on a coarse grid may have more records than leftover
    land pixels. Grow the window, then use every remaining unique cell
    rather than crashing.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    with rasterio.open(raster_path) as src:
        band = src.read(1)
        nodata = src.nodata
        transform = src.transform

    land = np.isfinite(band)
    if nodata is not None:
        land &= band != nodata

    presence_coords = np.asarray(presence_coords)
    pr, pc = rasterio.transform.rowcol(
        transform, presence_coords[:, 1], presence_coords[:, 0]
    )
    pr, pc = np.asarray(pr), np.asarray(pc)
    in_bounds = (
        (pr >= 0)
        & (pr < land.shape[0])
        & (pc >= 0)
        & (pc < land.shape[1])
    )

    rows = cols = np.array([], dtype=int)
    for pad_frac in (0.25, 0.5, 1.0, 2.0, 4.0):
        valid = _presence_window_mask(
            land, transform, presence_coords, pad_frac=pad_frac
        ).copy()
        valid[pr[in_bounds], pc[in_bounds]] = False
        rows, cols = np.where(valid)
        if len(rows) >= n_absences:
            break

    if len(rows) == 0:
        raise ValueError("No valid background pixels in the species window")

    n_take = min(int(n_absences), len(rows))
    pick = rng.choice(len(rows), size=n_take, replace=False)
    lons, lats = rasterio.transform.xy(transform, rows[pick], cols[pick])
    return np.column_stack([np.asarray(lats), np.asarray(lons)])


def spatial_block_ids(coords, n_blocks=N_SPATIAL_BLOCKS, random_state=42):
    """Assign each point to a geographic cluster (approx. equal-distance x/y)."""
    lat = np.asarray(coords[:, 0], dtype=float)
    lon = np.asarray(coords[:, 1], dtype=float)
    mean_lat = np.deg2rad(np.clip(np.nanmean(lat), -89.9, 89.9))
    xy = np.column_stack([lon * np.cos(mean_lat), lat])

    n_blocks = int(min(n_blocks, len(coords)))
    if n_blocks < 2:
        return np.zeros(len(coords), dtype=int)

    return KMeans(
        n_clusters=n_blocks, n_init=10, random_state=random_state
    ).fit_predict(xy)


def spatial_cv_splits(X, y, groups, n_splits=N_SPATIAL_FOLDS):
    """Leave whole map-blocks out of each fold (no nearby train/test leak)."""
    n_groups = len(np.unique(groups))
    n_splits = int(min(n_splits, n_groups))
    if n_splits < 2:
        return None

    try:
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        )
        return list(cv.split(X, y, groups))
    except ValueError:
        cv = GroupKFold(n_splits=n_splits)
        return list(cv.split(X, y, groups))


def _make_classifier(n_estimators, early_stopping_rounds=None):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        **XGB_GPU_PARAMS,
    )


# ─────────────────────────────────────────────────────
# STEP 2: Train + validate ONE species
# ─────────────────────────────────────────────────────
def train_species_sdm(presence_coords, absence_coords, climate):
    """Train and spatially cross-validate an SDM for one species."""

    coords = np.vstack([presence_coords, absence_coords])
    y = np.concatenate([
        np.ones(len(presence_coords), dtype=np.float32),
        np.zeros(len(absence_coords), dtype=np.float32),
    ])
    X = climate.sample(coords)

    valid = ~np.isnan(X).any(axis=1)
    X, y, coords = X[valid], y[valid], coords[valid]
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    groups = spatial_block_ids(coords)
    splits = spatial_cv_splits(X, y, groups)
    auc_scores = []
    best_ntrees = []

    if splits is None:
        raise ValueError("Not enough spatial blocks for cross-validation")

    n_usable = 0
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # AUC needs both classes in train and in the held-out region
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        n_usable += 1

        model = _make_classifier(
            n_estimators=MAX_BOOST_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        pred = model.predict_proba(X_test)[:, 1]
        auc_scores.append(roc_auc_score(y_test, pred))
        best_ntrees.append(int(model.best_iteration) + 1)

    mean_auc = float(np.mean(auc_scores)) if auc_scores else float("nan")

    # Retrain on all points, but do not grow more trees than CV found useful.
    n_trees = int(np.median(best_ntrees)) if best_ntrees else 100
    n_trees = max(n_trees, 20)
    final_model = _make_classifier(
        n_estimators=n_trees,
        early_stopping_rounds=None,
    )
    final_model.fit(X, y, verbose=False)

    return final_model, mean_auc, n_trees, n_usable, len(splits)


# ─────────────────────────────────────────────────────
# STEP 3: Batch process all 500 species
# ─────────────────────────────────────────────────────
def main():
    df = pd.read_csv("gbif_500_species.csv")
    climate = ClimateRasters("climate_current")

    results = {}
    for species in df["species"].unique():
        sp_data = df[df["species"] == species]
        presence = climate.unique_pixel_coords(
            sp_data[["latitude", "longitude"]].values
        )

        if len(presence) < 30:
            continue

        absence = generate_pseudo_absences(
            presence,
            climate.template_path,
            n_absences=len(presence),
        )

        model, auc, n_trees, n_usable, n_folds = train_species_sdm(
            presence, absence, climate
        )
        results[species] = {"model": model, "auc": auc, "n_trees": n_trees}
        auc_txt = "nan " if np.isnan(auc) else f"{auc:.3f}"
        print(
            f"{species:30s}  AUC = {auc_txt}  trees = {n_trees:3d}  "
            f"folds = {n_usable}/{n_folds}"
        )

    all_aucs = [r["auc"] for r in results.values()]
    n_ok = int(np.isfinite(all_aucs).sum())
    print(
        f"\nMean AUC across {n_ok}/{len(results)} species "
        f"with usable folds: {np.nanmean(all_aucs):.3f}"
    )


if __name__ == "__main__":
    main()
