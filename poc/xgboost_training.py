"""
GPU-Accelerated SDM using XGBoost
Trains one model per species with spatial block cross-validation.

Predictors: WorldClim BIO climate, OpenLandMap/SoilGrids soil (aligned 10m),
and GEDTM30 elevation. All three share the same 2160 x 1080 grid.

Overfitting controls: spatial CV (not random splits), early stopping,
shallower regularized trees, and a final model capped at the CV-chosen
number of boosting rounds.
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from repo_paths import data, relpath

N_SPATIAL_FOLDS = 5
N_SPATIAL_BLOCKS = 10  # geographic clusters; each CV fold holds some out
EARLY_STOPPING_ROUNDS = 30
MAX_BOOST_ROUNDS = 500  # ceiling; early stopping usually picks far fewer
MIN_UNIQUE_CELLS = 80  # spatial CV is noise below this on an 18 km grid
MIN_USABLE_FOLDS = 3   # do not save a model scored on 1–2 regions

CLIMATE_DIR = data("climate_current")
SOIL_DIR = data("soil_data", "aligned_10m")
TOPO_PATH = data("topography", "gedtm30", "gedtm30_v1.2_elev_10m.tif")
MODEL_DIR = data("models")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.csv")

# Must match download_species.py. GBIF genus search also returns moths, birds,
# bacteria, etc. whose names contain these words — keep binomials whose *genus*
# is actually a tree.
TREE_GENERA = {
    "Quercus", "Pinus", "Acer", "Fagus", "Betula",
    "Populus", "Salix", "Fraxinus", "Ulmus", "Tilia",
    "Abies", "Picea", "Larix", "Carpinus", "Alnus",
    "Prunus", "Sorbus", "Juniperus", "Eucalyptus", "Cedrus",
    "Magnolia", "Malus", "Corylus", "Castanea", "Platanus",
    "Aesculus", "Robinia", "Catalpa", "Liquidambar", "Nyssa",
    "Cornus", "Crataegus", "Ilex", "Rhamnus", "Viburnum",
    "Acacia", "Ficus", "Cinnamomum", "Terminalia",
}

# Regularized GPU histogram trees. Climate and soil layers are collinear,
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


def _list_tifs(directory, skip_substrings=()):
    files = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".tif"):
            continue
        if any(token in name for token in skip_substrings):
            continue
        files.append(os.path.join(directory, name))
    return files


def collect_predictor_paths():
    """Climate + aligned soil (not source_flag / raw SoilGrids) + elevation."""
    layers = [(os.path.basename(p), p) for p in _list_tifs(CLIMATE_DIR)]
    if not layers:
        raise FileNotFoundError(f"No climate GeoTIFFs in {CLIMATE_DIR}")

    soil = _list_tifs(SOIL_DIR, skip_substrings=("source_flag",))
    if not soil:
        raise FileNotFoundError(f"No aligned soil GeoTIFFs in {SOIL_DIR}")
    layers.extend((os.path.basename(p), p) for p in soil)

    if not os.path.isfile(TOPO_PATH):
        raise FileNotFoundError(f"Elevation raster not found: {TOPO_PATH}")
    layers.append((os.path.basename(TOPO_PATH), TOPO_PATH))
    return layers


def _read_band(path, expected_shape=None):
    with rasterio.open(path) as src:
        if expected_shape is not None and (src.height, src.width) != expected_shape:
            raise ValueError(
                f"{path} is {src.width}x{src.height}, expected "
                f"{expected_shape[1]}x{expected_shape[0]} (shared 10m grid)"
            )
        band = src.read(1).astype(np.float32, copy=False)
        if src.nodata is not None:
            band = np.where(band == src.nodata, np.nan, band)
        # GEDTM / float rasters sometimes use a huge sentinel instead of NaN
        band = np.where(np.abs(band) > 1e30, np.nan, band)
        return band, src.transform, src.height, src.width


# ─────────────────────────────────────────────────────
# STEP 1: Load climate, soil, elevation once; sample many points
# ─────────────────────────────────────────────────────
class PredictorRasters:
    """Keep all predictor GeoTIFFs in memory and sample with vectorized indexing.

    Layers must share the WorldClim 10-arc-minute grid (2160 x 1080).
    """

    def __init__(self, layer_paths):
        if not layer_paths:
            raise FileNotFoundError("No predictor rasters to load")

        self.names = []
        bands = []
        first_path = layer_paths[0][1]
        band, transform, height, width = _read_band(first_path)
        self.transform = transform
        self.height = height
        self.width = width
        self.template_path = first_path
        bands.append(band)
        self.names.append(layer_paths[0][0])

        for name, path in layer_paths[1:]:
            band, _, _, _ = _read_band(path, expected_shape=(height, width))
            bands.append(band)
            self.names.append(name)

        # shape: [n_variables, height, width]
        self.stack = np.stack(bands, axis=0)
        self.valid_mask = np.all(np.isfinite(self.stack), axis=0)

    def sample(self, coords):
        """Sample all predictor layers at (lat, lon) rows. Out-of-bounds → NaN."""
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


def generate_pseudo_absences(presence_coords, transform, land, n_absences,
                             rng=None):
    """Sample random background points on land inside the species' range.

    `land` should be True only where every predictor is valid, so background
    rows are not dropped later for missing soil or elevation.

    Endemic species on a coarse grid may have more records than leftover
    land pixels. Grow the window, then use every remaining unique cell
    rather than crashing.
    """
    if rng is None:
        rng = np.random.default_rng(42)

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


def is_tree_species(name):
    """True for a binomial whose genus is in TREE_GENERA (not 'Erigeron acer')."""
    parts = str(name).split()
    if len(parts) < 2:
        return False
    if parts[1].lower().rstrip(".") in {"sect", "sp", "spp", "x"}:
        return False
    return parts[0] in TREE_GENERA


def species_slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def xgb_params_for_n(n_samples):
    """Soften leaf constraints when a species has few unique cells."""
    params = dict(XGB_GPU_PARAMS)
    if n_samples < 200:
        params["min_child_weight"] = 2
        params["gamma"] = 0.3
        params["reg_lambda"] = 2.0
    elif n_samples < 500:
        params["min_child_weight"] = 4
        params["gamma"] = 0.5
        params["reg_lambda"] = 3.0
    return params


def _make_classifier(n_estimators, n_samples, early_stopping_rounds=None):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        **xgb_params_for_n(n_samples),
    )


# ─────────────────────────────────────────────────────
# STEP 2: Train + validate ONE species
# ─────────────────────────────────────────────────────
def train_species_sdm(presence_coords, absence_coords, predictors):
    """Train and spatially cross-validate an SDM for one species."""

    coords = np.vstack([presence_coords, absence_coords])
    y = np.concatenate([
        np.ones(len(presence_coords), dtype=np.float32),
        np.zeros(len(absence_coords), dtype=np.float32),
    ])
    X = predictors.sample(coords)

    valid = ~np.isnan(X).any(axis=1)
    X, y, coords = X[valid], y[valid], coords[valid]
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    groups = spatial_block_ids(coords)
    splits = spatial_cv_splits(X, y, groups)
    auc_scores = []
    best_ntrees = []
    n_folds = 0 if splits is None else len(splits)

    if splits is None:
        return None, float("nan"), 0, 0, n_folds

    n_usable = 0
    n_samples = len(y)
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # AUC needs both classes in train and in the held-out region
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        n_usable += 1

        model = _make_classifier(
            n_estimators=MAX_BOOST_ROUNDS,
            n_samples=n_samples,
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
    if not best_ntrees:
        return None, mean_auc, 0, n_usable, n_folds

    # Retrain on all points, capped at the CV-chosen number of rounds.
    n_trees = max(int(np.median(best_ntrees)), 10)
    final_model = _make_classifier(
        n_estimators=n_trees,
        n_samples=n_samples,
        early_stopping_rounds=None,
    )
    final_model.fit(X, y, verbose=False)

    return final_model, mean_auc, n_trees, n_usable, n_folds


# ─────────────────────────────────────────────────────
# STEP 3: Batch process all 500 species
# ─────────────────────────────────────────────────────
def _append_metric(rows, **kwargs):
    rows.append(kwargs)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(data("gbif_500_species.csv"))
    predictors = PredictorRasters(collect_predictor_paths())
    n_clim = sum(1 for n in predictors.names if n.startswith("wc2.1_"))
    n_soil = sum(1 for n in predictors.names if n.endswith("_10m.tif")
                 and "elev" not in n)
    print(
        f"Loaded {len(predictors.names)} predictors "
        f"({n_clim} climate, {n_soil} soil, 1 elevation)  "
        f"land cells = {int(predictors.valid_mask.sum()):,}"
    )
    with open(os.path.join(MODEL_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(predictors.names) + "\n")

    metrics = []
    saved_aucs = []
    n_skip = {"not_tree": 0, "few_cells": 0, "no_folds": 0, "few_folds": 0,
              "no_background": 0}

    for species in df["species"].unique():
        sp_data = df[df["species"] == species]
        n_records = len(sp_data)
        if not is_tree_species(species):
            n_skip["not_tree"] += 1
            _append_metric(
                metrics, species=species, n_records=n_records,
                n_unique_cells=0, n_absences=0, auc=np.nan, n_trees=0,
                n_folds_usable=0, n_folds=0, model_path="",
                status="skip_not_tree",
            )
            continue

        presence = predictors.unique_pixel_coords(
            sp_data[["latitude", "longitude"]].values
        )
        n_cells = len(presence)
        if n_cells < MIN_UNIQUE_CELLS:
            n_skip["few_cells"] += 1
            _append_metric(
                metrics, species=species, n_records=n_records,
                n_unique_cells=n_cells, n_absences=0, auc=np.nan, n_trees=0,
                n_folds_usable=0, n_folds=0, model_path="",
                status="skip_few_cells",
            )
            continue

        try:
            absence = generate_pseudo_absences(
                presence,
                predictors.transform,
                predictors.valid_mask,
                n_absences=len(presence),
            )
        except ValueError as exc:
            n_skip["no_background"] += 1
            print(f"{species:30s}  SKIP background ({exc})")
            _append_metric(
                metrics, species=species, n_records=n_records,
                n_unique_cells=n_cells, n_absences=0, auc=np.nan, n_trees=0,
                n_folds_usable=0, n_folds=0, model_path="",
                status="skip_no_background",
            )
            continue

        try:
            model, auc, n_trees, n_usable, n_folds = train_species_sdm(
                presence, absence, predictors
            )
        except Exception as exc:
            print(f"{species:30s}  SKIP train ({type(exc).__name__}: {exc})")
            _append_metric(
                metrics, species=species, n_records=n_records,
                n_unique_cells=n_cells, n_absences=len(absence), auc=np.nan,
                n_trees=0, n_folds_usable=0, n_folds=0,
                model_path="", status="skip_train_error",
            )
            continue
        auc_txt = "nan " if np.isnan(auc) else f"{auc:.3f}"

        if model is None or n_usable == 0:
            n_skip["no_folds"] += 1
            print(
                f"{species:30s}  AUC = {auc_txt}  trees = {n_trees:3d}  "
                f"folds = {n_usable}/{n_folds}  SKIP no mixed folds"
            )
            _append_metric(
                metrics, species=species, n_records=n_records,
                n_unique_cells=n_cells, n_absences=len(absence), auc=auc,
                n_trees=n_trees, n_folds_usable=n_usable, n_folds=n_folds,
                model_path="", status="skip_no_folds",
            )
            continue

        if n_usable < MIN_USABLE_FOLDS:
            n_skip["few_folds"] += 1
            print(
                f"{species:30s}  AUC = {auc_txt}  trees = {n_trees:3d}  "
                f"folds = {n_usable}/{n_folds}  SKIP <{MIN_USABLE_FOLDS} folds"
            )
            _append_metric(
                metrics, species=species, n_records=n_records,
                n_unique_cells=n_cells, n_absences=len(absence), auc=auc,
                n_trees=n_trees, n_folds_usable=n_usable, n_folds=n_folds,
                model_path="", status="skip_few_folds",
            )
            continue

        model_path = os.path.join(MODEL_DIR, f"{species_slug(species)}.json")
        model.save_model(model_path)
        saved_aucs.append(auc)
        print(
            f"{species:30s}  AUC = {auc_txt}  trees = {n_trees:3d}  "
            f"folds = {n_usable}/{n_folds}  saved"
        )
        _append_metric(
            metrics, species=species, n_records=n_records,
            n_unique_cells=n_cells, n_absences=len(absence), auc=auc,
            n_trees=n_trees, n_folds_usable=n_usable, n_folds=n_folds,
            model_path=relpath(model_path), status="saved",
        )

    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    n_saved = len(saved_aucs)
    print(
        f"\nSaved {n_saved} models to {MODEL_DIR}/  "
        f"mean AUC = {np.mean(saved_aucs) if n_saved else float('nan'):.3f}"
    )
    print(
        "Skipped: "
        f"{n_skip['not_tree']} not tree, "
        f"{n_skip['few_cells']} <{MIN_UNIQUE_CELLS} cells, "
        f"{n_skip['no_folds']} no mixed folds, "
        f"{n_skip['few_folds']} <{MIN_USABLE_FOLDS} folds, "
        f"{n_skip['no_background']} no background"
    )
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
