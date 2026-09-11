"""
GPU-Accelerated SDM using XGBoost
Trains one model per species with spatial block cross-validation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import rasterio
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import os

N_SPATIAL_FOLDS = 5
N_SPATIAL_BLOCKS = 10  # geographic clusters; each CV fold holds some out

# ─────────────────────────────────────────────────────
# STEP 1: Extract climate at each occurrence point
# ─────────────────────────────────────────────────────
def extract_climate(coords, climate_dir):
    """Sample all 19 climate rasters at given coordinates."""
    climate_files = sorted([f for f in os.listdir(climate_dir)
                           if f.endswith('.tif')])

    features = []
    for f in climate_files:
        path = os.path.join(climate_dir, f)
        with rasterio.open(path) as src:
            values = []
            for lat, lon in coords:
                try:
                    row, col = src.index(lon, lat)
                    val = src.read(1)[row, col]
                    values.append(val)
                except (IndexError, ValueError):
                    values.append(np.nan)
            features.append(values)

    # shape: [n_points, n_variables]
    return np.array(features).T


def generate_pseudo_absences(presence_coords, raster_path, n_absences,
                             rng=None):
    """Sample random background (pseudo-absence) points on valid land pixels.

    Presence coords are (lat, lon). Returned array has the same layout.
    Ocean / nodata cells and cells that already contain a presence are skipped.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    with rasterio.open(raster_path) as src:
        band = src.read(1)
        nodata = src.nodata
        transform = src.transform

    valid = np.isfinite(band)
    if nodata is not None:
        valid &= band != nodata

    # Do not place background points on the same pixels as known presences
    for lat, lon in presence_coords:
        row, col = rasterio.transform.rowcol(transform, lon, lat)
        if 0 <= row < valid.shape[0] and 0 <= col < valid.shape[1]:
            valid[row, col] = False

    rows, cols = np.where(valid)
    if len(rows) < n_absences:
        raise ValueError(
            f"Only {len(rows)} valid background pixels; need {n_absences}"
        )

    pick = rng.choice(len(rows), size=n_absences, replace=False)
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


# ─────────────────────────────────────────────────────
# STEP 2: Train + validate ONE species
# ─────────────────────────────────────────────────────
def train_species_sdm(presence_coords, absence_coords, climate_dir):
    """Train and spatially cross-validate an SDM for one species."""

    # Extract climate features
    X_pres = extract_climate(presence_coords, climate_dir)
    X_abs = extract_climate(absence_coords, climate_dir)

    # Combine: presence=1, absence=0  (keep coords aligned with rows)
    X = np.vstack([X_pres, X_abs])
    y = np.concatenate([np.ones(len(X_pres)),
                        np.zeros(len(X_abs))])
    coords = np.vstack([presence_coords, absence_coords])

    # Remove rows with NaN climate
    valid = ~np.isnan(X).any(axis=1)
    X, y, coords = X[valid], y[valid], coords[valid]

    groups = spatial_block_ids(coords)
    splits = spatial_cv_splits(X, y, groups)
    auc_scores = []

    if splits is None:
        raise ValueError("Not enough spatial blocks for cross-validation")

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Held-out region may be all presence or all background
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # XGBoost with GPU
        model = xgb.XGBClassifier(
            tree_method="hist",       # modern GPU method
            device="cuda",            # ← USE GPU
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="auc"
        )
        model.fit(X_train, y_train)

        # Validate on held-out fold
        pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        auc_scores.append(auc)

    mean_auc = float(np.mean(auc_scores)) if auc_scores else float("nan")

    # ── Train FINAL model on all data ──
    final_model = xgb.XGBClassifier(
        tree_method="hist", device="cuda",
        n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.8
    )
    final_model.fit(X, y)

    return final_model, mean_auc


# ─────────────────────────────────────────────────────
# STEP 3: Batch process all 500 species
# ─────────────────────────────────────────────────────
def main():
    df = pd.read_csv("gbif_500_species.csv")
    climate_dir = "climate_current"

    results = {}
    for species in df['species'].unique():
        sp_data = df[df['species'] == species]
        presence = sp_data[['latitude', 'longitude']].values

        if len(presence) < 30:   # skip species with too few records
            continue

        # Use any current-climate GeoTIFF as the spatial template
        template = next(
            f for f in sorted(os.listdir(climate_dir)) if f.endswith(".tif")
        )
        absence = generate_pseudo_absences(
            presence,
            os.path.join(climate_dir, template),
            n_absences=len(presence),
        )

        model, auc = train_species_sdm(presence, absence, climate_dir)
        results[species] = {'model': model, 'auc': auc}

        print(f"{species:30s}  AUC = {auc:.3f}")

    # Summary
    all_aucs = [r['auc'] for r in results.values()]
    print(f"\nMean AUC across {len(results)} species: {np.mean(all_aucs):.3f}")


if __name__ == "__main__":
    main()

