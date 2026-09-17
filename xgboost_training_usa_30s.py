"""
GPU XGBoost SDMs for the contiguous USA on the 1 km (30 arcsec) country grid.

Leaves poc/xgboost_training.py (global 10-arc-minute, ~18.5 km) untouched.

Predictors, all on data/country_data/USA 7020 x 3060:
  19 WorldClim BIO (numeric order bio_1..bio_19, not filename sort)
  9  climate extras (aridity, PET, srad, wind, …; not WorldClim elevation)
  22 soil (OpenLandMap/SoilGrids 30s; not source_flag)
  11 terrain (GEDTM30 elev/slope/northness/…; not raw aspect degrees)

Satellite / NDVI / land cover are refused (DO_NOT_TRAIN_ON_THIS.md).

Occurrences: cleaned + thinned 1 km table from clean_species_usa_30s.py.
Background: target-group — cells where *other* list trees were recorded,
not uniform land (which teaches "near a road").

Default species list is data/usa_tree_species_seed.csv (few, better).
Grow later with --species-list data/usa_tree_species_list.csv after
re-cleaning the full checklist.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from repo_paths import data, relpath
from clean_species_usa_30s import (
    FULL_LIST,
    SEED_LIST,
    load_species_list,
    read_grid,
    thin_to_grid,
)

N_SPATIAL_FOLDS = 5
N_SPATIAL_BLOCKS = 20
EARLY_STOPPING_ROUNDS = 30
MAX_BOOST_ROUNDS = 500
MIN_UNIQUE_CELLS = 150
MIN_USABLE_FOLDS = 3
MAX_PRESENCE_CELLS = 25_000
MAX_ABSENCES = 25_000

COUNTRY_DATA = data("country_data", "USA")
CLIMATE_DIR = os.path.join(COUNTRY_DATA, "climate_30s")
SOIL_DIR = os.path.join(COUNTRY_DATA, "soil_30s")
TOPO_DIR = os.path.join(COUNTRY_DATA, "topography_30s")
THINNED_CSV = data(
    "species_occurrences", "US", "gbif_trees_US_30s_thinned.csv"
)
CLEAN_CSV = data(
    "species_occurrences", "US", "gbif_trees_US_30s_clean.csv"
)
MODEL_DIR = data("models_usa_30s")
DO_NOT_TRAIN_MARKER = "DO_NOT_TRAIN_ON_THIS.md"
FORBIDDEN_PATH_TOKENS = (
    "satellite",
    "ndvi",
    "landcover",
    "land_cover",
    "vegetation",
    "treecover",
    "tree_frac",
    "evi",
    "lai",
    "greenness",
    "provenance",
    "source_flag",
)

BIOCLIM = [f"bio_{i}" for i in range(1, 20)]
CLIMATE_EXTRA = [
    "aridity_index_hargreaves",
    "aridity_index_penman",
    "pet_hargreaves_annual",
    "pet_penman_annual",
    "srad_annual_mean",
    "vapr_annual_mean",
    "vpd_annual_mean",
    "water_deficit_annual",
    "wind_annual_mean",
]
SOIL_PROPERTIES = [
    "ph_h2o",
    "sand",
    "silt",
    "clay",
    "bdod",
    "soc",
    "socd",
    "nitrogen",
    "cec",
    "cfvo",
    "drainage",
]
SOIL_DEPTHS = ["0-30cm", "30-60cm"]
TERRAIN_LAYERS = [
    "elev_mean.tif",
    "elev_min.tif",
    "elev_max.tif",
    "elev_range.tif",
    "elev_std.tif",
    "slope.tif",
    "steepness_q3.tif",
    "steepness_max.tif",
    "northness.tif",
    "eastness.tif",
    "terrain_vector_strength.tif",
]

XGB_GPU_PARAMS = dict(
    objective="binary:logistic",
    tree_method="hist",
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=10,
    gamma=1.0,
    reg_lambda=5.0,
    reg_alpha=0.5,
    max_bin=256,
    eval_metric="auc",
    n_jobs=0,
    random_state=42,
    verbosity=0,
)


def _assert_trainable_path(path):
    parts = os.path.abspath(path).split(os.sep)
    lowered = [p.lower() for p in parts]
    for token in FORBIDDEN_PATH_TOKENS:
        if any(token in part for part in lowered):
            raise SystemExit(
                f"Refusing predictor {path}: path matches forbidden token {token!r}"
            )
    cursor = os.path.abspath(path)
    for _ in range(len(parts)):
        marker = os.path.join(cursor, DO_NOT_TRAIN_MARKER)
        if os.path.isfile(marker):
            raise SystemExit(
                f"Refusing predictor {path}: {marker} is present"
            )
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        cursor = parent


def collect_predictor_paths():
    """Explicit USA 30s layers. Order is the training contract."""
    layers = []
    for bio in BIOCLIM:
        path = os.path.join(CLIMATE_DIR, f"wc2.1_30s_{bio}.tif")
        layers.append((os.path.basename(path), path))
    for name in CLIMATE_EXTRA:
        path = os.path.join(CLIMATE_DIR, f"{name}.tif")
        layers.append((os.path.basename(path), path))
    for prop in SOIL_PROPERTIES:
        for depth in SOIL_DEPTHS:
            fname = f"{prop}_{depth}_30s.tif"
            layers.append((fname, os.path.join(SOIL_DIR, fname)))
    for fname in TERRAIN_LAYERS:
        layers.append((fname, os.path.join(TOPO_DIR, fname)))

    missing = [p for _, p in layers if not os.path.isfile(p)]
    if missing:
        preview = "\n  ".join(missing[:8])
        raise FileNotFoundError(
            f"{len(missing)} USA 30s predictor(s) missing. First:\n  {preview}"
        )
    for _, path in layers:
        _assert_trainable_path(path)
    return layers


def _read_band(path, expected_shape=None):
    with rasterio.open(path) as src:
        if expected_shape is not None and (src.height, src.width) != expected_shape:
            raise ValueError(
                f"{path} is {src.width}x{src.height}, expected "
                f"{expected_shape[1]}x{expected_shape[0]} (USA 30s grid)"
            )
        band = src.read(1).astype(np.float32, copy=False)
        if src.nodata is not None:
            band = np.where(band == src.nodata, np.nan, band)
        band = np.where(np.abs(band) > 1e30, np.nan, band)
        return band, src.transform, src.height, src.width


class Usa30sPredictors:
    """Load the 1 km stack once (~6 GB float32) and sample by row/col."""

    def __init__(self, layer_paths):
        if not layer_paths:
            raise FileNotFoundError("No USA 30s predictors")
        self.names = [name for name, _ in layer_paths]
        first = layer_paths[0][1]
        band, transform, height, width = _read_band(first)
        self.transform = transform
        self.height = height
        self.width = width
        n_vars = len(layer_paths)
        print(
            f"Loading {n_vars} USA 30s layers into RAM "
            f"({n_vars * height * width * 4 / 1e9:.1f} GB) …"
        )
        self.stack = np.empty((n_vars, height, width), dtype=np.float32)
        self.stack[0] = band
        del band
        valid = np.isfinite(self.stack[0])
        for i, (name, path) in enumerate(layer_paths[1:], start=1):
            band, _, _, _ = _read_band(path, expected_shape=(height, width))
            self.stack[i] = band
            valid &= np.isfinite(band)
            del band
            if (i + 1) % 10 == 0 or i + 1 == n_vars:
                print(f"  {i + 1}/{n_vars}  {name}")
        self.valid_mask = valid
        print(f"  valid land cells = {int(valid.sum()):,}")

    def _rowcol(self, coords):
        coords = np.asarray(coords, dtype=np.float64)
        rows, cols = rasterio.transform.rowcol(
            self.transform, coords[:, 1], coords[:, 0]
        )
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        return rows, cols

    def sample(self, coords):
        coords = np.asarray(coords, dtype=np.float64)
        rows, cols = self._rowcol(coords)
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
        coords = np.asarray(coords, dtype=np.float64)
        rows, cols = self._rowcol(coords)
        rc = np.column_stack([rows, cols])
        _, idx = np.unique(rc, axis=0, return_index=True)
        return coords[np.sort(idx)]


def subsample_coords(coords, n_max, rng):
    if len(coords) <= n_max:
        return coords
    pick = rng.choice(len(coords), size=n_max, replace=False)
    pick.sort()
    return coords[pick]


def _cell_ids(predictors, coords):
    rows, cols = predictors._rowcol(coords)
    return rows.astype(np.int64) * predictors.width + cols.astype(np.int64)


def target_group_absences(
    presence_coords,
    pool_coords,
    n_absences,
    rng,
    predictors,
):
    """Draw background from other-tree cells (target-group), not uniform land.

    pool_coords should already exclude the focal species. Cells that appear
    more than once (several other species) are more likely to be drawn —
    that is the survey-effort weight.
    """
    presence_coords = np.asarray(presence_coords, dtype=np.float64)
    pool_coords = np.asarray(pool_coords, dtype=np.float64)
    if len(pool_coords) == 0:
        raise ValueError("empty target-group pool")

    keep = ~np.isin(
        _cell_ids(predictors, pool_coords),
        _cell_ids(predictors, presence_coords),
    )
    pool = pool_coords[keep]
    if len(pool) == 0:
        raise ValueError("target-group pool collapsed after dropping presences")

    n_take = min(int(n_absences), len(pool))
    pick = rng.choice(len(pool), size=n_take, replace=False)
    return pool[pick]


def spatial_block_ids(coords, n_blocks=N_SPATIAL_BLOCKS, random_state=42):
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


def species_slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def xgb_params_for_n(n_samples, device):
    params = dict(XGB_GPU_PARAMS)
    params["device"] = device
    if n_samples < 400:
        params["min_child_weight"] = 4
        params["gamma"] = 0.5
        params["reg_lambda"] = 3.0
        params["max_depth"] = 4
    elif n_samples < 2000:
        params["min_child_weight"] = 6
        params["gamma"] = 0.8
        params["reg_lambda"] = 4.0
    return params


def _make_classifier(
    n_estimators, n_samples, device, early_stopping_rounds=None,
    scale_pos_weight=None,
):
    params = xgb_params_for_n(n_samples, device)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        **params,
    )


def train_xy_sdm(X, y, coords, device, scale_pos_weight=None, groups=None):
    """5-fold spatial-block CV + final fit on a prepared (X, y)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if groups is None:
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
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        n_usable += 1
        model = _make_classifier(
            n_estimators=MAX_BOOST_ROUNDS,
            n_samples=n_samples,
            device=device,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(
            X[train_idx],
            y_train,
            eval_set=[(X[test_idx], y_test)],
            verbose=False,
        )
        pred = model.predict_proba(X[test_idx])[:, 1]
        auc_scores.append(roc_auc_score(y_test, pred))
        best_ntrees.append(int(model.best_iteration) + 1)

    mean_auc = float(np.mean(auc_scores)) if auc_scores else float("nan")
    if not best_ntrees:
        return None, mean_auc, 0, n_usable, n_folds

    n_trees = max(int(np.median(best_ntrees)), 10)
    final_model = _make_classifier(
        n_estimators=n_trees,
        n_samples=n_samples,
        device=device,
        early_stopping_rounds=None,
        scale_pos_weight=scale_pos_weight,
    )
    final_model.fit(X, y, verbose=False)
    return final_model, mean_auc, n_trees, n_usable, n_folds


def train_species_sdm(presence_coords, absence_coords, predictors, device):
    coords = np.vstack([presence_coords, absence_coords])
    y = np.concatenate(
        [
            np.ones(len(presence_coords), dtype=np.float32),
            np.zeros(len(absence_coords), dtype=np.float32),
        ]
    )
    X = predictors.sample(coords)
    valid = ~np.isnan(X).any(axis=1)
    X, y, coords = X[valid], y[valid], coords[valid]
    return train_xy_sdm(X, y, coords, device)


def load_occurrence_table(path, grid):
    if not os.path.isfile(path):
        raise SystemExit(
            f"No occurrences at {path}. Run: python clean_species_usa_30s.py"
        )
    frame = pd.read_csv(path)
    need = ["species", "latitude", "longitude"]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    if "grid_row" not in frame.columns:
        print(f"{path} is not thinned; thinning onto the USA 30s grid …")
        for extra, fill in (("year", np.nan), ("country", "US"), ("basis", "")):
            if extra not in frame.columns:
                frame[extra] = fill
        frame = thin_to_grid(
            frame[["species", "latitude", "longitude", "year", "country", "basis"]],
            grid,
        )
    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train USA 1 km XGBoost SDMs (does not modify the 10m trainer)."
    )
    parser.add_argument("--occurrences", default=THINNED_CSV)
    parser.add_argument("--species-list", default=SEED_LIST)
    parser.add_argument(
        "--all-species-in-file",
        action="store_true",
        help="train every species present in the occurrence table",
    )
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--max-species",
        type=int,
        default=0,
        help="stop after N saved-or-skipped species (smoke test)",
    )
    parser.add_argument(
        "--full-list",
        action="store_true",
        help=f"use {FULL_LIST} instead of the seed list",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="do not retrain species that already have a JSON in --model-dir",
    )
    parser.add_argument(
        "--shared-universe",
        action="store_true",
        help="batch training: one X for every unique 1 km tree cell "
             "(~4.5e5 rows), binary y per species, scale_pos_weight. "
             "GPU hist has real work. AUC is not comparable to the 1:1 "
             "absence subsample used by the shipped fleet.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="XGBoost / OpenMP threads. 0 = all visible CPUs. Use 8 for "
             "the workstation-vs-MI300X slide.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()
    if args.n_jobs:
        XGB_GPU_PARAMS["n_jobs"] = int(args.n_jobs)
        os.environ["OMP_NUM_THREADS"] = str(int(args.n_jobs))
    species_list_path = FULL_LIST if args.full_list else args.species_list
    os.makedirs(args.model_dir, exist_ok=True)
    metrics_path = os.path.join(args.model_dir, "metrics.csv")
    rng = np.random.default_rng(42)

    grid = read_grid()
    occ = load_occurrence_table(args.occurrences, grid)
    t_after_occ = time.perf_counter()
    if args.all_species_in_file:
        species_order = list(occ["species"].drop_duplicates())
        print(f"Species: every name in {args.occurrences} ({len(species_order)})")
    else:
        species_order = load_species_list(species_list_path)
        print(f"Species list: {species_list_path}  ({len(species_order)} names)")

    predictors = Usa30sPredictors(collect_predictor_paths())
    t_after_rasters = time.perf_counter()
    n_bio = sum(1 for n in predictors.names if "bio_" in n)
    n_soil = sum(1 for n in predictors.names if n.endswith("_30s.tif"))
    n_extra = sum(
        1
        for n in predictors.names
        if n.endswith(".tif")
        and n[:-4] in CLIMATE_EXTRA
    )
    n_topo = len(predictors.names) - n_bio - n_soil - n_extra
    print(
        f"Predictors: {len(predictors.names)}  "
        f"({n_bio} BIO, {n_extra} climate extra, {n_soil} soil, {n_topo} terrain)"
    )
    with open(os.path.join(args.model_dir, "feature_names.txt"), "w") as handle:
        handle.write("\n".join(predictors.names) + "\n")

    pool_all = occ[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    X_univ = None
    univ_coords = None
    univ_cell_ids = None
    univ_groups = None
    if args.shared_universe:
        univ_coords = predictors.unique_pixel_coords(pool_all)
        X_univ = predictors.sample(univ_coords)
        ok = ~np.isnan(X_univ).any(axis=1)
        X_univ, univ_coords = X_univ[ok], univ_coords[ok]
        X_univ = np.ascontiguousarray(X_univ, dtype=np.float32)
        univ_cell_ids = _cell_ids(predictors, univ_coords)
        univ_groups = spatial_block_ids(univ_coords)
        print(
            f"Shared universe: {len(X_univ):,} unique tree cells  "
            f"(one X, binary y per species)"
        )
    metrics = []
    saved_aucs = []
    n_skip = {
        "few_cells": 0,
        "no_folds": 0,
        "few_folds": 0,
        "no_background": 0,
        "no_records": 0,
        "kept": 0,
    }

    prev_by_species = {}
    if args.skip_existing and os.path.isfile(metrics_path):
        prev = pd.read_csv(metrics_path)
        for rec in prev.to_dict("records"):
            prev_by_species[rec["species"]] = rec

    n_done = 0
    n_fit = 0
    t_fit = 0.0
    for species in species_order:
        if args.skip_existing:
            model_path = os.path.join(
                args.model_dir, f"{species_slug(species)}.json"
            )
            prev = prev_by_species.get(species) or {}
            if os.path.isfile(model_path) and prev.get("status") == "saved":
                n_skip["kept"] += 1
                prev = dict(prev)
                prev["model_path"] = relpath(model_path)
                metrics.append(prev)
                if np.isfinite(prev.get("auc", np.nan)):
                    saved_aucs.append(float(prev["auc"]))
                print(f"{species:32s}  KEEP existing {model_path}")
                pd.DataFrame(metrics).to_csv(metrics_path, index=False)
                n_done += 1
                if args.max_species and n_done >= args.max_species:
                    break
                continue

        sp = occ[occ["species"] == species]
        n_records = int(sp["n_records"].sum()) if "n_records" in sp.columns else len(sp)
        if sp.empty:
            n_skip["no_records"] += 1
            metrics.append(
                dict(
                    species=species,
                    n_records=0,
                    n_unique_cells=0,
                    n_absences=0,
                    auc=np.nan,
                    n_trees=0,
                    n_folds_usable=0,
                    n_folds=0,
                    model_path="",
                    status="skip_no_records",
                )
            )
            continue

        presence = predictors.unique_pixel_coords(
            sp[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        )
        n_cells = len(presence)
        if n_cells < MIN_UNIQUE_CELLS:
            n_skip["few_cells"] += 1
            metrics.append(
                dict(
                    species=species,
                    n_records=n_records,
                    n_unique_cells=n_cells,
                    n_absences=0,
                    auc=np.nan,
                    n_trees=0,
                    n_folds_usable=0,
                    n_folds=0,
                    model_path="",
                    status="skip_few_cells",
                )
            )
            print(f"{species:32s}  SKIP <{MIN_UNIQUE_CELLS} cells ({n_cells})")
            n_done += 1
            if args.max_species and n_done >= args.max_species:
                break
            continue

        if args.shared_universe:
            pos_ids = _cell_ids(predictors, presence)
            y_univ = np.isin(univ_cell_ids, pos_ids).astype(np.float32)
            n_pos = int(y_univ.sum())
            n_neg = int(len(y_univ) - n_pos)
            absence = None
            n_absences = n_neg
            spw = n_neg / max(n_pos, 1)
            try:
                t_fit0 = time.perf_counter()
                model, auc, n_trees, n_usable, n_folds = train_xy_sdm(
                    X_univ,
                    y_univ,
                    univ_coords,
                    args.device,
                    scale_pos_weight=spw,
                    groups=univ_groups,
                )
                t_fit += time.perf_counter() - t_fit0
                n_fit += 1
            except Exception as exc:
                print(f"{species:32s}  SKIP train ({type(exc).__name__}: {exc})")
                metrics.append(
                    dict(
                        species=species,
                        n_records=n_records,
                        n_unique_cells=n_cells,
                        n_absences=n_absences,
                        auc=np.nan,
                        n_trees=0,
                        n_folds_usable=0,
                        n_folds=0,
                        model_path="",
                        status="skip_train_error",
                    )
                )
                n_done += 1
                if args.max_species and n_done >= args.max_species:
                    break
                continue
        else:
            presence = subsample_coords(presence, MAX_PRESENCE_CELLS, rng)
            others = occ.loc[
                occ["species"] != species, ["latitude", "longitude"]
            ].to_numpy(dtype=np.float64)
            try:
                absence = target_group_absences(
                    presence,
                    others if len(others) else pool_all,
                    n_absences=min(len(presence), MAX_ABSENCES),
                    rng=rng,
                    predictors=predictors,
                )
            except ValueError as exc:
                n_skip["no_background"] += 1
                print(f"{species:32s}  SKIP background ({exc})")
                metrics.append(
                    dict(
                        species=species,
                        n_records=n_records,
                        n_unique_cells=n_cells,
                        n_absences=0,
                        auc=np.nan,
                        n_trees=0,
                        n_folds_usable=0,
                        n_folds=0,
                        model_path="",
                        status="skip_no_background",
                    )
                )
                n_done += 1
                if args.max_species and n_done >= args.max_species:
                    break
                continue
            n_absences = len(absence)
            try:
                t_fit0 = time.perf_counter()
                model, auc, n_trees, n_usable, n_folds = train_species_sdm(
                    presence, absence, predictors, args.device
                )
                t_fit += time.perf_counter() - t_fit0
                n_fit += 1
            except Exception as exc:
                print(f"{species:32s}  SKIP train ({type(exc).__name__}: {exc})")
                metrics.append(
                    dict(
                        species=species,
                        n_records=n_records,
                        n_unique_cells=n_cells,
                        n_absences=n_absences,
                        auc=np.nan,
                        n_trees=0,
                        n_folds_usable=0,
                        n_folds=0,
                        model_path="",
                        status="skip_train_error",
                    )
                )
                n_done += 1
                if args.max_species and n_done >= args.max_species:
                    break
                continue

        auc_txt = "nan " if np.isnan(auc) else f"{auc:.3f}"
        row = dict(
            species=species,
            n_records=n_records,
            n_unique_cells=n_cells,
            n_absences=n_absences,
            auc=auc,
            n_trees=n_trees,
            n_folds_usable=n_usable,
            n_folds=n_folds,
            model_path="",
            status="",
        )

        if model is None or n_usable == 0:
            n_skip["no_folds"] += 1
            row["status"] = "skip_no_folds"
            print(
                f"{species:32s}  AUC = {auc_txt}  trees = {n_trees:3d}  "
                f"folds = {n_usable}/{n_folds}  SKIP no mixed folds"
            )
        elif n_usable < MIN_USABLE_FOLDS:
            n_skip["few_folds"] += 1
            row["status"] = "skip_few_folds"
            print(
                f"{species:32s}  AUC = {auc_txt}  trees = {n_trees:3d}  "
                f"folds = {n_usable}/{n_folds}  SKIP <{MIN_USABLE_FOLDS} folds"
            )
        else:
            model_path = os.path.join(
                args.model_dir, f"{species_slug(species)}.json"
            )
            model.save_model(model_path)
            saved_aucs.append(auc)
            row["model_path"] = relpath(model_path)
            row["status"] = "saved"
            print(
                f"{species:32s}  AUC = {auc_txt}  trees = {n_trees:3d}  "
                f"folds = {n_usable}/{n_folds}  cells = {n_cells:5d}  saved"
            )

        metrics.append(row)
        pd.DataFrame(metrics).to_csv(metrics_path, index=False)
        n_done += 1
        if args.max_species and n_done >= args.max_species:
            break

    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    n_saved = len(saved_aucs)
    print(
        f"\nSaved {n_saved} models to {args.model_dir}/  "
        f"mean AUC = {np.mean(saved_aucs) if n_saved else float('nan'):.3f}"
    )
    print(
        "Skipped: "
        f"{n_skip['no_records']} no records, "
        f"{n_skip['few_cells']} <{MIN_UNIQUE_CELLS} cells, "
        f"{n_skip['no_folds']} no mixed folds, "
        f"{n_skip['few_folds']} <{MIN_USABLE_FOLDS} folds, "
        f"{n_skip['no_background']} no background"
        + (f", {n_skip['kept']} kept existing" if n_skip.get("kept") else "")
    )
    print(f"Metrics: {metrics_path}")
    print("Global 10-arc-minute models in data/models/ were not changed.")
    elapsed = time.perf_counter() - t0
    rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    t_occ = t_after_occ - t0
    t_rasters = t_after_rasters - t_after_occ
    t_other = elapsed - t_occ - t_rasters - t_fit
    print(
        f"Wall {elapsed:.1f} s  peak RSS {rss_kb} KB  "
        f"device={args.device}  species={n_done}"
    )
    print(
        f"Timing  device={args.device}  n_jobs={XGB_GPU_PARAMS['n_jobs']}  "
        f"occ={t_occ:.1f}s  rasters={t_rasters:.1f}s  fit={t_fit:.1f}s "
        f"({n_fit} trained, 5-fold+final)  other={t_other:.1f}s  "
        f"wall={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
