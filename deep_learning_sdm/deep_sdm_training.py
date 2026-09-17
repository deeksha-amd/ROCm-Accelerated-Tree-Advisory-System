"""
Deep Species Distribution Model — one multi-species network, ROCm/MI210

Objective: given a location, rank which tree species can grow there. This is an
afforestation advisory, so the land it must score is mostly land that has no
trees on it today.

Why a multi-species network instead of per-species models
---------------------------------------------------------
`xgboost_training.py` (the baseline, left untouched for comparison) trains one
independent model per species. A species with 40 records gets a model fitted on
40 records and learns nothing from the 300 other species that share its
climate. That is a structural ceiling, not a tuning problem.

Here a **shared encoder** maps the environmental vector to a latent niche space
and **per-species output heads** read suitability off it. All species train
jointly on one loss, so the encoder is fitted on every record in the dataset and
a rare species inherits the representation that common species paid for. This is
the main reason a network could beat gradient boosting on this problem, and it
is the thing per-species XGBoost cannot do at all.

Why NOT a convolutional or patch-based model
--------------------------------------------
Deliberate. Every raster here is on a 1/6 degree grid — one cell is ~18.5 km, so
a 64x64 patch spans ~1,200 km. There is no meaningful local spatial structure to
convolve over at that scale; a "neighbourhood" is a subcontinent. A point-wise
network fits the data that actually exists. Do not silently switch this to a CNN
without first changing the resolution of the underlying rasters.

What this fixes relative to the baseline
----------------------------------------
1. Soil (22 layers) and terrain are in the feature set, not just climate.
2. Trains on `climate_recent/` (2015-2024), matching the median record year of
   2024, instead of the 1970-2000 window which is 1.28 degC cooler at the
   occurrence points.
3. Accepts externally supplied, effort-weighted target-group background points
   instead of sampling uniformly over land. Uniform background teaches the
   model that "near a city" is good habitat — built-up fraction measured as the
   strongest single presence/background separator in this project (0.246),
   which is survey bias, not ecology. The uniform fallback in `sdm_features.py`
   is labelled as wrong and exists only so the pipeline runs today.
4. Vectorised feature extraction: one raster read per layer, not one per point.
5. No hardcoded device, no hardcoded occurrence file, no hardcoded species count.
6. Reports AUC, TSS and the Boyce index with an explicit threshold-selection
   strategy, rather than AUC on raw scores.
7. Applies `satellite/planting_exclusion_mask_10m.tif` to model output.

What it keeps from the baseline
-------------------------------
The spatial-block cross-validation guarantee, which is the most important thing
the baseline gets right. Points are clustered into geographic blocks and whole
blocks are held out, so a validation point never has a training point from the
same neighbourhood. Random k-fold on spatially autocorrelated occurrence data
reports inflated scores; this does not. scikit-learn is not installed, so
KMeans and the grouped fold split are implemented here in numpy, along with
AUC, TSS and Boyce — following the precedent in
`validate_satellite_alignment.py`.

Usage
-----
    ./venv/bin/python deep_sdm_training.py --smoke
    ./venv/bin/python deep_sdm_training.py --climate-window recent --epochs 60
    ./venv/bin/python deep_sdm_training.py --country IN --predict
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import sdm_features as feat

try:
    import torch
    from torch import nn
except ImportError as error:                                # pragma: no cover
    raise SystemExit(
        "PyTorch is not importable. System pip is blocked by PEP 668, so use a "
        "virtualenv:\n"
        "    python3 -m venv venv --system-site-packages\n"
        "    ./venv/bin/pip install torch "
        "--index-url https://download.pytorch.org/whl/rocm7.1\n"
        "    ./venv/bin/python deep_sdm_training.py --smoke\n"
        f"(import error: {error})")

# ─────────────────────────────────────────────────────
# CONFIG — data
# ─────────────────────────────────────────────────────
PROFILE = feat.DEFAULT_PROFILE    # which grid: see feat.GRID_PROFILES
CLIMATE_WINDOW = feat.DEFAULT_CLIMATE_WINDOW   # "recent" = 2015-2024
MIN_RECORDS_PER_SPECIES = 30      # matches the baseline's cutoff
BACKGROUND_PER_PRESENCE_CELL = 2  # background cells per presence cell
COUNTRY = None                    # ISO code(s); the PoC is single-country
OUTPUT_DIR = "deep_sdm_output"

# ─────────────────────────────────────────────────────
# CONFIG — spatial cross-validation (kept from the baseline)
# ─────────────────────────────────────────────────────
N_SPATIAL_FOLDS = 5
N_SPATIAL_BLOCKS = 10             # geographic clusters; folds hold whole blocks
KMEANS_ITERATIONS = 60
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────
# CONFIG — network and optimiser
# ─────────────────────────────────────────────────────
HIDDEN_SIZES = (256, 128)
# Encoder width is the one capacity knob exposed on the CLI, because it is the
# one with a diagnosed mechanism behind it: the shared encoder is sized once and
# then divided among however many species heads exist. Measured on France —
# going from 69 to 96 species at 1 km pushed the original 69 species *down*
# (AUC 0.747 -> 0.741, improving in only 32 of 69) while the same species rose
# at 18.5 km (0.704 -> 0.741, improving in 62 of 69). The coarse grid is
# data-limited, so extra heads give its encoder more to learn from; the fine
# grid is not, so extra heads just compete for a fixed representation.
DROPOUT = 0.30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
BATCH_SIZE = 4096
EARLY_STOPPING_PATIENCE = 10
# What early stopping watches on the inner validation blocks.
#   "auc"  mean per-species AUC. The default, because the inner blocks are a
#          different region with a different species mix, so the masked BCE
#          there is dominated by which species happen to occur rather than by
#          how well the niche is fitted — watching raw loss stops after one or
#          two epochs for that reason alone.
#   "loss" masked BCE. Cheaper, and the right choice once the occurrence data
#          covers regions evenly.
EARLY_STOPPING_METRIC = "auc"
POS_WEIGHT_CAP = 50.0             # cap on n_background/n_presence per species
DEVICE = "auto"                   # "auto" | "cuda" | "cpu"

# ─────────────────────────────────────────────────────
# CONFIG — evaluation
# ─────────────────────────────────────────────────────
# How the continuous score becomes a yes/no recommendation.
#   "max_tss"          threshold maximising sensitivity+specificity on the
#                      TRAINING fold (never the held-out fold — that leaks)
#   "tenth_percentile" the 10th percentile of training presence scores, i.e.
#                      accept 90% of known presences; the conventional
#                      presence-only choice when absences are unreliable
THRESHOLD_STRATEGY = "max_tss"
BOYCE_WINDOWS = 101
BOYCE_WINDOW_WIDTH = 0.10         # as a fraction of the observed score range
MIN_FOLD_PRESENCES = 5            # held-out presences needed to score a species
# Training presences needed before a species' held-out score means anything.
# Many species here sit inside a single geographic block, so when that block is
# the test fold the species has *no* training presences at all. Its head then
# only ever saw negatives, and scoring it measures nothing — early runs
# produced AUC 0.02 for exactly this reason, which reads as a broken model but
# is really an unevaluable one. Those species-folds are skipped and counted.
MIN_TRAIN_PRESENCES = 5

# ─────────────────────────────────────────────────────
# CONFIG — smoke test
# ─────────────────────────────────────────────────────
SMOKE_SPECIES = 8
SMOKE_EPOCHS = 4
SMOKE_FOLDS = 3
SMOKE_MAX_RECORDS = 40000


def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ─────────────────────────────────────────────────────
# SPATIAL BLOCKS — numpy KMeans (sklearn is not installed)
# ─────────────────────────────────────────────────────
def kmeans(points, n_clusters, seed=RANDOM_SEED,
           iterations=KMEANS_ITERATIONS):
    """k-means++ seeding then Lloyd iterations. Stands in for sklearn.KMeans."""
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype="float64")
    n_clusters = int(min(n_clusters, len(points)))

    centres = np.empty((n_clusters, points.shape[1]))
    centres[0] = points[rng.integers(len(points))]
    nearest = ((points - centres[0]) ** 2).sum(axis=1)
    for k in range(1, n_clusters):
        total = nearest.sum()
        index = (rng.choice(len(points), p=nearest / total) if total > 0
                 else rng.integers(len(points)))
        centres[k] = points[index]
        nearest = np.minimum(nearest, ((points - centres[k]) ** 2).sum(axis=1))

    labels = np.zeros(len(points), dtype="int64")
    for _ in range(iterations):
        distance = (-2.0 * points @ centres.T
                    + (centres ** 2).sum(axis=1)[None, :])
        new_labels = distance.argmin(axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for k in range(n_clusters):
            members = points[labels == k]
            if len(members):
                centres[k] = members.mean(axis=0)
            else:
                # Empty cluster: reseed on the point furthest from any centre
                centres[k] = points[distance.min(axis=1).argmax()]
    return labels


def spatial_block_ids(lat, lon, n_blocks=N_SPATIAL_BLOCKS, seed=RANDOM_SEED):
    """Assign each point to a geographic cluster, same intent as the baseline.

    Longitude is scaled by cos(mean latitude) so that a degree of longitude and
    a degree of latitude cover comparable ground and the blocks come out roughly
    square rather than stretched east-west.
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    mean_lat = np.deg2rad(np.clip(np.nanmean(lat), -89.9, 89.9))
    xy = np.column_stack([lon * np.cos(mean_lat), lat])
    if int(min(n_blocks, len(lat))) < 2:
        return np.zeros(len(lat), dtype="int64")
    return kmeans(xy, n_blocks, seed=seed)


def inner_validation_mask(blocks, fraction=0.2, seed=RANDOM_SEED):
    """Hold out whole blocks for early stopping, sized by point count.

    Early stopping needs data the fit has not seen, and it has to be whole
    blocks for the same reason the outer folds are: a spatially adjacent
    validation set would say "keep going" long past the point where the model
    stopped generalising to new ground. Blocks are drawn in shuffled order
    until the target share of rows is reached, rather than taking the
    highest-numbered cluster labels, which are arbitrary and could hand over
    either half the data or almost none of it.
    """
    unique = np.unique(blocks)
    if len(unique) < 2:
        mask = np.zeros(len(blocks), bool)
        mask[::5] = True                      # no blocks to spare: stride it
        return mask

    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    target = fraction * len(blocks)
    chosen, held = [], 0
    for block in order:
        if held >= target and chosen:
            break
        chosen.append(block)
        held += int((blocks == block).sum())
    if len(chosen) == len(unique):            # never hand over everything
        chosen = chosen[:-1]
    return np.isin(blocks, chosen)


def spatial_cv_splits(blocks, n_splits=N_SPATIAL_FOLDS):
    """Partition whole blocks into folds. No block is ever split across folds.

    This is the guarantee that matters: a held-out point has no training point
    from its own neighbourhood, so the reported score is not inflated by spatial
    autocorrelation. Blocks are assigned largest-first to whichever fold
    currently holds the fewest points, which keeps folds comparable in size
    without ever breaking a block apart.
    """
    unique, counts = np.unique(blocks, return_counts=True)
    n_splits = int(min(n_splits, len(unique)))
    if n_splits < 2:
        return None

    fold_of_block = {}
    load = np.zeros(n_splits, dtype="int64")
    for block, count in sorted(zip(unique, counts), key=lambda p: -p[1]):
        fold = int(load.argmin())
        fold_of_block[block] = fold
        load[fold] += count

    fold_index = np.array([fold_of_block[b] for b in blocks])
    return [(np.flatnonzero(fold_index != f), np.flatnonzero(fold_index == f))
            for f in range(n_splits)]


# ─────────────────────────────────────────────────────
# METRICS — numpy only (scipy and sklearn are absent)
# ─────────────────────────────────────────────────────
def average_ranks(values):
    """Ranks with ties averaged, vectorised. Stands in for scipy.rankdata."""
    values = np.asarray(values, dtype="float64")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = values.size
    start = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    stop = np.r_[start[1:], n]
    mean_rank = (start + stop - 1) / 2.0 + 1.0
    ranks = np.empty(n, dtype="float64")
    ranks[order] = np.repeat(mean_rank, stop - start)
    return ranks


def auc(positive, negative):
    """Probability a random presence outscores a random background point."""
    positive = np.asarray(positive, dtype="float64")
    negative = np.asarray(negative, dtype="float64")
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    ranks = average_ranks(np.concatenate([positive, negative]))
    total = ranks[:positive.size].sum()
    return float((total - positive.size * (positive.size + 1) / 2.0)
                 / (positive.size * negative.size))


def auc_per_species(scores, y, mask, unused=None):
    """AUC for every species at once, from a [rows, species] score matrix.

    Ordinal rather than tie-averaged ranks: sigmoid outputs almost never tie,
    and this runs once per epoch for early stopping, so speed wins. The
    reported metrics in `evaluate_fold` use the tie-aware `auc()`.
    """
    del unused
    n_rows, n_species = scores.shape
    order = np.argsort(scores, axis=0, kind="stable")
    ranks = np.empty((n_rows, n_species), dtype="float64")
    ladder = np.broadcast_to(np.arange(1, n_rows + 1, dtype="float64")[:, None],
                             (n_rows, n_species))
    np.put_along_axis(ranks, order, np.ascontiguousarray(ladder), axis=0)

    # A negative for species s is any counted cell where s was not recorded.
    # It cannot be "any row that is not a presence row": with target-group
    # background, 2,357 of 2,373 background cells also hold a presence of some
    # other species, so a row-level presence flag would throw nearly every
    # negative away.
    counted = mask > 0
    positive = counted & (y > 0)
    negative = counted & (y <= 0)
    n_positive = positive.sum(axis=0).astype("float64")
    n_negative = negative.sum(axis=0).astype("float64")

    rank_sum = (ranks * positive).sum(axis=0)
    denominator = n_positive * n_negative
    out = np.full(n_species, np.nan)
    ok = denominator > 0
    out[ok] = ((rank_sum[ok] - n_positive[ok] * (n_positive[ok] + 1) / 2.0)
               / denominator[ok])
    return out, n_positive


def spearman(a, b):
    """Rank correlation, used by the Boyce index."""
    a, b = average_ranks(a), average_ranks(b)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denominator) if denominator > 0 \
        else float("nan")


def confusion_at(labels, scores, threshold):
    """Sensitivity, specificity, TSS and omission rate at one threshold."""
    labels = np.asarray(labels).astype(bool)
    predicted = np.asarray(scores) >= threshold
    true_positive = int((predicted & labels).sum())
    false_negative = int((~predicted & labels).sum())
    true_negative = int((~predicted & ~labels).sum())
    false_positive = int((predicted & ~labels).sum())

    sensitivity = (true_positive / (true_positive + false_negative)
                   if true_positive + false_negative else float("nan"))
    specificity = (true_negative / (true_negative + false_positive)
                   if true_negative + false_positive else float("nan"))
    return {"threshold": float(threshold),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tss": sensitivity + specificity - 1.0,
            "omission_rate": 1.0 - sensitivity}


def max_tss_threshold(labels, scores):
    """Threshold maximising sensitivity+specificity-1, computed in one sweep.

    TSS = sensitivity + specificity - 1, so it is not inflated by the
    presence/background ratio the way plain accuracy is. Unlike AUC it needs a
    threshold, which is why the strategy is chosen explicitly and always fitted
    on the training fold.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype="float64")
    n_positive, n_negative = int(labels.sum()), int((~labels).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan"), float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(~sorted_labels)
    # Only consider cut points between distinct scores
    last_of_run = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tss = (true_positive / n_positive) - (false_positive / n_negative)
    tss[~last_of_run] = -np.inf
    best = int(tss.argmax())
    return float(tss[best]), float(sorted_scores[best])


def tenth_percentile_threshold(presence_scores):
    """Accept 90% of known presences. Conventional for presence-only data."""
    if len(presence_scores) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(presence_scores, dtype="float64"),
                               10.0))


def boyce_index(presence_scores, available_scores,
                n_windows=BOYCE_WINDOWS, width=BOYCE_WINDOW_WIDTH):
    """Continuous Boyce index (Hirzel et al. 2006).

    Slides a window across the suitability range and measures the
    predicted-to-expected ratio F = (share of presences) / (share of available
    land) in each one, then rank-correlates F against suitability. A good model
    gives +1: presences pile up in the high-suitability windows faster than
    available land does. It needs no absence data, which is why the reference
    study asks for it — AUC against pseudo-absences is sensitive to where the
    background was drawn, and this is not.

    `available_scores` are the background points of the fold, standing in for
    the suitability distribution of the whole study area.

    WARNING — Boyce runs *opposite* to AUC across the range-restriction
    gradient, so a run-level Boyce mean is inflated by widespread species.
    Measured on the 96 French species: species occupying under 250 cells score
    AUC 0.898 but Boyce 0.466, while species on over 500 cells score AUC 0.712
    and Boyce 0.891. The reason is structural, not a bug. For a species present
    across nearly all of France almost all land genuinely is suitable, so the
    predicted-to-expected ratio climbs with suitability almost by construction
    and Boyce rewards a model that AUC correctly says cannot discriminate.
    Read Boyce per species, alongside range size, and never as a headline on
    its own. The cleaner summary statistic is the correlation between AUC and
    presence-cell count, which was -0.408 with monotonic bins.
    """
    presence_scores = np.asarray(presence_scores, dtype="float64")
    available_scores = np.asarray(available_scores, dtype="float64")
    if presence_scores.size == 0 or available_scores.size < 2:
        return float("nan")

    low, high = float(available_scores.min()), float(available_scores.max())
    if not np.isfinite(low) or high <= low:
        return float("nan")

    span = (high - low) * width
    starts = np.linspace(low, high - span, n_windows)
    midpoints, ratios = [], []
    for start in starts:
        stop = start + span
        expected = np.mean((available_scores >= start) &
                           (available_scores <= stop))
        if expected <= 0:
            continue
        predicted = np.mean((presence_scores >= start) &
                            (presence_scores <= stop))
        midpoints.append(start + span / 2.0)
        ratios.append(predicted / expected)

    if len(ratios) < 3:
        return float("nan")
    return spearman(np.array(midpoints), np.array(ratios))


# ─────────────────────────────────────────────────────
# DATASET ASSEMBLY
# ─────────────────────────────────────────────────────
def build_dataset(args):
    """Occurrences + target-group background -> features, labels, loss mask."""
    divider("1. DATA")
    profile = args.profile
    grid = feat.load_grid(profile)
    catalog = feat.build_catalog(profile, climate_window=args.climate_window)
    feat.describe_catalog(catalog, profile)
    feat.check_grid(catalog, grid)
    feat.report_unused_layers(catalog, profile)
    climate_dir = feat.GRID_PROFILES[profile]["climate"][args.climate_window][0]
    print(f"climate window   {args.climate_window} -> {climate_dir}")
    print(f"grid             {grid['width']}x{grid['height']} cells, "
          f"{grid['cell'] * 111:.1f} km")

    extent = tuple(args.bbox) if args.bbox else \
        feat.GRID_PROFILES[profile]["extent"]
    occurrences, source = feat.load_occurrences(
        path=args.occurrences, country=args.country, extent=extent,
        min_records=args.min_records)
    if len(occurrences) == 0:
        raise SystemExit("No occurrence records survived the filters")

    # Thin at the grid the model actually uses, not at whatever grid the file
    # happened to be thinned on.
    if args.thin:
        occurrences = feat.thin_to_grid(occurrences, grid)

    # Species selection, always read from the data and never hardcoded.
    counts = occurrences["species"].value_counts()
    species = list(counts.index)
    if args.max_species:
        species = species[:args.max_species]
        occurrences = occurrences[occurrences["species"].isin(species)]
    if args.max_records and len(occurrences) > args.max_records:
        occurrences = occurrences.sample(n=args.max_records,
                                         random_state=RANDOM_SEED)
    species_index = {name: i for i, name in enumerate(species)}
    print(f"species          {len(species):,} kept "
          f"(>= {args.min_records} records; {counts.iloc[0]:,} most, "
          f"{counts[species].iloc[-1]:,} fewest)")

    # Background count is set against distinct presence *cells*, not records:
    # the rows of the design matrix are cells, and a species-rich cell is still
    # one row.
    row, col, _ = feat.lonlat_to_rowcol(occurrences["latitude"].to_numpy(),
                                        occurrences["longitude"].to_numpy(),
                                        grid)
    n_presence_cells = len(set(zip(row.tolist(), col.tolist())))
    n_background = int(n_presence_cells * args.background_ratio)
    spec = feat.GRID_PROFILES[profile]
    # --background-from-effort ignores the supplied point file and draws
    # `n_background` cells from the effort surface instead. Its purpose is to
    # hold the negative-pool *size* fixed while the effort surface underneath
    # changes: a supplied file comes with a count someone else chose, and the
    # size of the negative pool moves AUC on its own (a 31.7% change in it was
    # worth -0.006 AUC here), so comparing two files of different sizes
    # confounds background quality with background count.
    background_path = None if args.background_from_effort else (
        args.background or spec.get("background"))
    back_lat, back_lon, background_kind = feat.load_background(
        grid, n_points=n_background, n_presence_cells=n_presence_cells,
        path=background_path, extent=extent,
        effort_raster=spec["effort_raster"])

    lat, lon, y, mask, has_record, surveyed = feat.build_cell_table(
        occurrences, species_index, back_lat, back_lon, grid)
    print(f"cells            {len(lat):,} distinct cells: "
          f"{int(has_record.sum()):,} with a record, "
          f"{int(surveyed.sum()):,} target-group surveyed, "
          f"{int((has_record & surveyed).sum()):,} both")
    print(f"labels           {int(y.sum()):,} species-cell presences, "
          f"{int(mask.sum()):,} of {mask.size:,} cells x species counted "
          f"({100 * mask.sum() / mask.size:.1f}%)")

    if args.other_presence_as_background:
        # Every cell holding any record becomes a negative for species not
        # recorded there. Stronger signal, weaker justification: GBIF is
        # presence-only, so a missing record is not an absence unless the
        # target group was actually surveyed.
        mask = np.maximum(mask, has_record[:, None].astype("float32"))
        print(f"labels           widened to {int(mask.sum()):,} counted "
              f"(--other-presence-as-background)")

    features, inside = feat.sample_points(catalog, lat, lon, grid,
                                          desc="extracting predictors")
    groups = feat.feature_groups(catalog)
    indicators, indicator_names = feat.missingness_columns(features, groups)
    names = feat.feature_names(catalog) + indicator_names
    features = np.concatenate([features, indicators], axis=1)

    keep = inside & feat.drop_empty_rows(features[:, :len(groups)])
    # A row with no counted species contributes nothing to the loss.
    keep &= (mask.sum(axis=1) > 0)
    print(f"rows             {int(keep.sum()):,} kept of {len(keep):,} "
          f"(dropped: off-grid, or >"
          f"{feat.MAX_MISSING_FRACTION:.0%} of predictors missing)")

    data = {
        "features": features[keep],
        "names": names,
        "y": y[keep],
        "mask": mask[keep],
        "lat": lat[keep],
        "lon": lon[keep],
        "surveyed": surveyed[keep],
        "has_record": has_record[keep],
        "species": species,
        "catalog": catalog,
        "grid": grid,
        "profile": profile,
        "extent": extent,
        "background_kind": background_kind,
        "background_path": background_path or spec["effort_raster"],
        "effort_raster": spec["effort_raster"],
        "occurrence_source": source,
    }
    print(f"features         {data['features'].shape[1]} columns "
          f"({len(groups)} predictors + {len(indicator_names)} missingness "
          f"indicators)")
    return data


# ─────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────
class MultiSpeciesSDM(nn.Module):
    """Shared environmental encoder, one output head per species.

    Independent heads were chosen over a learned species embedding. With a
    fixed, closed species list the final linear layer already *is* a species
    embedding — row s of its weight matrix is species s's vector in the
    encoder's latent niche space, learned end to end, with no extra machinery
    and no second hyperparameter. An explicit embedding table only starts to
    pay for itself when it conditions the encoder (so species can bend the
    shared representation) or when scores are needed for species absent from
    training via a taxonomic prior. Neither is required here, and both add
    capacity that 293 species with a median of a few hundred records cannot
    support. Revisit if the species list becomes open-ended.
    """

    def __init__(self, n_features, n_species, hidden=HIDDEN_SIZES,
                 dropout=DROPOUT):
        super().__init__()
        layers, width = [], n_features
        for size in hidden:
            layers += [nn.Linear(width, size), nn.LayerNorm(size),
                       nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.Linear(width, n_species)

    def forward(self, x):
        return self.heads(self.encoder(x))     # logits [batch, n_species]


def pick_device(requested=DEVICE):
    """Resolve the device. Never hardcoded — the baseline hardcodes "cuda".

    PyTorch's ROCm build keeps the `cuda` device namespace, so on this MI210
    `torch.cuda.is_available()` is True and `torch.device("cuda")` is the
    AMD GPU. Confusing, but correct.
    """
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        version = getattr(torch.version, "hip", None) or torch.version.cuda
        backend = "ROCm/HIP" if getattr(torch.version, "hip", None) else "CUDA"
        print(f"device           {device} -> {name}  ({backend} {version})")
    else:
        print(f"device           {device}  (no GPU visible to torch)")
    return device


def positive_weights(y, mask, cap=POS_WEIGHT_CAP):
    """Per-species negative/positive ratio, so rare species are not drowned.

    Background points outnumber presences for every species, and by wildly
    different factors across species. BCEWithLogitsLoss(pos_weight=n_neg/n_pos)
    rescales each species' positive term so that all species contribute
    comparable gradient. Capped, because a species with three presences would
    otherwise get a weight in the thousands and dominate the shared encoder.
    """
    counted = mask > 0
    n_positive = (y * counted).sum(axis=0)
    n_negative = ((1.0 - y) * counted).sum(axis=0)
    weight = np.where(n_positive > 0, n_negative / np.maximum(n_positive, 1), 1.0)
    return np.clip(weight, 1.0, cap).astype("float32")


def train_network(train_x, train_y, train_mask, val_x, val_y, val_mask,
                  device, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  patience=EARLY_STOPPING_PATIENCE,
                  metric=EARLY_STOPPING_METRIC, hidden=None, quiet=False):
    """Fit one network. `val_*` is an inner split used only for early stopping.

    The inner validation set is carved out of the *training* blocks, never from
    the held-out fold, so early stopping cannot peek at the test data.
    """
    model = MultiSpeciesSDM(train_x.shape[1], train_y.shape[1],
                            hidden=hidden or HIDDEN_SIZES).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                          T_max=max(epochs, 1))
    weight = torch.as_tensor(positive_weights(train_y, train_mask),
                             device=device)

    tensors = {}
    for key, array in (("tx", train_x), ("ty", train_y), ("tm", train_mask),
                       ("vx", val_x), ("vy", val_y), ("vm", val_mask)):
        tensors[key] = torch.as_tensor(np.ascontiguousarray(array),
                                       dtype=torch.float32, device=device)

    def masked_loss(logits, target, mask):
        raw = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=weight, reduction="none")
        total = mask.sum()
        return (raw * mask).sum() / torch.clamp(total, min=1.0)

    # Early stopping maximises a score, so loss is negated to share one path.
    best_score, best_state, best_epoch, stale = -float("inf"), None, -1, 0
    best_loss = float("nan")
    n_train = len(train_x)
    generator = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    bar = None if quiet else tqdm(range(epochs), desc="training", unit="epoch")

    for epoch in (range(epochs) if quiet else bar):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        running = 0.0
        for start in range(0, n_train, batch_size):
            index = order[start:start + batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = masked_loss(model(tensors["tx"][index]),
                               tensors["ty"][index], tensors["tm"][index])
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(index)
        schedule.step()

        model.eval()
        with torch.no_grad():
            logits = model(tensors["vx"])
            val_loss = float(masked_loss(logits, tensors["vy"], tensors["vm"]))
            val_auc = float("nan")
            if metric == "auc":
                scores, n_positive = auc_per_species(
                    torch.sigmoid(logits).cpu().numpy(), val_y, val_mask)
                scorable = n_positive >= MIN_FOLD_PRESENCES
                if scorable.any():
                    val_auc = float(np.nanmean(scores[scorable]))

        score = val_auc if metric == "auc" and np.isfinite(val_auc) \
            else -val_loss
        if bar is not None:
            bar.set_postfix(train=f"{running / n_train:.4f}",
                            val_loss=f"{val_loss:.4f}",
                            val_auc=f"{val_auc:.3f}")

        if score > best_score + 1e-5:
            best_score, best_loss, best_epoch, stale = (score, val_loss,
                                                        epoch, 0)
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch + 1,
                   "best_val_loss": best_loss,
                   "best_val_score": best_score,
                   "early_stopping_metric": metric}


def predict(model, features, device, batch_size=BATCH_SIZE * 4):
    """Suitability in [0, 1] for every species at every row."""
    out = np.empty((len(features), model.heads.out_features), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            block = np.ascontiguousarray(features[start:start + batch_size])
            tensor = torch.as_tensor(block, dtype=torch.float32, device=device)
            out[start:start + batch_size] = torch.sigmoid(
                model(tensor)).cpu().numpy()
    return out


# ─────────────────────────────────────────────────────
# SPATIAL-BLOCK CROSS-VALIDATION
# ─────────────────────────────────────────────────────
def evaluate_fold(scores, y, mask, species, strategy, train_presences):
    """Per-species AUC, TSS, Boyce, sensitivity and omission on one fold.

    Returns (rows, skipped) where `skipped` counts species that could not be
    scored, split by reason.
    """
    rows = []
    skipped = {"no_training_presence": 0, "too_few_held_out": 0}
    for s, name in enumerate(species):
        counted = mask[:, s] > 0
        presence = counted & (y[:, s] > 0)
        background = counted & (y[:, s] <= 0)        # shared negatives
        n_presence = int(presence.sum())
        if train_presences[s] < MIN_TRAIN_PRESENCES:
            skipped["no_training_presence"] += 1
            continue
        if n_presence < MIN_FOLD_PRESENCES or background.sum() < 1:
            skipped["too_few_held_out"] += 1
            continue

        presence_scores = scores[presence, s]
        background_scores = scores[background, s]
        labels = np.r_[np.ones(n_presence, bool),
                       np.zeros(background.sum(), bool)]
        combined = np.r_[presence_scores, background_scores]

        # `tss` is at the threshold carried over from the training fold, which
        # is what deployment would actually use. `tss_max` re-optimises the
        # threshold on the held-out fold itself — not a usable number, but it
        # separates "the model cannot discriminate here" from "it can, and the
        # threshold simply did not transfer across regions", which is a common
        # and distinct failure under spatial-block CV.
        best_tss, _ = max_tss_threshold(labels, combined)
        rows.append({
            "species": name,
            "n_presence": n_presence,
            "n_train_presence": int(train_presences[s]),
            "n_background": int(background.sum()),
            "auc": auc(presence_scores, background_scores),
            "boyce": boyce_index(presence_scores, background_scores),
            **confusion_at(labels, combined, strategy["threshold"][s]),
            "tss_max": best_tss,
        })
    return rows, skipped


def choose_thresholds(scores, y, mask, n_species, strategy):
    """Pick one threshold per species on the TRAINING fold only."""
    thresholds = np.full(n_species, 0.5, dtype="float64")
    for s in range(n_species):
        counted = mask[:, s] > 0
        presence = counted & (y[:, s] > 0)
        background = counted & (y[:, s] <= 0)
        if presence.sum() == 0 or background.sum() == 0:
            continue
        if strategy == "tenth_percentile":
            thresholds[s] = tenth_percentile_threshold(scores[presence, s])
        else:
            labels = np.r_[np.ones(int(presence.sum()), bool),
                           np.zeros(int(background.sum()), bool)]
            combined = np.r_[scores[presence, s], scores[background, s]]
            _, thresholds[s] = max_tss_threshold(labels, combined)
    return {"strategy": strategy, "threshold": thresholds}


def cross_validate(data, args, device):
    """Spatial-block CV over the whole multi-species problem at once."""
    divider("2. SPATIAL-BLOCK CROSS-VALIDATION")
    blocks = spatial_block_ids(data["lat"], data["lon"],
                               n_blocks=args.blocks)
    splits = spatial_cv_splits(blocks, n_splits=args.folds)
    if splits is None:
        raise SystemExit("Not enough spatial blocks for cross-validation")

    sizes = np.bincount(blocks)
    print(f"blocks           {len(sizes)} geographic clusters, "
          f"{sizes.min():,}-{sizes.max():,} points each")
    print(f"folds            {len(splits)}, whole blocks held out "
          f"(no train point shares a neighbourhood with a test point)")
    print(f"threshold        {args.threshold_strategy}, fitted on training "
          f"folds only")

    per_fold = []
    skipped_total = {"no_training_presence": 0, "too_few_held_out": 0}
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train_x_raw = data["features"][train_index]
        test_x_raw = data["features"][test_index]

        # Inner split for early stopping: hold out whole training blocks again.
        inner_val = inner_validation_mask(blocks[train_index],
                                          seed=RANDOM_SEED + fold)

        scaler = feat.FeatureScaler().fit(train_x_raw[~inner_val])
        fit_x = scaler.transform(train_x_raw[~inner_val])
        val_x = scaler.transform(train_x_raw[inner_val])
        test_x = scaler.transform(test_x_raw)

        train_y, train_mask = data["y"][train_index], data["mask"][train_index]
        model, history = train_network(
            fit_x, train_y[~inner_val], train_mask[~inner_val],
            val_x, train_y[inner_val], train_mask[inner_val],
            device, epochs=args.epochs, batch_size=args.batch_size,
            hidden=args.hidden)

        train_scores = predict(model, scaler.transform(train_x_raw), device)
        strategy = choose_thresholds(train_scores, train_y, train_mask,
                                     len(data["species"]),
                                     args.threshold_strategy)

        # How many presences each species actually had to learn from. A species
        # with none here cannot be evaluated, only extrapolated to.
        train_presences = ((train_y > 0) & (train_mask > 0)).sum(axis=0)

        test_scores = predict(model, test_x, device)
        rows, skipped = evaluate_fold(
            test_scores, data["y"][test_index], data["mask"][test_index],
            data["species"], strategy, train_presences)
        for row in rows:
            row["fold"] = fold
        per_fold.extend(rows)
        skipped_total["no_training_presence"] += skipped["no_training_presence"]
        skipped_total["too_few_held_out"] += skipped["too_few_held_out"]

        tail = (f"(stopped at epoch {history['best_epoch']}, "
                f"val loss {history['best_val_loss']:.4f})")
        if rows:
            summary = pd.DataFrame(rows)
            print(f"fold {fold}: {len(rows):>4d} species scored  "
                  f"AUC {summary['auc'].mean():.3f}  "
                  f"TSS {summary['tss'].mean():.3f}  "
                  f"Boyce {summary['boyce'].mean():.3f}  {tail}")
        else:
            print(f"fold {fold}: no species had >= {MIN_FOLD_PRESENCES} "
                  f"held-out presences  {tail}")

    total = sum(skipped_total.values()) + len(per_fold)
    print(f"\nspecies-folds    {len(per_fold):,} scored of {total:,}")
    print(f"   skipped: {skipped_total['no_training_presence']:,} had "
          f"< {MIN_TRAIN_PRESENCES} training presences (the species sits "
          f"inside the held-out")
    print(f"            region, so there is nothing to evaluate), "
          f"{skipped_total['too_few_held_out']:,} had "
          f"< {MIN_FOLD_PRESENCES} held-out presences")
    return pd.DataFrame(per_fold), blocks, skipped_total


def report(results, data, args, skipped=None):
    """Species-level and overall metrics."""
    divider("3. RESULTS")
    if results.empty:
        print("No species had enough held-out presences to score. With the "
              "current occurrence data this is expected.")
        return results

    metrics = ["auc", "tss", "tss_max", "boyce", "sensitivity",
               "omission_rate"]
    by_species = (results.groupby("species")[metrics + ["n_presence"]]
                  .mean().sort_values("auc", ascending=False))

    print(f"{'metric':<16s}{'mean':>8s}{'median':>9s}{'min':>8s}{'max':>8s}")
    for metric in metrics:
        column = results[metric].dropna()
        if column.empty:
            continue
        print(f"{metric:<16s}{column.mean():>8.3f}{column.median():>9.3f}"
              f"{column.min():>8.3f}{column.max():>8.3f}")

    print(f"\nspecies scored   {len(by_species):,} of {len(data['species']):,}")
    print(f"folds            {results['fold'].nunique()}")

    # Range size is what most of the per-species spread is about, and the two
    # threshold-free metrics disagree about it on purpose — see boyce_index().
    if "n_presence" in by_species and len(by_species) > 5:
        size = by_species["n_presence"]
        print(f"\nrange-size effect (AUC vs presence cells, "
              f"r = {by_species['auc'].corr(size):+.3f})")
        edges = [(0, size.quantile(0.33)), (size.quantile(0.33),
                 size.quantile(0.67)), (size.quantile(0.67), np.inf)]
        for low, high in edges:
            block = by_species[(size >= low) & (size < high)]
            if len(block):
                print(f"   {low:>7.0f}-{high:>7.0f} cells  n={len(block):>3d}  "
                      f"AUC {block['auc'].mean():.3f}  "
                      f"Boyce {block['boyce'].mean():.3f}")
        print("   Boyce runs OPPOSITE to AUC here: a species on nearly all of "
              "France has")
        print("   almost all land genuinely suitable, so its P/E ratio climbs "
              "by construction.")
        print("   The run-level Boyce mean above is therefore inflated by "
              "widespread species.")
    def listing(label, block):
        print(f"\n{label}")
        for name, row in block.iterrows():
            print(f"   {name[:34]:<34s} AUC {row['auc']:.3f}  "
                  f"TSS {row['tss']:.3f}  Boyce {row['boyce']:.3f}  "
                  f"n={row['n_presence']:.0f}")

    show = min(5, len(by_species))
    listing(f"best {show} by AUC", by_species.head(show))
    if len(by_species) > 2 * show:
        listing(f"worst {show} by AUC", by_species.tail(show))

    os.makedirs(args.output_dir, exist_ok=True)
    fold_path = os.path.join(args.output_dir, "cv_metrics_by_fold.csv")
    species_path = os.path.join(args.output_dir, "cv_metrics_by_species.csv")
    results.to_csv(fold_path, index=False)
    by_species.to_csv(species_path)
    run = {
        "profile": args.profile,
        "grid": f"{data['grid']['width']}x{data['grid']['height']} @ "
                f"{data['grid']['cell']:.10f} deg",
        "climate_window": args.climate_window,
        "occurrence_source": data["occurrence_source"],
        "background_kind": data["background_kind"],
        "background_path": data["background_path"],
        "effort_raster": data["effort_raster"],
        "background_is_correct_sampler":
            data["background_kind"] != "uniform_fallback",
        "n_species_total": len(data["species"]),
        "n_species_scored": int(len(by_species)),
        "n_rows": int(len(data["features"])),
        "n_feature_columns": int(data["features"].shape[1]),
        "predictor_layers": [layer["path"] for layer in data["catalog"]],
        "vegetation_layers_used": 0,
        "threshold_strategy": args.threshold_strategy,
        "hidden_sizes": list(args.hidden or HIDDEN_SIZES),
        "batch_size": int(args.batch_size),
        "species_folds_skipped": skipped or {},
        "folds": int(args.folds),
        "blocks": int(args.blocks),
        "epochs": int(args.epochs),
        "extent": data["extent"],
        "mean": {m: float(results[m].mean()) for m in metrics},
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w") as handle:
        json.dump(run, handle, indent=2, default=str)
    print(f"\nwritten          {fold_path}")
    print(f"                 {species_path}")
    print(f"                 {args.output_dir}/run_summary.json")
    return by_species


# ─────────────────────────────────────────────────────
# FINAL MODEL + PREDICTION + EXCLUSION MASK
# ─────────────────────────────────────────────────────
def fit_final_model(data, args, device):
    """Refit on everything, holding back whole blocks only for early stopping."""
    divider("4. FINAL MODEL (all blocks)")
    blocks = spatial_block_ids(data["lat"], data["lon"], n_blocks=args.blocks)
    inner_val = inner_validation_mask(blocks)

    scaler = feat.FeatureScaler().fit(data["features"][~inner_val])
    model, history = train_network(
        scaler.transform(data["features"][~inner_val]),
        data["y"][~inner_val], data["mask"][~inner_val],
        scaler.transform(data["features"][inner_val]),
        data["y"][inner_val], data["mask"][inner_val],
        device, epochs=args.epochs, batch_size=args.batch_size,
        hidden=args.hidden)

    scores = predict(model, scaler.transform(data["features"]), device)
    strategy = choose_thresholds(scores, data["y"], data["mask"],
                                 len(data["species"]),
                                 args.threshold_strategy)
    print(f"trained          stopped at epoch {history['best_epoch']}, "
          f"val loss {history['best_val_loss']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = os.path.join(args.output_dir, "deep_sdm.pt")
    torch.save({"state_dict": model.state_dict(),
                "species": data["species"],
                "feature_names": data["names"],
                "thresholds": strategy["threshold"],
                "threshold_strategy": strategy["strategy"],
                "scaler": {"median": scaler.median, "mean": scaler.mean,
                           "scale": scaler.scale},
                "climate_window": args.climate_window,
                "hidden_sizes": list(args.hidden or HIDDEN_SIZES)},
               checkpoint)
    print(f"checkpoint       {checkpoint}")
    return model, scaler, strategy


def predict_extent(model, scaler, strategy, data, args, device,
                   species_subset=None):
    """Score every cell in the extent, then apply the exclusion mask.

    The mask is bit-coded. Bits 1|2|4|8 (ocean, inland water, permanent snow
    and ice, built up) are hard exclusions — nothing can be planted there, so
    those cells are blanked. Bits 16 (closed-canopy forest) and 32 (cropland)
    are advisory only and are reported, not removed: closed-canopy forest is
    not unsuitable land, it is land that has already proven it grows trees and
    simply does not need planting, and cropland is a land-use decision rather
    than an ecological one.
    """
    divider("5. PREDICTION + EXCLUSION MASK")
    import rasterio

    grid, catalog = data["grid"], data["catalog"]
    extent = data["extent"]
    chosen = species_subset or data["species"][:args.predict_species]
    columns = [data["species"].index(name) for name in chosen]

    r0, r1, c0, c1 = feat.extent_to_window(extent, grid)
    height, width = r1 - r0, c1 - c0
    surface = np.full((len(columns), height, width), np.nan, dtype="float32")
    n_predictors = len(feat.feature_groups(catalog))
    print(f"extent           rows {r0}-{r1}, cols {c0}-{c1} "
          f"({height}x{width} cells)")

    for rows, cols, block in feat.iter_grid_blocks(
            catalog, grid, extent=extent, desc="scoring extent"):
        indicators, _ = feat.missingness_columns(
            block, feat.feature_groups(catalog))
        block = np.concatenate([block, indicators], axis=1)
        usable = feat.drop_empty_rows(block[:, :n_predictors])
        if not usable.any():
            continue
        scores = predict(model, scaler.transform(block[usable]), device)
        local_rows = rows[usable] - r0
        local_cols = cols[usable] - c0
        for i, column in enumerate(columns):
            surface[i, local_rows, local_cols] = scores[:, column]

    mask, _ = feat.read_exclusion_mask(grid, extent=extent)
    os.makedirs(args.output_dir, exist_ok=True)
    transform = rasterio.windows.transform(
        rasterio.windows.Window(c0, r0, width, height), grid["transform"])

    land = np.isfinite(surface[0])
    for i, name in enumerate(chosen):
        filtered, advisory = feat.apply_exclusions(surface[i], mask)
        binary = np.where(np.isnan(filtered), np.nan,
                          (filtered >= strategy["threshold"][columns[i]])
                          .astype("float32"))
        slug = name.lower().replace(" ", "_").replace("/", "_")
        path = os.path.join(args.output_dir, f"suitability_{slug}.tif")
        with rasterio.open(path, "w", driver="GTiff", height=height,
                           width=width, count=2, dtype="float32",
                           crs=grid["crs"], transform=transform,
                           nodata=np.nan, compress="deflate", tiled=True) as dst:
            dst.write(filtered, 1)
            dst.write(binary, 2)
            dst.set_band_description(1, "suitability 0-1, hard exclusions blank")
            dst.set_band_description(
                2, f"recommended at {strategy['strategy']} threshold "
                   f"{strategy['threshold'][columns[i]]:.3f}")
            dst.update_tags(
                species=name,
                threshold_strategy=strategy["strategy"],
                threshold=f"{strategy['threshold'][columns[i]]:.6f}",
                hard_exclusions="ocean|inland_water|permanent_snow_ice|built_up",
                advisory_flags="closed_canopy_forest, cropland — reported, "
                               "not removed",
                predictors="climate + soil + terrain; no vegetation layer")
        kept = np.isfinite(filtered)
        print(f"   {name[:30]:<30s} {int(kept.sum()):>8,} cells scored "
              f"({100 * kept.sum() / max(land.sum(), 1):5.1f}% of land kept), "
              f"{int(np.nansum(binary)):>8,} recommended")
        for flag, flagged in advisory.items():
            print(f"      advisory {flag:<22s} "
                  f"{int((flagged & kept).sum()):>8,} cells")
        print(f"      written {path}")
    return surface


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Multi-species deep SDM: climate + soil + terrain, "
                    "spatial-block CV, ROCm GPU")
    parser.add_argument("--profile", default=PROFILE,
                        choices=sorted(feat.GRID_PROFILES),
                        help="which grid and country to train on")
    parser.add_argument("--climate-window", default=CLIMATE_WINDOW,
                        help="which climate window to train on")
    parser.add_argument("--no-thin", dest="thin", action="store_false",
                        help="skip re-thinning occurrences to the run's grid")
    parser.add_argument("--occurrences", default=None,
                        help="occurrence CSV; default picks the best available")
    parser.add_argument("--background", default=None,
                        help="background CSV; default is the profile's own")
    parser.add_argument("--background-from-effort", action="store_true",
                        help="ignore the point file and draw from the effort "
                             "surface, so the background count is controlled")
    parser.add_argument("--country", default=COUNTRY, nargs="*",
                        help="ISO code(s) for the single-country PoC")
    parser.add_argument("--bbox", type=float, nargs=4, default=None,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--min-records", type=int,
                        default=MIN_RECORDS_PER_SPECIES)
    parser.add_argument("--max-species", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--background-ratio", type=float,
                        default=BACKGROUND_PER_PRESENCE_CELL)
    parser.add_argument("--folds", type=int, default=N_SPATIAL_FOLDS)
    parser.add_argument("--blocks", type=int, default=N_SPATIAL_BLOCKS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--hidden", type=int, nargs="+", default=None,
                        metavar="WIDTH",
                        help="shared-encoder layer widths, e.g. --hidden 512 "
                             f"256 (default {list(HIDDEN_SIZES)})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="lower this if the GPU is shared and contends")
    parser.add_argument("--threshold-strategy", default=THRESHOLD_STRATEGY,
                        choices=["max_tss", "tenth_percentile"])
    parser.add_argument("--other-presence-as-background", action="store_true",
                        help="treat a cell holding species A as an absence of "
                             "species B (default: mask it, since GBIF is "
                             "presence-only)")
    parser.add_argument("--device", default=DEVICE,
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--predict", action="store_true",
                        help="score the extent and write masked GeoTIFFs")
    parser.add_argument("--predict-species", type=int, default=3,
                        help="how many species to write rasters for")
    parser.add_argument("--no-final-model", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end run: few species, few epochs")

    args = parser.parse_args(argv)
    if args.country == []:
        args.country = None
    if args.smoke:
        args.max_species = args.max_species or SMOKE_SPECIES
        args.max_records = args.max_records or SMOKE_MAX_RECORDS
        args.epochs = SMOKE_EPOCHS if args.epochs == EPOCHS else args.epochs
        args.folds = min(args.folds, SMOKE_FOLDS)
        args.predict = True
        args.predict_species = min(args.predict_species, 2)
        args.output_dir = os.path.join(args.output_dir, "smoke")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.time()

    divider("DEEP SDM  -  multi-species network over climate + soil + terrain")
    if args.smoke:
        print("SMOKE TEST: proving the plumbing, not the science. The current "
              "occurrence data")
        print("            cannot support conclusions — treat every number "
              "below as a shape check.")
    device = pick_device(args.device)
    torch.manual_seed(RANDOM_SEED)

    data = build_dataset(args)
    if data["background_kind"] == "uniform_fallback":
        print("\nWARNING: background points are the uniform fallback. Any "
              "metric below is")
        print("         inflated by survey bias and is not a real skill "
              "estimate.")

    results, _, skipped = cross_validate(data, args, device)
    report(results, data, args, skipped=skipped)

    if not args.no_final_model:
        model, scaler, strategy = fit_final_model(data, args, device)
        if args.predict:
            predict_extent(model, scaler, strategy, data, args, device)

    divider("DONE")
    print(f"elapsed          {time.time() - started:.1f}s")
    print(f"output           {args.output_dir}/")


if __name__ == "__main__":
    main()
