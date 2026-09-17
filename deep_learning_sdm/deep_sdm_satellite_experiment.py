"""
Controlled experiment: what does satellite data do as an SDM predictor?

The question
------------
This project excluded every vegetation layer from the predictor set on measured
grounds, not on principle. Globally: vegetation layers alone, with no climate,
soil or terrain at all, reached cross-validated AUC 0.791 against 0.711 for the
whole genuine environmental set — but that +0.080 lead collapsed to +0.001 on
cleared farmland, and tree-cover fraction separated occurrences from background
in the same direction for 90% of 40 species against 52% for annual mean
temperature. Read together those say the lead is the model reading off where
trees already are, which is no use to an advisory whose job is to rank land
where trees are *not*.

That is an empirical claim, and the France 1 km data is much better than the
data it was measured on. So it gets re-measured here rather than assumed.

Why this is not a replacement model
-----------------------------------
`deep_sdm_training.py` stays the production path and stays satellite-free. This
script is a measurement instrument: it trains the same network on four feature
sets and compares them, and the number it exists to produce is not the headline
AUC but the **difference between the headline and the score on treeless land**.

A headline AUC that goes up when satellite layers are added is the *expected*
result under the circularity hypothesis, not evidence against it. Tree cover
predicts tree occurrence almost by definition, and most of the held-out
presences sit on land that already has trees. The test is the stratified
evaluation.

The four feature sets
---------------------
    env                identical to production: climate + soil + terrain
    env_satellite      the same, plus all 17 satellite layers
    satellite          the 17 satellite layers alone, no climate/soil/terrain
    env_soilmoisture   production plus soil moisture only — microwave-derived,
                       not greenness-derived, and the one layer previously
                       judged defensible

How "identical in every other respect" is guaranteed
----------------------------------------------------
Not by running the script four times with different flags, which would make the
occurrence file, the background draw and the row-drop decisions four separate
events that a concurrent agent could change underneath. Instead:

  * `deep_sdm_training.build_dataset()` is called **once**, unmodified, so the
    environment-only arm is byte-identical in setup to production.
  * The satellite columns are sampled once at those same cells and concatenated.
  * Each feature set is then a **column subset of one matrix**. Same rows, same
    labels, same loss mask, same spatial blocks, same folds, same background
    draw, same occurrence snapshot — by construction, not by convention.
  * Rows are dropped on environmental coverage only, the production rule, so
    poorer satellite coverage cannot quietly change which cells are evaluated.
  * Missingness indicators are recomputed per subset, so a feature set is never
    told about the coverage of a group it does not contain.
  * `torch.manual_seed(RANDOM_SEED + fold)` at the start of every fold of every
    arm, so results do not depend on the order the arms happen to run in.

The strata
----------
Evaluation is repeated on subsets of the held-out cells, using the same fitted
model and the same training-fold threshold:

    all           every held-out cell — the headline, and the least informative
    low_tree      <= 10% tree cover: land that is not already forest
    cropland      >= 50% cropland: cleared farmland, the afforestation target
    high_tree     >= 50% tree cover: the contrast, where circularity should pay

Strata are defined from `landcover_tree_frac` and `landcover_cropland_frac`.
Used this way they are evaluation metadata, not predictors — the env arm never
sees them, and they are read from the same block for all four arms.

Usage
-----
    ./venv/bin/python deep_sdm_satellite_experiment.py --smoke
    ./venv/bin/python deep_sdm_satellite_experiment.py
"""

import argparse
import json
import os
import time
import types

import numpy as np
import pandas as pd

import deep_sdm_training as dst
import sdm_features as feat

try:
    import torch
except ImportError as error:                               # pragma: no cover
    raise SystemExit(f"PyTorch is not importable: {error}")

# ─────────────────────────────────────────────────────
# CONFIG — the satellite block under test
# ─────────────────────────────────────────────────────
# Written by `satellite_predictor_block.py`. Group names carry the source, so
# each source contributes its own missingness indicator: land cover reaches
# 69.2% of grid cells, NDVI 66.4% and soil moisture 64.9%, and lumping them
# would tell the network "some satellite layer was unknown here" instead of
# which one.
SATELLITE_BLOCKS = {"fra_30s": "satellite_experiment/FRA_30s",
                    "fra_10m": "satellite_experiment/FRA_10m"}
SATELLITE_LAYERS = [
    ("landcover_tree_frac", "satellite_landcover"),
    ("landcover_tree_broadleaf_frac", "satellite_landcover"),
    ("landcover_tree_needleleaf_frac", "satellite_landcover"),
    ("landcover_shrub_frac", "satellite_landcover"),
    ("landcover_grass_natural_frac", "satellite_landcover"),
    ("landcover_cropland_frac", "satellite_landcover"),
    ("landcover_builtup_frac", "satellite_landcover"),
    ("landcover_bare_frac", "satellite_landcover"),
    ("landcover_water_frac", "satellite_landcover"),
    ("landcover_snowice_frac", "satellite_landcover"),
    ("landcover_land_frac", "satellite_landcover"),
    ("ndvi_observed_mean", "satellite_ndvi"),
    ("ndvi_peak", "satellite_ndvi"),
    ("ndvi_growing_season_mean", "satellite_ndvi"),
    ("ndvi_seasonal_amplitude", "satellite_ndvi"),
    ("soilmoisture_mean", "satellite_soilmoisture"),
    ("soilmoisture_amplitude", "satellite_soilmoisture"),
]
SOIL_MOISTURE_GROUP = "satellite_soilmoisture"
ENVIRONMENT_GROUPS = ("climate", "soil", "terrain")

# Three subsets of the block, because "does satellite data help" is not one
# question. The strictly circular layers are the ones that measure standing
# woody vegetation; the availability layers describe what the land is being
# used for, which this project already accepted as legitimate for filtering
# output; soil moisture is microwave and not a greenness proxy at all.
CIRCULAR_LAYERS = ["landcover_tree_frac", "landcover_tree_broadleaf_frac",
                   "landcover_tree_needleleaf_frac", "ndvi_observed_mean",
                   "ndvi_peak", "ndvi_growing_season_mean",
                   "ndvi_seasonal_amplitude"]
AVAILABILITY_LAYERS = ["landcover_shrub_frac", "landcover_grass_natural_frac",
                       "landcover_cropland_frac", "landcover_builtup_frac",
                       "landcover_bare_frac", "landcover_water_frac",
                       "landcover_snowice_frac", "landcover_land_frac"]
SOIL_MOISTURE_LAYERS = ["soilmoisture_mean", "soilmoisture_amplitude"]
ALL_SATELLITE = CIRCULAR_LAYERS + AVAILABILITY_LAYERS + SOIL_MOISTURE_LAYERS

# ─────────────────────────────────────────────────────
# CONFIG — the arms
# ─────────────────────────────────────────────────────
# The first four are the experiment as specified. The last two split the
# satellite block in half, because a gain that survives on treeless land still
# has to be attributed: if it comes from tree cover and NDVI it is the circular
# signal being cleverer than expected, and if it comes from cropland, built-up
# and water it is land use, which is a different and much more defensible
# thing. Reporting only "satellite helps" would leave that unanswered.
FEATURE_SETS = {
    "env": {
        "environment": True, "satellite": [],
        "label": "environment only (production baseline)"},
    "env_satellite": {
        "environment": True, "satellite": ALL_SATELLITE,
        "label": "environment + all satellite"},
    "satellite": {
        "environment": False, "satellite": ALL_SATELLITE,
        "label": "satellite only (no climate, soil or terrain)"},
    "env_soilmoisture": {
        "environment": True, "satellite": SOIL_MOISTURE_LAYERS,
        "label": "environment + soil moisture only"},
    "env_circular": {
        "environment": True, "satellite": CIRCULAR_LAYERS,
        "label": "environment + tree cover and NDVI only (the circular half)"},
    "env_availability": {
        "environment": True, "satellite": AVAILABILITY_LAYERS,
        "label": "environment + non-tree land cover only (land use)"},
    # Three single-layer probes, because "land use helps" is still too coarse
    # to act on. Built-up fraction measured as the strongest single
    # presence/background separator earlier in this project (0.246) and was
    # called survey bias, not ecology; the effort-weighted background corrects
    # bias only *between* 18.5 km cells, so at 1 km a built-up cell is still
    # more likely to hold a record than its neighbour. If the land-use gain is
    # mostly this layer, the gain is bias absorption and raising AUC with it
    # would make the advisory worse.
    "env_builtup": {
        "environment": True, "satellite": ["landcover_builtup_frac"],
        "label": "environment + built-up fraction only (bias probe)"},
    "env_cropland": {
        "environment": True, "satellite": ["landcover_cropland_frac"],
        "label": "environment + cropland fraction only (land-use probe)"},
    "env_availability_no_builtup": {
        "environment": True,
        "satellite": [l for l in AVAILABILITY_LAYERS
                      if l != "landcover_builtup_frac"],
        "label": "environment + land use except built-up"},
}
REFERENCE_SET = "env"        # every delta is measured against this arm

# ─────────────────────────────────────────────────────
# CONFIG — tree-cover strata for the held-out evaluation
# ─────────────────────────────────────────────────────
# Thresholds are on percent of the non-sea part of the cell, the units the
# land-cover block is written in.
#
# `no_tree` exists because `low_tree` turns out to be a weaker test than it
# sounds: 10% of a 1 km cell is still about 8 hectares of trees, so within that
# stratum tree-cover fraction retains real range and can still be read as "are
# there trees here". 2% is about the finest distinction the 300 m source
# supports at 1 km (one native pixel in nine) and is the strictest version of
# the question this data can answer.
NO_TREE_MAX_PCT = 2.0
LOW_TREE_MAX_PCT = 10.0
CROPLAND_MIN_PCT = 50.0
HIGH_TREE_MIN_PCT = 50.0
STRATA = ["all", "low_tree", "no_tree", "cropland", "high_tree"]
STRATUM_LABEL = {
    "all": "all held-out cells",
    "low_tree": f"tree cover <= {LOW_TREE_MAX_PCT:.0f}%",
    "no_tree": f"tree cover <= {NO_TREE_MAX_PCT:.0f}% (strictest)",
    "cropland": f"cropland >= {CROPLAND_MIN_PCT:.0f}%",
    "high_tree": f"tree cover >= {HIGH_TREE_MIN_PCT:.0f}% (contrast)",
}

# ─────────────────────────────────────────────────────
# CONFIG — inputs and output
# ─────────────────────────────────────────────────────
# Snapshots, not the live files. Another agent may replace the occurrence file
# with a larger French species set while this runs, and a four-way comparison
# where two arms saw different species would be worthless.
OCCURRENCE_SNAPSHOT = "satellite_experiment/occurrences_snapshot.csv"
BACKGROUND_SNAPSHOT = "satellite_experiment/background_snapshot.csv"
OUTPUT_DIR = "satellite_experiment/results"
OVERRIDE_REASON = ("deep_sdm_satellite_experiment.py — re-measuring the "
                   "vegetation-circularity finding on the France 1 km data")

divider = dst.divider


# ─────────────────────────────────────────────────────
# DATASET — built once, sliced four ways
# ─────────────────────────────────────────────────────
def satellite_catalog(profile):
    """The satellite layers, or a hard stop naming the script that builds them."""
    directory = SATELLITE_BLOCKS.get(profile)
    if not directory or not os.path.isdir(directory):
        raise SystemExit(
            f"No satellite block for profile {profile!r} at {directory!r}. "
            f"Build it first:\n"
            f"    ./venv/bin/python satellite_predictor_block.py "
            f"--profile {profile}")
    catalog = []
    for name, group in SATELLITE_LAYERS:
        path = os.path.join(directory, f"{name}.tif")
        if not os.path.exists(path):
            raise SystemExit(f"satellite layer missing: {path}")
        catalog.append({"name": name, "group": group, "kind": "linear",
                        "path": path})
    return catalog


def production_args(args):
    """The argument namespace `deep_sdm_training.build_dataset` expects.

    Deliberately assembled here rather than reusing that script's parser, so
    that every value the production dataset depends on is visible in one place
    and pinned to the snapshots.
    """
    return types.SimpleNamespace(
        profile=args.profile,
        climate_window=args.climate_window,
        thin=True,
        occurrences=args.occurrences,
        background=args.background,
        country=None,
        bbox=None,
        min_records=args.min_records,
        max_species=args.max_species,
        max_records=args.max_records,
        background_ratio=args.background_ratio,
        other_presence_as_background=False,
        threshold_strategy=args.threshold_strategy,
        folds=args.folds,
        blocks=args.blocks,
        epochs=args.epochs,
        output_dir=args.output_dir)


def build_union_dataset(args):
    """Production dataset, plus satellite columns sampled at the same cells."""
    divider("1. DATA  -  production dataset, then satellite columns bolted on")
    print("The environment arm is set up by deep_sdm_training.build_dataset() "
          "unmodified, with")
    print("the vegetation guard fully armed, so it cannot drift from "
          "production.\n")
    data = dst.build_dataset(production_args(args))

    n_env = len(feat.feature_groups(data["catalog"]))
    env_features = data["features"][:, :n_env]
    env_groups = np.array(feat.feature_groups(data["catalog"]))
    env_names = feat.feature_names(data["catalog"])

    divider("2. SATELLITE COLUMNS  -  the guard is opened here, loudly")
    feat.allow_vegetation_predictors_experiment(OVERRIDE_REASON)
    catalog = satellite_catalog(args.profile)
    feat.assert_no_vegetation_layers(catalog)
    feat.check_grid(catalog, data["grid"])
    sat_features, sat_inside = feat.sample_points(
        catalog, data["lat"], data["lon"], data["grid"],
        desc="extracting satellite")
    sat_groups = np.array([layer["group"] for layer in catalog])
    sat_names = [layer["name"] for layer in catalog]
    if not sat_inside.all():
        raise SystemExit("satellite sampling put cells off the grid, which "
                         "cannot happen for cells the production dataset "
                         "already accepted")

    for group in sorted(set(sat_groups.tolist())):
        block = sat_features[:, sat_groups == group]
        coverage = 100 * np.isfinite(block).all(axis=1).mean()
        print(f"   {group:<24s} {int((sat_groups == group).sum()):>2d} layers, "
              f"{coverage:5.1f}% of the {len(block):,} modelled cells complete")

    data["env_features"] = env_features
    data["env_groups"] = env_groups
    data["env_names"] = env_names
    data["sat_features"] = sat_features
    data["sat_groups"] = sat_groups
    data["sat_names"] = sat_names
    data["satellite_catalog"] = catalog
    data["tree_frac"] = sat_features[:, sat_names.index("landcover_tree_frac")]
    data["cropland_frac"] = sat_features[
        :, sat_names.index("landcover_cropland_frac")]
    return data


def feature_matrix(data, spec):
    """Columns for one feature set, with its own missingness indicators.

    Indicators are recomputed from the selected columns only. A feature set is
    never handed an indicator for a group it does not contain: "soil was
    unknown here" is a free hint about coastline and sea that the satellite-only
    arm has no business receiving.
    """
    blocks, groups, names = [], [], []
    if spec["environment"]:
        blocks.append(data["env_features"])
        groups.append(data["env_groups"])
        names.extend(data["env_names"])
    if spec["satellite"]:
        # Selected by layer name, in the block's own order, so that two arms
        # asking for the same layer always get the same column.
        select = np.array([n in set(spec["satellite"])
                           for n in data["sat_names"]])
        missing = set(spec["satellite"]) - set(data["sat_names"])
        if missing:
            raise SystemExit(f"feature set asks for unknown layers: {missing}")
        blocks.append(data["sat_features"][:, select])
        groups.append(data["sat_groups"][select])
        names.extend([n for n, keep in zip(data["sat_names"], select) if keep])
    if not blocks:
        raise SystemExit("a feature set with no features")

    predictors = np.concatenate(blocks, axis=1)
    group_labels = np.concatenate(groups)
    indicators, indicator_names = feat.missingness_columns(predictors,
                                                            group_labels)
    return (np.concatenate([predictors, indicators], axis=1).astype("float32"),
            names + indicator_names, len(names))


# ─────────────────────────────────────────────────────
# STRATA
# ─────────────────────────────────────────────────────
def stratum_masks(data):
    """Boolean row masks for each tree-cover stratum, over all modelled cells."""
    tree, crop = data["tree_frac"], data["cropland_frac"]
    known = np.isfinite(tree) & np.isfinite(crop)
    masks = {
        "all": np.ones(len(tree), dtype=bool),
        "low_tree": known & (tree <= LOW_TREE_MAX_PCT),
        "no_tree": known & (tree <= NO_TREE_MAX_PCT),
        "cropland": known & (crop >= CROPLAND_MIN_PCT),
        "high_tree": known & (tree >= HIGH_TREE_MIN_PCT),
    }
    print(f"land cover known  {int(known.sum()):,} of {len(tree):,} modelled "
          f"cells ({100 * known.mean():.1f}%)")
    presence = data["y"].any(axis=1)
    for name in STRATA:
        mask = masks[name]
        print(f"   {name:<12s} {int(mask.sum()):>8,} cells "
              f"({100 * mask.mean():5.1f}%), "
              f"{int((mask & presence).sum()):>7,} hold a record   "
              f"{STRATUM_LABEL[name]}")
    return masks


# ─────────────────────────────────────────────────────
# ONE ARM OF THE EXPERIMENT
# ─────────────────────────────────────────────────────
def cross_validate_arm(name, features, data, blocks, splits, masks, args,
                       device):
    """Spatial-block CV for one feature set, scored on every stratum.

    The fold loop is the production one — same scaler fitted on the training
    fold only, same inner validation blocks for early stopping, same threshold
    chosen on the training fold. The only addition is that `evaluate_fold` is
    called once per stratum on row subsets of the same held-out predictions.
    """
    rows = []
    epochs_used = []
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        # Per-fold rather than once per process, so an arm's numbers do not
        # depend on which arms ran before it.
        torch.manual_seed(dst.RANDOM_SEED + fold)

        train_raw = features[train_index]
        inner_val = dst.inner_validation_mask(blocks[train_index],
                                              seed=dst.RANDOM_SEED + fold)
        scaler = feat.FeatureScaler().fit(train_raw[~inner_val])
        train_y = data["y"][train_index]
        train_mask = data["mask"][train_index]

        model, history = dst.train_network(
            scaler.transform(train_raw[~inner_val]),
            train_y[~inner_val], train_mask[~inner_val],
            scaler.transform(train_raw[inner_val]),
            train_y[inner_val], train_mask[inner_val],
            device, epochs=args.epochs, quiet=True)
        epochs_used.append(history["best_epoch"])

        train_scores = dst.predict(model, scaler.transform(train_raw), device)
        strategy = dst.choose_thresholds(train_scores, train_y, train_mask,
                                         len(data["species"]),
                                         args.threshold_strategy)
        train_presences = ((train_y > 0) & (train_mask > 0)).sum(axis=0)
        test_scores = dst.predict(model, scaler.transform(features[test_index]),
                                  device)

        for stratum in STRATA:
            keep = masks[stratum][test_index]
            if not keep.any():
                continue
            scored, _ = dst.evaluate_fold(
                test_scores[keep], data["y"][test_index][keep],
                data["mask"][test_index][keep], data["species"], strategy,
                train_presences)
            for row in scored:
                row["fold"] = fold
                row["stratum"] = stratum
                row["feature_set"] = name
            rows.append(pd.DataFrame(scored))

        frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        headline = frame[(frame["fold"] == fold) & (frame["stratum"] == "all")] \
            if not frame.empty else frame
        print(f"   fold {fold}: {len(headline):>3d} species  "
              f"AUC {headline['auc'].mean():.3f}  "
              f"TSS {headline['tss'].mean():.3f}  "
              f"Boyce {headline['boyce'].mean():.3f}  "
              f"(stopped at epoch {history['best_epoch']})")
        del model

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, epochs_used


# ─────────────────────────────────────────────────────
# THE SPECIES-DIRECTION AGREEMENT TEST
# ─────────────────────────────────────────────────────
def direction_agreement(data, variables):
    """Does a variable separate occurrences from background the same way for
    most species?

    A genuine niche axis should not: some species are warm-adapted and some are
    cold-adapted, so the sign of (presence mean - background mean) ought to
    split roughly evenly. A variable that answers "are there trees here?"
    separates *every* species in the same direction, because every tree species
    is more likely to be recorded where trees are. Globally tree-cover fraction
    agreed for 90% of 40 species and annual mean temperature for 52%; this
    repeats the test on France.

    Background is the surveyed cells where the species was not recorded, i.e.
    the negatives the model itself is trained against.
    """
    counted = data["mask"] > 0
    positive = counted & (data["y"] > 0)
    negative = counted & (data["y"] <= 0)
    out = []
    for label, values in variables.items():
        values = np.asarray(values, dtype="float64")
        known = np.isfinite(values)
        signs, effects = [], []
        for s in range(len(data["species"])):
            present = positive[:, s] & known
            absent = negative[:, s] & known
            if present.sum() < dst.MIN_FOLD_PRESENCES or absent.sum() < 2:
                continue
            difference = values[present].mean() - values[absent].mean()
            spread = np.sqrt((values[present].var() + values[absent].var()) / 2)
            signs.append(np.sign(difference))
            effects.append(difference / spread if spread > 0 else 0.0)
        if not signs:
            continue
        signs = np.array(signs)
        majority = max((signs > 0).mean(), (signs < 0).mean())
        out.append({"variable": label,
                    "species_tested": len(signs),
                    "share_positive": float((signs > 0).mean()),
                    "majority_direction_share": float(majority),
                    "mean_abs_effect": float(np.mean(np.abs(effects)))})
    return pd.DataFrame(out).sort_values("majority_direction_share",
                                          ascending=False)


# ─────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────
METRICS = ["auc", "tss", "boyce"]


def comparison_table(results):
    """mean AUC / TSS / Boyce per feature set per stratum, plus species-folds."""
    grouped = (results.groupby(["stratum", "feature_set"])[METRICS]
               .mean().reset_index())
    counts = (results.groupby(["stratum", "feature_set"]).size()
              .rename("species_folds").reset_index())
    return grouped.merge(counts, on=["stratum", "feature_set"])


def paired_deltas(results, reference=REFERENCE_SET):
    """Per-species-fold differences against the reference arm.

    Comparing arm means hides that the arms are scored on the same species in
    the same folds. Pairing on (stratum, fold, species) removes the
    species-composition variance, which is much larger than the effect being
    measured, and makes a sign test available.
    """
    keys = ["stratum", "fold", "species"]
    base = results[results["feature_set"] == reference].set_index(keys)
    out = []
    for name in results["feature_set"].unique():
        if name == reference:
            continue
        arm = results[results["feature_set"] == name].set_index(keys)
        shared = base.index.intersection(arm.index)
        for metric in METRICS:
            difference = (arm.loc[shared, metric]
                          - base.loc[shared, metric]).dropna()
            if difference.empty:
                continue
            wins = int((difference > 0).sum())
            out.append({
                "feature_set": name, "metric": metric,
                "stratum": "", "n": len(difference),
                "mean_delta": float(difference.mean()),
                "sd_delta": float(difference.std(ddof=1)),
                "share_improved": wins / len(difference)})
        # Same again, split by stratum, which is the part that answers the
        # question.
        for stratum in STRATA:
            in_stratum = [k for k in shared if k[0] == stratum]
            if not in_stratum:
                continue
            index = pd.MultiIndex.from_tuples(in_stratum, names=keys)
            for metric in METRICS:
                difference = (arm.loc[index, metric]
                              - base.loc[index, metric]).dropna()
                if difference.empty:
                    continue
                out.append({
                    "feature_set": name, "metric": metric,
                    "stratum": stratum, "n": len(difference),
                    "mean_delta": float(difference.mean()),
                    "sd_delta": float(difference.std(ddof=1)),
                    "share_improved": float((difference > 0).mean())})
    frame = pd.DataFrame(out)
    return frame[frame["stratum"] != ""].reset_index(drop=True)


def presence_background_separation(data, masks):
    """Pooled mean of each layer at presence cells vs background-only cells.

    The cheapest and most direct diagnostic there is, and the one that explains
    the model results: a layer the network can profitably use to rank presences
    above background is a layer whose two means differ. Reported overall and
    again inside the strictest treeless stratum, because a layer that separates
    there is a layer that will lift the stratified AUC without contributing any
    ecology.
    """
    presence = data["has_record"]
    background = data["surveyed"] & ~data["has_record"]
    variables = {name: data["sat_features"][:, i]
                 for i, name in enumerate(data["sat_names"])}
    for name in ("bio_1", "bio_12", "elevation"):
        if name in data["env_names"]:
            variables[f"ENV {name}"] = data["env_features"][
                :, data["env_names"].index(name)]

    strict = masks["no_tree"]
    rows = []
    for label, values in variables.items():
        known = np.isfinite(values)
        p, b = presence & known, background & known
        if p.sum() < 2 or b.sum() < 2:
            continue
        spread = np.sqrt((values[p].var() + values[b].var()) / 2) or np.nan
        row = {"layer": label,
               "presence_mean": float(values[p].mean()),
               "background_mean": float(values[b].mean()),
               "difference": float(values[p].mean() - values[b].mean()),
               "standardised": float((values[p].mean() - values[b].mean())
                                     / spread)}
        ps, bs = p & strict, b & strict
        if ps.sum() >= 2 and bs.sum() >= 2:
            inner = np.sqrt((values[ps].var() + values[bs].var()) / 2) or np.nan
            row["no_tree_difference"] = float(values[ps].mean()
                                              - values[bs].mean())
            row["no_tree_standardised"] = float(
                (values[ps].mean() - values[bs].mean()) / inner)
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.reindex(frame["standardised"].abs()
                         .sort_values(ascending=False).index)


def print_headline(table, deltas):
    divider("RESULTS  -  feature sets (rows) x tree-cover strata (columns)")
    print("Feature sets are rows because the comparison of interest runs "
          "across a row: an")
    print("advantage that is real ecology holds its size from 'all' through "
          "to 'no_tree'.\n")
    for metric in METRICS:
        print(f"{metric.upper()}")
        print(f"   {'feature set':<20s}", end="")
        for stratum in STRATA:
            print(f"{stratum:>11s}", end="")
        print()
        for name in FEATURE_SETS:
            if (table["feature_set"] == name).sum() == 0:
                continue
            print(f"   {name:<20s}", end="")
            for stratum in STRATA:
                cell = table[(table["feature_set"] == name)
                             & (table["stratum"] == stratum)]
                value = f"{cell[metric].iloc[0]:.3f}" if not cell.empty else "-"
                print(f"{value:>11s}", end="")
            print()
        print(f"   {'species-folds':<20s}", end="")
        for stratum in STRATA:
            block = table[table["stratum"] == stratum]
            n = int(block["species_folds"].max()) if not block.empty else 0
            print(f"{n:>11,d}", end="")
        print("\n")

    divider(f"PAIRED DELTA against '{REFERENCE_SET}', same species and folds")
    print("Positive means the arm beat the environment-only baseline on the "
          "same species-folds.")
    print("share = fraction of species-folds improved; a real effect is "
          "well above 0.5.\n")
    for name in FEATURE_SETS:
        if name == REFERENCE_SET:
            continue
        print(f"{name}  ({FEATURE_SETS[name]['label']})")
        print(f"   {'stratum':<12s}{'metric':>8s}{'delta':>10s}"
              f"{'sd':>8s}{'share':>8s}{'n':>8s}")
        block = deltas[deltas["feature_set"] == name]
        for stratum in STRATA:
            for metric in METRICS:
                row = block[(block["stratum"] == stratum)
                            & (block["metric"] == metric)]
                if row.empty:
                    continue
                row = row.iloc[0]
                print(f"   {stratum:<12s}{metric:>8s}"
                      f"{row['mean_delta']:>+10.3f}{row['sd_delta']:>8.3f}"
                      f"{row['share_improved']:>8.2f}{int(row['n']):>8,d}")
        print()


def verdict(deltas, direction, separation):
    """State what the numbers say about circularity, in the report itself."""
    divider("VERDICT")
    auc = deltas[deltas["metric"] == "auc"]
    arm = auc[auc["feature_set"] == "env_satellite"].set_index("stratum")
    if arm.empty:
        print("no paired AUC deltas; nothing to conclude")
        return {}
    headline = float(arm.loc["all", "mean_delta"])
    conclusions = {"auc_delta_all": headline}
    for stratum in STRATA[1:]:
        if stratum in arm.index:
            conclusions[f"auc_delta_{stratum}"] = float(
                arm.loc[stratum, "mean_delta"])

    print("env + satellite vs env, mean paired AUC delta:")
    for key, value in conclusions.items():
        print(f"   {key:<24s} {value:>+8.3f}")

    treeless = [conclusions.get(f"auc_delta_{s}")
                for s in ("low_tree", "no_tree", "cropland")]
    retained = [v / headline for v in treeless
                if v is not None and headline > 1e-6]
    if headline <= 0.005:
        summary = ("satellite layers add nothing even on the headline, so "
                   "there is no advantage to explain away")
    elif retained and max(retained) < 0.35:
        summary = ("CONFIRMS the circularity finding: the advantage is largely "
                   "gone on land without existing tree cover")
    elif retained and min(retained) > 0.7:
        summary = ("REFUTES the circularity finding on this data: the "
                   "advantage persists on treeless land")
    else:
        summary = ("AMBIGUOUS: the advantage shrinks on treeless land but does "
                   "not vanish")
    print(f"\n{summary}")
    conclusions["summary"] = summary

    # Attribution. A surviving advantage still has to be traced to a layer
    # group before it can be called defensible.
    print("\nwhere the advantage comes from (mean paired AUC delta vs env):")
    print(f"   {'arm':<20s}", end="")
    for stratum in STRATA:
        print(f"{stratum:>11s}", end="")
    print()
    for name in ("env_circular", "env_availability",
                 "env_availability_no_builtup", "env_builtup", "env_cropland",
                 "env_soilmoisture", "env_satellite"):
        block = auc[auc["feature_set"] == name].set_index("stratum")
        if block.empty:
            continue
        print(f"   {name:<20s}", end="")
        for stratum in STRATA:
            value = (f"{block.loc[stratum, 'mean_delta']:+.3f}"
                     if stratum in block.index else "-")
            print(f"{value:>11s}", end="")
        print()
        conclusions[f"auc_delta_all_{name}"] = (
            float(block.loc["all", "mean_delta"])
            if "all" in block.index else None)

    # The mechanism, in one line. A layer at the top of this list inside the
    # treeless stratum is the layer doing the work there.
    if "no_tree_standardised" in separation:
        strict = separation.dropna(subset=["no_tree_standardised"])
        strict = strict.reindex(strict["no_tree_standardised"].abs()
                                .sort_values(ascending=False).index)
        print("\nstrongest presence/background separators inside the treeless "
              "stratum:")
        for _, row in strict.head(4).iterrows():
            print(f"   {row['layer']:<32s} presence "
                  f"{row['presence_mean']:>7.2f} vs background "
                  f"{row['background_mean']:>7.2f}   "
                  f"standardised {row['no_tree_standardised']:>+.2f}")
        conclusions["top_separator_no_tree"] = str(strict.iloc[0]["layer"])

    if not direction.empty:
        print("\nspecies-direction agreement (share of species separated the "
              "same way):")
        for _, row in direction.iterrows():
            print(f"   {row['variable']:<30s} "
                  f"{100 * row['majority_direction_share']:5.1f}% of "
                  f"{int(row['species_tested']):>3d} species  "
                  f"|effect| {row['mean_abs_effect']:.2f}")
    return conclusions


def write_outputs(results, table, deltas, direction, separation, data, args,
                  meta):
    os.makedirs(args.output_dir, exist_ok=True)
    paths = {
        "per_species_fold": "metrics_by_species_fold_stratum.csv",
        "comparison": "comparison_by_stratum.csv",
        "deltas": "paired_deltas.csv",
        "direction": "direction_agreement.csv",
        "separation": "presence_background_separation.csv",
    }
    separation.to_csv(os.path.join(args.output_dir, paths["separation"]),
                      index=False)
    results.to_csv(os.path.join(args.output_dir, paths["per_species_fold"]),
                   index=False)
    table.to_csv(os.path.join(args.output_dir, paths["comparison"]),
                 index=False)
    deltas.to_csv(os.path.join(args.output_dir, paths["deltas"]), index=False)
    direction.to_csv(os.path.join(args.output_dir, paths["direction"]),
                     index=False)

    summary = {
        "experiment": "satellite layers as SDM predictors, France",
        "not_a_production_model": True,
        "profile": args.profile,
        "grid": f"{data['grid']['width']}x{data['grid']['height']} @ "
                f"{data['grid']['cell']:.10f} deg",
        "occurrence_source": data["occurrence_source"],
        "background_kind": data["background_kind"],
        "n_rows": int(len(data["y"])),
        "n_species": len(data["species"]),
        "folds": int(args.folds),
        "blocks": int(args.blocks),
        "epochs": int(args.epochs),
        "climate_window": args.climate_window,
        "threshold_strategy": args.threshold_strategy,
        "feature_sets": {name: {"label": spec["label"],
                                "n_columns": meta["columns"][name],
                                "n_predictors": meta["predictors"][name],
                                "epochs_used": meta["epochs"][name]}
                         for name, spec in FEATURE_SETS.items()},
        "satellite_layers": [layer["path"]
                             for layer in data["satellite_catalog"]],
        "strata": {name: STRATUM_LABEL[name] for name in STRATA},
        "stratum_cells": meta["stratum_cells"],
        "input_checksums": meta["checksums"],
        "verdict": meta["verdict"],
        "elapsed_seconds": round(meta["elapsed"], 1),
    }
    with open(os.path.join(args.output_dir, "experiment_summary.json"),
              "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"\nwritten          {args.output_dir}/")
    for path in paths.values():
        print(f"                 {path}")
    print(f"                 experiment_summary.json")


def checksum(path):
    import hashlib
    if not os.path.exists(path):
        return None
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="controlled experiment: does satellite data help an "
                    "afforestation SDM, or is it circular?")
    parser.add_argument("--profile", default="fra_30s",
                        choices=sorted(SATELLITE_BLOCKS))
    parser.add_argument("--climate-window", default=feat.DEFAULT_CLIMATE_WINDOW)
    parser.add_argument("--occurrences", default=OCCURRENCE_SNAPSHOT)
    parser.add_argument("--background", default=BACKGROUND_SNAPSHOT)
    parser.add_argument("--min-records", type=int,
                        default=dst.MIN_RECORDS_PER_SPECIES)
    parser.add_argument("--max-species", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--background-ratio", type=float,
                        default=dst.BACKGROUND_PER_PRESENCE_CELL)
    parser.add_argument("--folds", type=int, default=dst.N_SPATIAL_FOLDS)
    parser.add_argument("--blocks", type=int, default=dst.N_SPATIAL_BLOCKS)
    parser.add_argument("--epochs", type=int, default=dst.EPOCHS)
    parser.add_argument("--threshold-strategy", default=dst.THRESHOLD_STRATEGY,
                        choices=["max_tss", "tenth_percentile"])
    parser.add_argument("--feature-sets", nargs="*", default=None,
                        choices=sorted(FEATURE_SETS),
                        help="run a subset of the arms (default: all four)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run: few species, few epochs, 3 folds")
    args = parser.parse_args(argv)
    if args.smoke:
        args.max_species = args.max_species or 12
        args.max_records = args.max_records or 40000
        args.epochs = 4 if args.epochs == dst.EPOCHS else args.epochs
        args.folds = min(args.folds, 3)
        args.output_dir = os.path.join(args.output_dir, "smoke")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.time()

    divider("SATELLITE-AS-PREDICTOR EXPERIMENT  -  NOT A PRODUCTION MODEL")
    print("Headline AUC is expected to RISE when vegetation layers are added. "
          "That is not")
    print("evidence they should be used: tree cover predicts tree occurrence "
          "almost by")
    print("definition. The result that decides the question is the "
          "low_tree and cropland")
    print("stratum, where a real ecological signal would survive and a "
          "circular one would")
    print("not.")
    if args.smoke:
        print("\nSMOKE TEST: plumbing only, every number below is a shape "
              "check.")

    device = dst.pick_device(args.device)
    data = build_union_dataset(args)

    divider("3. STRATA  -  defined from tree and cropland fraction")
    masks = stratum_masks(data)

    divider("4. SPATIAL BLOCKS  -  one split shared by every feature set")
    blocks = dst.spatial_block_ids(data["lat"], data["lon"],
                                   n_blocks=args.blocks)
    splits = dst.spatial_cv_splits(blocks, n_splits=args.folds)
    if splits is None:
        raise SystemExit("Not enough spatial blocks for cross-validation")
    sizes = np.bincount(blocks)
    print(f"blocks           {len(sizes)} clusters, {sizes.min():,}-"
          f"{sizes.max():,} cells each; {len(splits)} folds, whole blocks "
          f"held out")

    chosen = args.feature_sets or list(FEATURE_SETS)
    meta = {"columns": {}, "predictors": {}, "epochs": {},
            "stratum_cells": {k: int(v.sum()) for k, v in masks.items()},
            "checksums": {
                "occurrences": checksum(args.occurrences),
                "background": checksum(args.background),
                "effort_raster": checksum(
                    feat.GRID_PROFILES[args.profile]["effort_raster"])}}
    frames = []
    for name in chosen:
        spec = FEATURE_SETS[name]
        features, names, n_predictors = feature_matrix(data, spec)
        meta["columns"][name] = features.shape[1]
        meta["predictors"][name] = n_predictors
        divider(f"5. ARM '{name}'  -  {spec['label']}")
        print(f"features         {features.shape[1]} columns "
              f"({n_predictors} predictors + "
              f"{features.shape[1] - n_predictors} missingness indicators)")
        frame, epochs_used = cross_validate_arm(
            name, features, data, blocks, splits, masks, args, device)
        meta["epochs"][name] = epochs_used
        frames.append(frame)
        del features

    results = pd.concat(frames, ignore_index=True)
    table = comparison_table(results)
    deltas = paired_deltas(results)
    print_headline(table, deltas)

    # Every satellite layer, plus four environmental ones as the reference
    # behaviour a genuine niche axis has: species should disagree about it.
    variables = {name: data["sat_features"][:, i]
                 for i, name in enumerate(data["sat_names"])}
    for name in ("bio_1", "bio_12", "ph_h2o_0-30cm", "elevation"):
        if name in data["env_names"]:
            variables[f"ENV {name}"] = data["env_features"][
                :, data["env_names"].index(name)]

    divider("6. PRESENCE vs BACKGROUND  -  which layers separate the labels")
    separation = presence_background_separation(data, masks)
    print("`difference` is presence mean minus background mean in the layer's "
          "own units;")
    print("`standardised` divides by the pooled spread. The no_tree_* columns "
          "repeat it")
    print("inside the strictest treeless stratum, where a separating layer "
          "lifts AUC")
    print("without supplying any ecology.\n")
    print(separation.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    divider("7. SPECIES-DIRECTION AGREEMENT  -  does the global 90% reproduce?")
    direction = direction_agreement(data, variables)
    print("majority_direction_share is the fraction of species separated the "
          "same way.")
    print("A niche axis should sit near 0.5; a 'are there trees here' variable "
          "near 1.0.")
    print("mean_abs_effect is how strongly the variable separates presence "
          "from background.\n")
    print(direction.to_string(index=False))

    meta["verdict"] = verdict(deltas, direction, separation)
    meta["elapsed"] = time.time() - started
    write_outputs(results, table, deltas, direction, separation, data, args,
                  meta)

    divider("DONE")
    print(f"elapsed          {meta['elapsed']:.1f}s")
    print("reminder         deep_sdm_training.py is unchanged and still "
          "satellite-free.")


if __name__ == "__main__":
    main()
