"""
Feature extraction for the deep SDM — climate + soil + terrain, vectorised

What this is
------------
The data layer shared by `deep_sdm_training.py`. It turns coordinates into a
predictor matrix, and it is the only place that knows which rasters are allowed
to be predictors.

Grid profiles
-------------
Everything is selected through GRID_PROFILES, so switching country or
resolution is a flag rather than an edit. Three profiles ship today:

    global_10m   2160x1080  1/6 deg    the original global rasters
    fra_10m        93x63    1/6 deg    metropolitan France, 10 arcmin
    fra_30s      1860x1260  1/120 deg  metropolitan France, ~1 km

The two France grids share the origin (-5.5, 51.5) and nest exactly 20x20, and
fra_10m in turn nests exactly in global_10m (its origin is global column 1047,
row 231). Adding a country means adding a dict entry.

Why terrain layers are listed explicitly
----------------------------------------
An earlier version discovered terrain layers by substring match, which was a
latent silent-failure generator: `aspect_strength` would have matched "aspect"
and been pushed through cos/sin despite being a 0-1 magnitude; `slope_mean`,
`slope_q3` and `slope_max` would all have matched "slope" and collapsed into
three columns sharing one name; and `northness`/`eastness` matched nothing and
would have been dropped without a word. Layers are now named one by one per
profile, and `report_unused_layers()` prints anything sitting in a predictor
directory that no profile consumed, so a dropped layer is always visible.

Why it exists as its own module
-------------------------------
`xgboost_training.py` (the baseline, left untouched for comparison) samples
rasters like this:

    for f in climate_files:
        with rasterio.open(path) as src:
            for lat, lon in coords:
                val = src.read(1)[row, col]        # <- full array, per point

`src.read(1)` decodes the entire band on every iteration, so the cost is
O(points x layers) full-raster reads. This module reads each raster **once**
(through a window covering only the points asked for) and fancy-indexes every
point out of it in one step: O(layers) reads.

What is deliberately excluded
-----------------------------
Every vegetation layer. NDVI and tree-cover fraction are circular for this
problem — they measure where trees already are, and the job is to rank land
where trees are *not*. That was measured, not assumed: vegetation layers alone
reached cross-validated AUC 0.791 against 0.711 for the whole genuine
environmental set, but the advantage collapsed to +0.001 on cleared farmland,
which is the land the advisory actually has to rank.

`assert_no_vegetation_layers()` enforces this two ways, because name-matching
alone already failed once: the check used to look for a path component equal to
`satellite`, which would not have caught `country_data/FRA/satellite_download/`.
It now matches vegetation tokens as substrings of any path component *and*
refuses any layer whose directory (or any ancestor) carries a
`DO_NOT_TRAIN_ON_THIS.md` marker. The marker is the real safeguard: it works
for directories nobody has thought of yet.

There is exactly one door through it: `allow_vegetation_predictors_experiment()`
sets `ALLOW_VEGETATION_PREDICTORS_EXPERIMENT`, which downgrades the hard stop to
a printed list of what was let through. It exists because the exclusion is an
empirical claim and `deep_sdm_satellite_experiment.py` re-measures it on each
new country's data. It defaults to off, so the production path is unchanged.

The soil provenance flag is likewise not a predictor. It is now in
`country_data/FRA/provenance/`, outside the soil directory entirely.

Missing values
--------------
Coverage is high but not complete, and the gaps are *not* random — they are
narrow coastlines, small islands and, in a country subset, the sea inside the
bounding box. The default is **impute plus a missingness indicator**, not row
dropping: each feature group contributes one extra binary column saying "this
group was unknown here", and the network can learn that unknown soil is its own
condition rather than average soil. Rows are dropped only when more than
MAX_MISSING_FRACTION of predictors are gone, which in practice means sea.
Imputation statistics come from the training fold only (`FeatureScaler`),
because computing them over all data leaks the validation fold into training —
trees get away without normalisation, networks do not.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from tqdm import tqdm

# ─────────────────────────────────────────────────────
# CONFIG — soil and climate layer names (shared by every profile)
# ─────────────────────────────────────────────────────
# 19 bioclim variables, named explicitly rather than globbed, so that
# coverage_mask.tif cannot wander into the feature set.
BIOCLIM = [f"bio_{i}" for i in range(1, 20)]

SOIL_PROPERTIES = ["ph_h2o", "sand", "silt", "clay", "bdod", "soc", "socd",
                   "nitrogen", "cec", "cfvo", "drainage"]
SOIL_DEPTHS = ["0-30cm", "30-60cm"]   # 11 x 2 = 22 layers

# Water- and energy-balance layers that exist only in the 1 km French block.
# elevation_worldclim is left out: it is a second copy of elevation from a
# different DEM, and terrain already supplies it.
CLIMATE_EXTRA_30S = ["aridity_index_hargreaves", "aridity_index_penman",
                     "pet_hargreaves_annual", "pet_penman_annual",
                     "srad_annual_mean", "vapr_annual_mean",
                     "vpd_annual_mean", "water_deficit_annual",
                     "wind_annual_mean"]

# `kind` says how a layer is consumed:
#   "linear"           -> used as-is
#   "circular_degrees" -> expanded to northness=cos and eastness=sin, because
#                         raw degrees are discontinuous (359 and 1 point almost
#                         the same way but sit at opposite ends of the number
#                         line). No profile needs it today — both French terrain
#                         blocks ship northness and eastness already decomposed
#                         at native 1 arcsec, which is strictly better than
#                         decomposing an averaged aspect. Kept because other
#                         DEM products publish aspect and nothing else.
TERRAIN_FRA_10M = [
    ("elevation", "gedtm30_v1.2_elev_10m.tif", "linear"),
    ("elev_std", "elev_std_10m.tif", "linear"),
    ("slope", "slope_10m.tif", "linear"),
    ("northness", "northness_10m.tif", "linear"),
    ("eastness", "eastness_10m.tif", "linear"),
]
# aspect_10m.tif / aspect.tif are excluded on purpose: verified to equal
# atan2(eastness, northness) in degrees to within 0.0000 at p99, so they are an
# exact reparameterisation of two columns already in the set.
TERRAIN_FRA_30S = [
    ("elevation", "elev_mean.tif", "linear"),
    ("elev_min", "elev_min.tif", "linear"),
    ("elev_max", "elev_max.tif", "linear"),
    ("elev_range", "elev_range.tif", "linear"),
    ("elev_std", "elev_std.tif", "linear"),
    ("slope", "slope.tif", "linear"),
    ("steepness_q3", "steepness_q3.tif", "linear"),
    ("steepness_max", "steepness_max.tif", "linear"),
    ("northness", "northness.tif", "linear"),
    ("eastness", "eastness.tif", "linear"),
    ("terrain_vector_strength", "terrain_vector_strength.tif", "linear"),
]
TERRAIN_GLOBAL = [
    ("elevation", "gedtm30_v1.2_elev_10m.tif", "linear"),
]

# ─────────────────────────────────────────────────────
# CONFIG — grid profiles
# ─────────────────────────────────────────────────────
# `climate` maps a window name to (directory, filename template). The default
# window is `recent` (2015-2024) everywhere: the median GBIF record year is
# 2024, and annual mean temperature at the occurrence points is +1.28 degC
# warmer in 2015-2024 than in the 1970-2000 window the baseline uses.
GRID_PROFILES = {
    "global_10m": {
        "label": "global, 10 arcmin (~18.5 km)",
        "template": "climate_current/wc2.1_10m_bio_1.tif",
        "climate": {
            "recent": ("climate_recent", "wc2.1_10m_{bio}.tif"),
            "current": ("climate_current", "wc2.1_10m_{bio}.tif"),
            "2000_2015": ("climate_2000_2015", "wc2.1_10m_{bio}.tif"),
        },
        "climate_extra": [],
        "soil": ("soil_data/aligned_10m", "{prop}_{depth}_10m.tif"),
        "terrain": ("topography/gedtm30", TERRAIN_GLOBAL),
        "exclusion_mask": "satellite/planting_exclusion_mask_10m.tif",
        "background": "sampling_effort/background_points.csv",
        "effort_raster": "sampling_effort/plant_effort_10m.tif",
        "extent": None,
    },
    "fra_10m": {
        "label": "metropolitan France, 10 arcmin (~18.5 km)",
        "template": "country_data/FRA/climate_recent/wc2.1_10m_bio_1.tif",
        "climate": {
            "recent": ("country_data/FRA/climate_recent",
                       "wc2.1_10m_{bio}.tif"),
            "current": ("country_data/FRA/climate_current",
                        "wc2.1_10m_{bio}.tif"),
        },
        "climate_extra": [],
        "soil": ("country_data/FRA/aligned_10m", "{prop}_{depth}_10m.tif"),
        "terrain": ("country_data/FRA/gedtm30", TERRAIN_FRA_10M),
        "exclusion_mask": "satellite/planting_exclusion_mask_10m.tif",
        # The 10 arcmin background is native to this grid: drawn directly from
        # the 10 arcmin effort surface, so no resampling is involved either way.
        # Named per country rather than as background_points.csv, which
        # sampling_effort.py rewrites for whichever country it last ran and which
        # is currently a copy of Spain's.
        "background": "sampling_effort/background_FR.csv",
        "effort_raster": "sampling_effort/plant_effort_FR_10m.tif",
        "extent": None,
    },
    "fra_30s": {
        "label": "metropolitan France, 30 arcsec (~1 km)",
        "template": "country_data/FRA/climate_30s/wc2.1_30s_bio_1.tif",
        "climate": {
            "recent": ("country_data/FRA/climate_30s", "wc2.1_30s_{bio}.tif"),
        },
        "climate_extra": CLIMATE_EXTRA_30S,
        "soil": ("country_data/FRA/soil_30s", "{prop}_{depth}_30s.tif"),
        "terrain": ("country_data/FRA/topography_30s", TERRAIN_FRA_30S),
        # The mask is only published on the global 10 arcmin grid, so it is
        # nearest-neighbour upsampled onto this one. The two nest exactly
        # 20x20, so every 1 km cell inherits its 18.5 km parent's bits. That is
        # the resolution of the filter, not of the model.
        "exclusion_mask": "satellite/planting_exclusion_mask_10m.tif",
        # Both are now native to this grid. Until the effort surface was
        # rebuilt at 1 km there was only an 18.5 km one, which this module had
        # to resample — correcting survey bias between 18.5 km cells but not
        # within them. That gap is closed: the 1 km surface resolves 734,793
        # surveyed cells against 2,858 at 18.5 km, and the resulting background
        # cuts built-up separation from presences by 42% (AUC 0.582 uniform,
        # 0.557 at 18.5 km, 0.533 at 1 km). France uses the unsmoothed surface
        # on purpose: smoothing widened the built-up gap from 2.47 to 3.14 at a
        # 1 km kernel and 3.65 at 5 km, because blurring destroys the
        # town-scale structure the background has to reproduce.
        "background": "sampling_effort/background_FR_30s.csv",
        "effort_raster": "sampling_effort/plant_effort_FR_30s.tif",
        "extent": None,
    },
}
DEFAULT_PROFILE = "fra_30s"
DEFAULT_CLIMATE_WINDOW = "recent"

# ─────────────────────────────────────────────────────
# CONFIG — occurrence and background inputs
# ─────────────────────────────────────────────────────
# First existing path wins, so a better occurrence file is picked up without a
# code change. gbif_trees_clean.csv is preferred over the pre-thinned file
# because thinning belongs at the *analysis* grid: the thinned file was reduced
# to 10 arcmin cells, which throws away 2.7x of the records that a 1 km run can
# use. `thin_to_grid()` re-thins to whatever grid is actually in play.
OCCURRENCE_CANDIDATES = [
    "species_occurrences/gbif_trees_clean.csv",      # cleaned, un-thinned
    "species_occurrences/gbif_trees_thinned.csv",    # cleaned + 10 arcmin
    "gbif_500_species.csv",                          # original, uncleaned
]
# Fallback only. The right background depends on the grid, so each profile
# names its own in GRID_PROFILES["background"]; this list is what gets used if
# a profile does not, or if its file is missing.
# Country-specific names only. background_points.csv used to lead this list, but
# sampling_effort.py writes that name for whichever country it last ran, so after
# Spain was prepared the France fallback silently became Spanish points.
BACKGROUND_CANDIDATES = [
    "sampling_effort/background_FR_30s.csv",         # 1 km, target-group
    "sampling_effort/background_FR.csv",             # 18.5 km, target-group
]

# ─────────────────────────────────────────────────────
# CONFIG — what may never be a predictor
# ─────────────────────────────────────────────────────
# Substrings, matched against every component of a layer's path. The old check
# compared components for equality with "satellite", which missed
# `country_data/FRA/satellite_download/` entirely.
FORBIDDEN_PATH_TOKENS = ("satellite", "ndvi", "landcover", "land_cover",
                         "vegetation", "treecover", "tree_frac", "evi", "lai",
                         "greenness", "provenance", "source_flag")
# A directory carrying this file is off limits, along with everything under it.
# Name-independent, so it also covers directories that do not exist yet.
DO_NOT_TRAIN_MARKER = "DO_NOT_TRAIN_ON_THIS.md"

# The single, deliberate hole in the guard, for `deep_sdm_satellite_experiment.py`
# and nothing else. It is a module-level flag rather than a function argument so
# that it cannot be switched on by a stray keyword passed through some other
# call: turning it on takes an import and a named function call that prints a
# banner, and `allow_vegetation_predictors_experiment()` is the only way to do
# it. The default is off, so every existing caller — including
# `deep_sdm_training.py` — is unaffected and still hard-stops.
#
# Why an override at all: the exclusion is an empirical claim ("the advantage is
# circular and collapses on treeless land"), and an empirical claim has to stay
# testable on new data. Weakening the guard to allow the test would have removed
# the protection; this keeps the protection and adds one loud, auditable door.
ALLOW_VEGETATION_PREDICTORS_EXPERIMENT = False
_VEGETATION_OVERRIDE_REASON = None


def allow_vegetation_predictors_experiment(reason):
    """Open the vegetation guard for a measurement run. Prints a banner.

    `reason` is required and echoed, so a log that contains vegetation
    predictors always says on its own why.
    """
    global ALLOW_VEGETATION_PREDICTORS_EXPERIMENT, _VEGETATION_OVERRIDE_REASON
    if not reason:
        raise SystemExit("allow_vegetation_predictors_experiment() needs a "
                         "reason; it goes in the log")
    ALLOW_VEGETATION_PREDICTORS_EXPERIMENT = True
    _VEGETATION_OVERRIDE_REASON = str(reason)
    print("!" * 72)
    print("ALLOW_VEGETATION_PREDICTORS_EXPERIMENT is ON. Vegetation layers may "
          "enter the")
    print("predictor set for this process. Any model fitted now is an "
          "experiment, not a")
    print("production model, and its headline scores are not comparable to the "
          "advisory's.")
    print(f"reason: {_VEGETATION_OVERRIDE_REASON}")
    print("!" * 72)
    return True

# ─────────────────────────────────────────────────────
# CONFIG — numerics
# ─────────────────────────────────────────────────────
NODATA_FLOOR = -1e30        # WorldClim writes -3.4e38 as its nodata sentinel
MAX_MISSING_FRACTION = 0.5  # drop a row only if over half its predictors are gone
CLIP_SIGMA = 5.0            # clamp standardised features, blunts raster outliers
GRID_BLOCK_ROWS = 60        # raster rows per chunk when scoring a whole extent
EFFORT_FLOOR = 0.5          # so a never-surveyed cell keeps a small chance
EFFORT_POWER = 1.0          # 1.0 = background exactly proportional to effort
RANDOM_SEED = 42


def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ─────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────
def load_grid(profile=DEFAULT_PROFILE):
    """Read the grid definition from a profile's template raster."""
    spec = GRID_PROFILES[profile]
    with rasterio.open(spec["template"]) as src:
        return {
            "profile": profile,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "cell": abs(src.transform.a),
            "template": spec["template"],
        }


def check_grid(catalog, grid, verbose=True):
    """Confirm every predictor sits on the profile's grid. Cheap insurance."""
    reference = (grid["width"], grid["height"], grid["crs"],
                 tuple(round(v, 12) for v in tuple(grid["transform"])[:6]))
    mismatches = []
    for layer in catalog:
        with rasterio.open(layer["path"]) as src:
            here = (src.width, src.height, str(src.crs),
                    tuple(round(v, 12) for v in tuple(src.transform)[:6]))
        if here != reference:
            mismatches.append((layer["name"], here))
    if verbose:
        print(f"grid check       {len(catalog)} layers, "
              f"{len(mismatches)} mismatches")
    if mismatches:
        raise SystemExit(
            "Layers are not on the profile grid, so row/col indexing would "
            f"silently sample the wrong place: {mismatches}")
    return True


def lonlat_to_rowcol(lat, lon, grid):
    """Map coordinates onto grid cells. Returns row, col and an inside mask."""
    transform = grid["transform"]
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    col = np.floor((lon - transform.c) / transform.a).astype("int64")
    row = np.floor((lat - transform.f) / transform.e).astype("int64")
    inside = ((row >= 0) & (row < grid["height"]) &
              (col >= 0) & (col < grid["width"]))
    return row, col, inside


def cell_centres(row, col, grid):
    """Inverse of lonlat_to_rowcol: cell indices back to centre lat/lon."""
    transform = grid["transform"]
    lat = transform.f + (np.asarray(row) + 0.5) * transform.e
    lon = transform.c + (np.asarray(col) + 0.5) * transform.a
    return lat, lon


# ─────────────────────────────────────────────────────
# LAYER CATALOG
# ─────────────────────────────────────────────────────
def build_catalog(profile=DEFAULT_PROFILE, climate_window=None,
                  include_soil=True, include_terrain=True):
    """List the predictor rasters for a profile, in a fixed order, by group."""
    spec = GRID_PROFILES[profile]
    climate_window = climate_window or DEFAULT_CLIMATE_WINDOW
    if climate_window not in spec["climate"]:
        raise SystemExit(
            f"Profile {profile!r} has no {climate_window!r} climate window; "
            f"choose from {sorted(spec['climate'])}")

    climate_dir, pattern = spec["climate"][climate_window]
    catalog = [{"name": bio, "group": "climate", "kind": "linear",
                "path": os.path.join(climate_dir, pattern.format(bio=bio))}
               for bio in BIOCLIM]
    for name in spec["climate_extra"]:
        catalog.append({"name": name, "group": "climate", "kind": "linear",
                        "path": os.path.join(climate_dir, f"{name}.tif")})

    if include_soil:
        soil_dir, soil_pattern = spec["soil"]
        for prop in SOIL_PROPERTIES:
            for depth in SOIL_DEPTHS:
                catalog.append({
                    "name": f"{prop}_{depth}", "group": "soil",
                    "kind": "linear",
                    "path": os.path.join(soil_dir, soil_pattern.format(
                        prop=prop, depth=depth))})

    if include_terrain:
        terrain_dir, layers = spec["terrain"]
        for name, filename, kind in layers:
            catalog.append({"name": name, "group": "terrain", "kind": kind,
                            "path": os.path.join(terrain_dir, filename)})

    missing = [layer for layer in catalog if not os.path.exists(layer["path"])]
    catalog = [layer for layer in catalog if os.path.exists(layer["path"])]
    for layer in missing:
        warnings.warn(f"predictor not on disk, skipped: {layer['path']}")

    assert_no_vegetation_layers(catalog)
    return catalog


def assert_no_vegetation_layers(catalog):
    """Hard stop if a circular or non-predictor layer is being used as input.

    Two independent tests, because the token list alone already failed once:
    a directory named `satellite_download` did not match a check that compared
    path components for equality with `satellite`.
    """
    offenders = []
    for layer in catalog:
        path = os.path.normpath(layer["path"])
        for part in path.split(os.sep):
            lowered = part.lower()
            if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
                offenders.append((layer["path"], f"path token in {part!r}"))
                break
        marker = find_do_not_train_marker(path)
        if marker:
            offenders.append((layer["path"], f"marked by {marker}"))
    if offenders and ALLOW_VEGETATION_PREDICTORS_EXPERIMENT:
        # Still listed, never silent: the override changes the consequence from
        # a hard stop to a printed admission, not the detection.
        print(f"OVERRIDE         {len(offenders)} layer(s) would normally be "
              f"refused as predictors,")
        print(f"                 allowed because "
              f"ALLOW_VEGETATION_PREDICTORS_EXPERIMENT is on:")
        for path, why in offenders:
            print(f"   {path}  ({why})")
        return True
    if offenders:
        raise SystemExit(
            "These layers cannot be predictors — vegetation is circular for "
            "this problem, and provenance flags are metadata:\n" +
            "\n".join(f"   {path}  ({why})" for path, why in offenders))
    return True


def find_do_not_train_marker(path):
    """Walk up from a layer to the repo root looking for the opt-out marker."""
    directory = os.path.dirname(os.path.abspath(path))
    root = os.path.abspath(os.curdir)
    while True:
        candidate = os.path.join(directory, DO_NOT_TRAIN_MARKER)
        if os.path.exists(candidate):
            return os.path.relpath(candidate, root)
        parent = os.path.dirname(directory)
        if parent == directory or os.path.normpath(directory) == root:
            return None
        directory = parent


def report_unused_layers(catalog, profile=DEFAULT_PROFILE):
    """Print rasters sitting in a predictor directory that nothing consumed.

    The point is that a dropped layer is never silent. The old substring
    auto-discovery would have quietly ignored northness and eastness; this
    makes that class of mistake visible without guessing at filenames.
    """
    used = {os.path.normpath(layer["path"]) for layer in catalog}
    # Only the directories this catalog actually draws from, so unselected
    # climate windows are not reported as if something had been forgotten.
    directories = [os.path.dirname(path) for path in used]

    unused = []
    for directory in dict.fromkeys(directories):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".tif"):
                continue
            path = os.path.normpath(os.path.join(directory, name))
            if path not in used:
                unused.append(path)
    if unused:
        print(f"unused layers    {len(unused)} raster(s) present but not used "
              f"as predictors:")
        for path in unused:
            print(f"   {path}")
    return unused


def feature_names(catalog):
    """Column names after circular layers are expanded into two columns."""
    names = []
    for layer in catalog:
        if layer["kind"] == "circular_degrees":
            names.extend([f"{layer['name']}_northness",
                          f"{layer['name']}_eastness"])
        else:
            names.append(layer["name"])
    return names


def feature_groups(catalog):
    """Group label per output column, aligned with feature_names()."""
    groups = []
    for layer in catalog:
        span = 2 if layer["kind"] == "circular_degrees" else 1
        groups.extend([layer["group"]] * span)
    return groups


def describe_catalog(catalog, profile=DEFAULT_PROFILE):
    counts = {}
    for layer in catalog:
        counts[layer["group"]] = counts.get(layer["group"], 0) + 1
    print(f"profile          {profile}  ({GRID_PROFILES[profile]['label']})")
    print(f"predictors       {len(feature_names(catalog))} columns from "
          f"{len(catalog)} rasters")
    for group in sorted(counts):
        print(f"   {group:<12s} {counts[group]:>3d} layers")
    # Read off the catalog rather than hardcoded, so that during the
    # satellite experiment this line reports the truth instead of the policy.
    vegetation = sum(n for g, n in counts.items() if g.startswith("satellite"))
    if vegetation:
        print(f"   vegetation     {vegetation:>3d} layers  (EXPERIMENT — "
              f"circular for this problem)")
    else:
        print("   vegetation     0 layers  "
              "(circular for this problem — excluded)")


# ─────────────────────────────────────────────────────
# RASTER READING
# ─────────────────────────────────────────────────────
def read_band(src, window=None):
    """One band as float32 with every flavour of nodata turned into NaN."""
    band = src.read(1, window=window).astype("float32")
    nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        band[band == np.float32(nodata)] = np.nan
    band[band < NODATA_FLOOR] = np.nan
    return band


def expand_circular(values, kind):
    """Turn a raw-degree layer into (northness, eastness); pass others through.

    Aspect must never reach the network as degrees: 359 and 1 point at almost
    the same direction but sit at opposite ends of the number line.
    """
    if kind != "circular_degrees":
        return values[:, None]
    radians = np.deg2rad(values.astype("float64"))
    return np.column_stack([np.cos(radians), np.sin(radians)]).astype("float32")


def sample_points(catalog, lat, lon, grid, progress=True, desc="sampling"):
    """Sample every predictor at every coordinate. One raster read per layer.

    Returns (features, inside) where features is [n_points, n_columns] float32
    with NaN for nodata and for points outside the grid.
    """
    row, col, inside = lonlat_to_rowcol(lat, lon, grid)
    n_columns = len(feature_names(catalog))
    features = np.full((len(row), n_columns), np.nan, dtype="float32")
    if not inside.any():
        return features, inside

    # A window covering only the requested points.
    row_in, col_in = row[inside], col[inside]
    r0, r1 = int(row_in.min()), int(row_in.max()) + 1
    c0, c1 = int(col_in.min()), int(col_in.max()) + 1
    window = Window(c0, r0, c1 - c0, r1 - r0)
    local_row, local_col = row_in - r0, col_in - c0

    column = 0
    layers = tqdm(catalog, desc=desc, unit="layer") if progress else catalog
    for layer in layers:
        with rasterio.open(layer["path"]) as src:
            band = read_band(src, window=window)          # <- read once
        values = band[local_row, local_col]               # <- all points, once
        block = expand_circular(values, layer["kind"])
        features[np.flatnonzero(inside), column:column + block.shape[1]] = block
        column += block.shape[1]
        del band

    return features, inside


def iter_grid_blocks(catalog, grid, extent=None, block_rows=GRID_BLOCK_ROWS,
                     progress=True, desc="scoring grid"):
    """Yield (rows, cols, features) for every cell in an extent, block by block.

    Used for prediction. Reads each raster once per block rather than once per
    cell, which is what makes whole-extent scoring tractable.
    """
    r0, r1, c0, c1 = extent_to_window(extent, grid)
    blocks = range(r0, r1, block_rows)
    bar = tqdm(list(blocks), desc=desc, unit="block") if progress else blocks

    for start in bar:
        stop = min(start + block_rows, r1)
        window = Window(c0, start, c1 - c0, stop - start)
        stack = []
        for layer in catalog:
            with rasterio.open(layer["path"]) as src:
                band = read_band(src, window=window)
            stack.append(expand_circular(band.reshape(-1), layer["kind"]))
            del band
        features = np.concatenate(stack, axis=1)
        rows, cols = np.meshgrid(np.arange(start, stop),
                                 np.arange(c0, c1), indexing="ij")
        yield rows.reshape(-1), cols.reshape(-1), features


def extent_to_window(extent, grid):
    """(min_lon, min_lat, max_lon, max_lat) -> (row0, row1, col0, col1)."""
    if extent is None:
        return 0, grid["height"], 0, grid["width"]
    min_lon, min_lat, max_lon, max_lat = extent
    top_row, left_col, _ = lonlat_to_rowcol([max_lat], [min_lon], grid)
    bottom_row, right_col, _ = lonlat_to_rowcol([min_lat], [max_lon], grid)
    r0 = int(np.clip(top_row[0], 0, grid["height"] - 1))
    r1 = int(np.clip(bottom_row[0], 0, grid["height"] - 1)) + 1
    c0 = int(np.clip(left_col[0], 0, grid["width"] - 1))
    c1 = int(np.clip(right_col[0], 0, grid["width"] - 1)) + 1
    return r0, r1, c0, c1


def read_onto_grid(path, grid, resampling=Resampling.nearest):
    """Read any raster onto the profile grid, resampling if grids differ.

    Needed because the planting exclusion mask is only published on the global
    10 arcmin grid, while a run may be on the 1 km French grid.
    """
    with rasterio.open(path) as src:
        same = (src.width == grid["width"] and src.height == grid["height"]
                and tuple(round(v, 12) for v in tuple(src.transform)[:6])
                == tuple(round(v, 12) for v in tuple(grid["transform"])[:6]))
        if same:
            return src.read(1), src.nodata
        out = np.zeros((grid["height"], grid["width"]), dtype=src.dtypes[0])
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  resampling=resampling,
                  src_nodata=src.nodata, dst_nodata=src.nodata)
        return out, src.nodata


# ─────────────────────────────────────────────────────
# OCCURRENCES
# ─────────────────────────────────────────────────────
def resolve_input(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise SystemExit(f"No {label} found. Looked for: {candidates}")


def load_occurrences(path=None, country=None, extent=None, min_records=1):
    """Load presence records as species / latitude / longitude.

    `country` filters on the ISO code column when the source has one; `extent`
    clips to a bounding box. Both exist because the project runs as a
    single-country proof of concept, and neither should require a code change.
    """
    path = path or resolve_input(OCCURRENCE_CANDIDATES, "occurrence file")
    frame = pd.read_csv(path)
    lower = {c.lower(): c for c in frame.columns}
    needed = ("species", "latitude", "longitude")
    if not all(key in lower for key in needed):
        raise SystemExit(f"{path} is missing one of {needed}; has "
                         f"{list(frame.columns)}")
    frame = frame.rename(columns={lower[k]: k for k in needed})

    before = len(frame)
    if country:
        column = lower.get("country") or lower.get("countrycode")
        if column is None:
            warnings.warn(f"{path} has no country column; --country ignored")
        else:
            wanted = {c.strip().upper() for c in np.atleast_1d(country)}
            codes = frame[column].astype("string").str.strip().str.upper()
            frame = frame[codes.isin(wanted)]

    frame = frame[np.isfinite(frame["latitude"]) &
                  np.isfinite(frame["longitude"])]
    if extent is not None:
        min_lon, min_lat, max_lon, max_lat = extent
        frame = frame[(frame["longitude"] >= min_lon) &
                      (frame["longitude"] <= max_lon) &
                      (frame["latitude"] >= min_lat) &
                      (frame["latitude"] <= max_lat)]

    frame = frame[["species", "latitude", "longitude"]].reset_index(drop=True)
    if min_records > 1:
        keep = frame["species"].value_counts()
        keep = keep[keep >= min_records].index
        frame = frame[frame["species"].isin(keep)].reset_index(drop=True)

    print(f"occurrences      {path}")
    print(f"                 {len(frame):,} records "
          f"({before:,} before filters), "
          f"{frame['species'].nunique():,} species")
    return frame, path


def thin_to_grid(frame, grid):
    """Keep one record per species per grid cell.

    Thinning has to happen at the grid the model actually uses. Two records of
    one species in the same cell carry no extra environmental information — the
    cell has a single set of predictor values — but duplicate records do pull
    the loss toward heavily-surveyed places. The supplied thinned file was
    reduced to 10 arcmin cells, so on the 1 km grid it has already discarded
    records that are environmentally distinct there; re-thinning the un-thinned
    file recovers them.
    """
    row, col, inside = lonlat_to_rowcol(frame["latitude"].to_numpy(),
                                        frame["longitude"].to_numpy(), grid)
    frame = frame.assign(grid_row=row, grid_col=col)[inside]
    before = len(frame)
    frame = (frame.drop_duplicates(subset=["species", "grid_row", "grid_col"])
             .reset_index(drop=True))
    print(f"thinning         {len(frame):,} kept of {before:,} on-grid records "
          f"(one per species per {grid['cell'] * 111:.1f} km cell)")
    return frame


def occurrence_extent(frame, pad_degrees=1.0):
    """Bounding box around a set of records, padded. Fallback when no bbox."""
    return (float(frame["longitude"].min()) - pad_degrees,
            float(frame["latitude"].min()) - pad_degrees,
            float(frame["longitude"].max()) + pad_degrees,
            float(frame["latitude"].max()) + pad_degrees)


# ─────────────────────────────────────────────────────
# BACKGROUND POINTS
# ─────────────────────────────────────────────────────
def load_background(grid, n_points=None, n_presence_cells=None, path=None,
                    extent=None, effort_raster=None, rng=None):
    """Background ("pseudo-absence") points.

    Preference order:

    1. An externally supplied target-group background file, used as delivered,
       if its points resolve to distinct cells on this grid. This is the
       correct sampler — drawn proportional to a vascular-plant recording-effort
       surface, so it reproduces where people actually looked.
    2. The same effort surface, resampled onto this grid and sampled directly.
       Needed when the supplied file is on a coarser grid than the run: the
       10 arcmin file collapses to ~2,373 distinct locations, which is far too
       few to stand as background for ~70,000 presence cells at 1 km.
    3. A uniform fallback over land, which is **wrong** and labelled as such.

    Option 2 inherits a real limitation: the effort surface itself is only
    known at 10 arcmin, so within any 18.5 km cell the background is placed
    uniformly. Survey bias is corrected between 18.5 km cells and not within
    them.
    """
    path = path or next((p for p in BACKGROUND_CANDIDATES
                         if os.path.exists(p)), None)
    if path:
        lat, lon, distinct = read_background_file(path, extent, grid)
        # The test is whether the supplied points cover ground at a density
        # comparable to the presences on *this* grid — not whether there are as
        # many as were requested. The 10 arcmin file has 2,373 distinct cells
        # against ~2,400 presence cells, which is native and correct; on the
        # 1 km grid the same file faces ~69,500 presence cells and is 30x too
        # coarse to serve as background.
        reference = n_presence_cells or n_points
        enough = reference is None or distinct >= 0.5 * reference
        if enough:
            # Used exactly as delivered, not subsampled: the file's size and
            # its effort weighting were chosen together upstream.
            print(f"background       {path}  ({len(lat):,} points, "
                  f"{distinct:,} distinct cells, effort-weighted target group)")
            return lat, lon, "target_group_file"
        print(f"background       {path} resolves to only {distinct:,} distinct "
              f"cells on this grid,")
        print(f"                 against {reference:,} presence cells. "
              f"Resampling the effort surface")
        print(f"                 onto this grid instead.")

    if effort_raster and os.path.exists(effort_raster):
        return effort_weighted_background(grid, n_points, effort_raster,
                                          extent=extent, rng=rng)

    return uniform_background_fallback(grid, n_points, extent=extent, rng=rng)


def read_background_file(path, extent, grid):
    """Read supplied background points and count how many distinct cells."""
    frame = pd.read_csv(path)
    lower = {c.lower(): c for c in frame.columns}
    if "latitude" not in lower or "longitude" not in lower:
        raise SystemExit(f"{path} needs latitude and longitude columns; has "
                         f"{list(frame.columns)}")
    lat = frame[lower["latitude"]].to_numpy(dtype="float64")
    lon = frame[lower["longitude"]].to_numpy(dtype="float64")

    if extent is not None:
        min_lon, min_lat, max_lon, max_lat = extent
        keep = ((lon >= min_lon) & (lon <= max_lon) &
                (lat >= min_lat) & (lat <= max_lat))
        lat, lon = lat[keep], lon[keep]

    row, col, inside = lonlat_to_rowcol(lat, lon, grid)
    distinct = len(set(zip(row[inside].tolist(), col[inside].tolist())))
    return lat, lon, distinct


def effort_weighted_background(grid, n_points, effort_raster, extent=None,
                               rng=None):
    """Sample cells with probability proportional to recording effort.

    This is the same target-group idea as the supplied file, applied on the
    run's own grid. The effort raster is nearest-neighbour resampled, so a
    1 km cell inherits its 18.5 km parent's effort and every 1 km cell inside
    one 18.5 km cell is equally likely.
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)
    effort, _ = read_onto_grid(effort_raster, grid, Resampling.nearest)
    effort = np.nan_to_num(np.asarray(effort, dtype="float64"), nan=0.0)

    with rasterio.open(grid["template"]) as src:
        land = np.isfinite(read_band(src))
    r0, r1, c0, c1 = extent_to_window(extent, grid)
    eligible = np.zeros_like(land, dtype=bool)
    eligible[r0:r1, c0:c1] = land[r0:r1, c0:c1]

    rows, cols = np.nonzero(eligible)
    weights = (effort[rows, cols] + EFFORT_FLOOR) ** EFFORT_POWER
    take = min(n_points, len(rows))
    if take < n_points:
        warnings.warn(f"only {take:,} eligible cells, wanted {n_points:,}")
    # Exponential race (Efraimidis-Spirakis): drawing k of 1.6M cells weighted
    # and without replacement is exact this way and costs one pass, where
    # rng.choice(p=...) with replace=False is quadratic and does not finish.
    keys = rng.exponential(size=len(rows)) / weights
    pick = np.argpartition(keys, take - 1)[:take]
    lat, lon = cell_centres(rows[pick], cols[pick], grid)

    surveyed = 100 * (effort[rows[pick], cols[pick]] > 0).mean()
    print(f"background       effort-weighted from {effort_raster} "
          f"({take:,} cells)")
    print(f"                 {surveyed:.1f}% fall in a cell with plant "
          f"records; effort is only known at")
    print(f"                 10 arcmin, so bias is corrected between 18.5 km "
          f"cells, not within them")
    return lat, lon, "effort_weighted_resampled"


def uniform_background_fallback(grid, n_points, extent=None, rng=None):
    """UNIFORM FALLBACK — runs anywhere, but is methodologically wrong.

    Occurrence records cluster near roads and cities, so a background drawn
    uniformly over land teaches the model that "near a city" is good habitat.
    In this project built-up fraction measured as the strongest single
    separator of presence from background (0.246) — survey bias, not ecology.
    """
    print("background       UNIFORM FALLBACK over land — NOT a correct "
          "sampler.")
    print("                 Uniform background encodes survey bias (built-up "
          "fraction was the")
    print("                 strongest presence/background separator at 0.246).")

    rng = rng or np.random.default_rng(RANDOM_SEED)
    r0, r1, c0, c1 = extent_to_window(extent, grid)
    with rasterio.open(grid["template"]) as src:
        band = read_band(src, window=Window(c0, r0, c1 - c0, r1 - r0))

    valid = np.isfinite(band)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        raise SystemExit("No valid background cells in the requested extent")
    take = min(n_points, len(rows))
    pick = rng.choice(len(rows), size=take, replace=False)
    lat, lon = cell_centres(rows[pick] + r0, cols[pick] + c0, grid)
    return lat, lon, "uniform_fallback"


# ─────────────────────────────────────────────────────
# CELL TABLE — presences and background merged
# ─────────────────────────────────────────────────────
def build_cell_table(occurrences, species_index, background_lat,
                     background_lon, grid):
    """One row per grid cell, with a multi-hot label and a per-species mask.

    Presence and background cells are **merged rather than stacked**, which
    matters here: the supplied target-group background puts 2,357 of its 2,373
    distinct 10 arcmin cells on cells that also hold a presence. Stacking would
    produce two rows with identical predictors and opposite labels for the same
    species, which is unlearnable noise. Merging gives the interpretation
    target-group background actually has: the target group was surveyed in this
    cell, species A was recorded and species B was not, so B is a usable
    negative *here*.

    Returns lat, lon, y, mask, has_record, surveyed.
      y[i, s]      1 if species s was recorded in cell i
      mask[i, s]   1 if cell i counts for species s in the loss
      surveyed[i]  the cell came from the target-group background
    """
    n_species = len(species_index)
    width = grid["width"]

    row, col, inside = lonlat_to_rowcol(occurrences["latitude"].to_numpy(),
                                        occurrences["longitude"].to_numpy(),
                                        grid)
    species = occurrences["species"].map(species_index)
    keep = inside & species.notna().to_numpy()
    presence_key = row[keep].astype("int64") * width + col[keep].astype("int64")
    species = species[keep].to_numpy(dtype="int64")

    back_row, back_col, back_inside = lonlat_to_rowcol(background_lat,
                                                       background_lon, grid)
    background_key = (back_row[back_inside].astype("int64") * width
                      + back_col[back_inside].astype("int64"))

    keys = np.unique(np.concatenate([presence_key, background_key]))
    index_of = {int(k): i for i, k in enumerate(keys)}

    y = np.zeros((len(keys), n_species), dtype="float32")
    y[[index_of[int(k)] for k in presence_key], species] = 1.0

    surveyed = np.zeros(len(keys), dtype=bool)
    surveyed[[index_of[int(k)] for k in np.unique(background_key)]] = True
    has_record = y.any(axis=1)

    # A positive always counts. A zero counts only where the target group was
    # surveyed — GBIF is presence-only, so "not recorded" is evidence of
    # absence only somewhere people actually looked.
    mask = np.maximum(y, surveyed[:, None].astype("float32"))

    cell_row = (keys // width).astype("int64")
    cell_col = (keys % width).astype("int64")
    lat, lon = cell_centres(cell_row, cell_col, grid)
    return lat, lon, y, mask, has_record, surveyed


# ─────────────────────────────────────────────────────
# MISSING VALUES + NORMALISATION
# ─────────────────────────────────────────────────────
def missingness_columns(features, groups):
    """One binary column per feature group: was this group unknown here?

    Per-group rather than per-column because 22 soil layers would otherwise add
    22 nearly identical indicators — soil gaps are whole-cell, not per-property.
    """
    groups = np.asarray(groups)
    order = sorted(set(groups.tolist()))
    indicators = np.zeros((len(features), len(order)), dtype="float32")
    for i, group in enumerate(order):
        block = features[:, groups == group]
        indicators[:, i] = np.isnan(block).any(axis=1).astype("float32")
    return indicators, [f"missing_{g}" for g in order]


def drop_empty_rows(features, max_missing=MAX_MISSING_FRACTION):
    """Keep rows that have at least some predictors. Mostly removes sea."""
    fraction = np.isnan(features).mean(axis=1)
    return fraction <= max_missing


class FeatureScaler:
    """Median imputation + z-scoring, fitted on training rows only.

    Fitting on all rows would leak the validation fold's distribution into
    training. XGBoost does not care about either step, which is how the
    baseline gets away with skipping them; a network needs both.
    """

    def __init__(self, clip_sigma=CLIP_SIGMA):
        self.clip_sigma = clip_sigma
        self.median = None
        self.mean = None
        self.scale = None

    def fit(self, features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            self.median = np.nanmedian(features, axis=0)
            self.mean = np.nanmean(features, axis=0)
            self.scale = np.nanstd(features, axis=0)
        self.median = np.nan_to_num(self.median, nan=0.0)
        self.mean = np.nan_to_num(self.mean, nan=0.0)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 1e-8),
                              self.scale, 1.0)
        return self

    def transform(self, features):
        filled = np.where(np.isnan(features), self.median, features)
        scaled = (filled - self.mean) / self.scale
        scaled = np.clip(scaled, -self.clip_sigma, self.clip_sigma)
        return np.nan_to_num(scaled, nan=0.0).astype("float32")

    def fit_transform(self, features):
        return self.fit(features).transform(features)


# ─────────────────────────────────────────────────────
# EXCLUSION MASK (applied to output, never to input)
# ─────────────────────────────────────────────────────
EXCLUSION_BITS = {"ocean": 1, "inland_water": 2, "permanent_snow_ice": 4,
                  "built_up": 8, "closed_canopy_forest": 16, "cropland": 32}
HARD_EXCLUSION_BITS = 1 | 2 | 4 | 8     # nothing can be planted here
ADVISORY_BITS = {"closed_canopy_forest": 16, "cropland": 32}
# 16 and 32 are advisory, not disqualifying: closed-canopy forest is not
# unsuitable land, it is land that has already proven it grows trees and simply
# does not need planting. Cropland is a land-use decision, not an ecological one.


def load_exclusion_bits(path):
    """Bit codes from the raster's tags, falling back to the documented set."""
    if not path or not os.path.exists(path):
        return dict(EXCLUSION_BITS)
    with rasterio.open(path) as src:
        raw = src.tags().get("satellite_bits")
    try:
        return {k: int(v) for k, v in json.loads(raw).items()} if raw \
            else dict(EXCLUSION_BITS)
    except (ValueError, TypeError):
        return dict(EXCLUSION_BITS)


def read_exclusion_mask(grid, extent=None, path=None):
    """Read the bit-coded mask over an extent, resampling onto this grid."""
    path = path or GRID_PROFILES[grid["profile"]]["exclusion_mask"]
    r0, r1, c0, c1 = extent_to_window(extent, grid)
    if not path or not os.path.exists(path):
        warnings.warn(f"{path} not found; no exclusions will be applied")
        return np.zeros((r1 - r0, c1 - c0), dtype="uint8"), (r0, r1, c0, c1)

    full, nodata = read_onto_grid(path, grid, Resampling.nearest)
    mask = np.asarray(full, dtype="int32")[r0:r1, c0:c1]
    if nodata is not None:
        mask[mask == int(nodata)] = int(HARD_EXCLUSION_BITS)  # unknown->exclude
    return mask.astype("uint8"), (r0, r1, c0, c1)


def sample_exclusion_mask(lat, lon, grid, path=None):
    """Mask value at scattered points, for post-processing point predictions."""
    mask, (r0, _, c0, _) = read_exclusion_mask(grid, path=path)
    row, col, inside = lonlat_to_rowcol(lat, lon, grid)
    values = np.zeros(len(row), dtype="uint8")
    values[inside] = mask[row[inside] - r0, col[inside] - c0]
    return values


def apply_exclusions(suitability, mask, hard_bits=HARD_EXCLUSION_BITS):
    """Blank hard-excluded cells; report which advisory flags remain.

    Returns (suitability with hard exclusions set to NaN, {flag: bool array}).
    """
    mask = np.asarray(mask).astype("int32")
    hard = (mask & int(hard_bits)) > 0
    out = np.asarray(suitability, dtype="float32").copy()
    out[hard] = np.nan
    advisory = {name: ((mask & bit) > 0) & ~hard
                for name, bit in ADVISORY_BITS.items()}
    return out, advisory


# ─────────────────────────────────────────────────────
# SELF-CHECK
# ─────────────────────────────────────────────────────
def main():
    """Print the wired-in feature set for every profile. No training."""
    import argparse
    parser = argparse.ArgumentParser(description="inspect the feature layer")
    parser.add_argument("--profile", default=None,
                        choices=sorted(GRID_PROFILES))
    args = parser.parse_args()
    profiles = [args.profile] if args.profile else sorted(GRID_PROFILES)

    for profile in profiles:
        divider(f"PROFILE {profile}  -  {GRID_PROFILES[profile]['label']}")
        if not os.path.exists(GRID_PROFILES[profile]["template"]):
            print(f"template missing ({GRID_PROFILES[profile]['template']}), "
                  f"skipped")
            continue
        grid = load_grid(profile)
        print(f"grid             {grid['width']}x{grid['height']}  "
              f"{grid['crs']}  cell {grid['cell']:.10f} deg  "
              f"origin ({grid['transform'].c}, {grid['transform'].f})")
        catalog = build_catalog(profile)
        describe_catalog(catalog, profile)
        check_grid(catalog, grid)
        report_unused_layers(catalog, profile)
        names = feature_names(catalog)
        print(f"terrain columns  "
              f"{[n for n, g in zip(names, feature_groups(catalog)) if g == 'terrain']}")

    divider("VEGETATION GUARD  -  the check that used to have a hole")
    for path in ["country_data/FRA/satellite_download/ndvi_pku_2021.tif",
                 "satellite/landcover_tree_frac_10m.tif",
                 "country_data/FRA/provenance/source_flag_30s.tif"]:
        try:
            assert_no_vegetation_layers([{"name": "probe", "path": path}])
            print(f"NOT BLOCKED (bug) {path}")
        except SystemExit:
            print(f"blocked          {path}")


if __name__ == "__main__":
    main()
