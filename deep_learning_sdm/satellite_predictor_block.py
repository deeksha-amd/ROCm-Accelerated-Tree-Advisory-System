"""
Satellite layers resampled onto the predictor grid — FOR ONE EXPERIMENT ONLY

What this is
------------
`country_data/FRA/satellite_download/` holds NDVI and ESA CCI land cover at
their sources' native resolutions, deliberately **off** the predictor grid and
marked `DO_NOT_TRAIN_ON_THIS.md`, so that nothing can be stacked with the
predictors by accident. That was the right default and it stays the default.

This script exists to run the controlled experiment that tests the decision
instead of assuming it: put those layers onto the 1 km France predictor grid so
that `deep_sdm_satellite_experiment.py` can train with and without them under
otherwise identical conditions. The output directory carries its own
`DO_NOT_TRAIN_ON_THIS.md`, so `sdm_features.assert_no_vegetation_layers()`
still refuses it twice over — once on the path token `satellite`, once on the
marker — and only the explicit `ALLOW_VEGETATION_PREDICTORS_EXPERIMENT` opt-in
lets it through. Nothing here is wired into the production path.

Why the layers are rebuilt rather than resampled from `satellite/`
------------------------------------------------------------------
The global 22-layer block is on the 18.5 km grid. Upsampling it 20x20 onto the
1 km grid would hand the experiment land cover that is genuinely 18.5 km, which
would understate satellite data rather than test it: the fair version of "does
satellite data help" uses the best satellite data available at the run's
resolution. ESA CCI PFT is natively 300 m, which aggregates to a true 1 km
fraction, so the France files are aggregated directly.

Order of operations matters, as it did for slope: non-linear summaries
(NDVI peak, amplitude, growing-season mean) are computed at the source's own
resolution and only then put on the grid. Averaging first would clip the peak
and flatten the amplitude in every cell that mixes cover types.

Honest resolution of each output
--------------------------------
    landcover_*      genuinely ~1 km   300 m PFT, area-mean of 3x3 native cells
    ndvi_*           genuinely ~9.3 km 1/12 deg, nearest onto 1 km
    soilmoisture_*   genuinely ~28 km  0.25 deg source, read from the 18.5 km
                                       global block, nearest onto 1 km

Only the land-cover block is real 1 km data. The two vegetation-index and
soil-moisture groups are stored at 1 km because the feature sampler needs one
grid, not because they carry 1 km detail; upsampling invents no information and
nearest-neighbour at least does not pretend to.

What is NOT written, on purpose
-------------------------------
Three layers that exist in the global block are left out, because including
them would make the experiment easier to pass rather than more informative:

    planting_exclusion_mask   it is the output filter, and bit 16 is literally
                              "closed canopy forest" — a thresholded copy of
                              the label this project is trying to predict
    ndvi_observed_semimonths  a count of retrievable composites, i.e. metadata
    soilmoisture_months_obs   its own source tag says retrieval gaps track
                              canopy density, so the coverage count is a
                              back-door tree-cover layer

`landcover_majority_class` is also skipped: it is a categorical code (10 tree,
20 shrub, ... 80 water) and feeding integer class codes to a network as one
linear column asserts an ordering that does not exist. The fractions it is
derived from are all present anyway.

Usage
-----
    ./venv/bin/python satellite_predictor_block.py                # fra_30s
    ./venv/bin/python satellite_predictor_block.py --profile fra_10m
"""

import argparse
import json
import os
import time

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

import sdm_features as feat

# ─────────────────────────────────────────────────────
# CONFIG — where the native-resolution downloads live
# ─────────────────────────────────────────────────────
DOWNLOAD_DIR = "country_data/FRA/satellite_download"
PFT_DIR = os.path.join(DOWNLOAD_DIR, "landcover_pft_2020")
NDVI_FILES = ["ndvi_pku_2021.tif", "ndvi_pku_2022.tif"]
# No France soil moisture was downloaded (the manifest records files: 0), so it
# comes from the global block, which is itself an upsample of a 0.25 deg source.
GLOBAL_SOILMOISTURE = {"soilmoisture_mean": "satellite/soilmoisture_mean_10m.tif",
                       "soilmoisture_amplitude":
                           "satellite/soilmoisture_amplitude_10m.tif"}

OUTPUT_ROOT = "satellite_experiment"

# ─────────────────────────────────────────────────────
# CONFIG — land cover
# ─────────────────────────────────────────────────────
# Same composition as `satellite_rasters.PFT_COMPONENTS`, so a layer called
# landcover_tree_frac means the same thing here as in the global block.
PFT_COMPONENTS = {
    "tree":            ["trees-bd", "trees-be", "trees-nd", "trees-ne"],
    "tree_broadleaf":  ["trees-bd", "trees-be"],
    "tree_needleleaf": ["trees-nd", "trees-ne"],
    "shrub":           ["shrubs-bd", "shrubs-be", "shrubs-nd", "shrubs-ne"],
    "grass_natural":   ["grass-nat"],
    "cropland":        ["grass-man"],   # "managed grass" is CCI's cropland class
    "builtup":         ["built"],
    "bare":            ["bare"],
    "water":           ["water_inland"],
    "snowice":         ["snowice"],
}
# The subset that partitions the land surface. Measured on the France files:
# these sum to exactly 100 on land and exactly 0 at sea, which is what makes
# the sea separable here without the WATER_OCEAN variable the global build used.
PFT_PARTITION = ["tree", "shrub", "grass_natural", "cropland",
                 "builtup", "bare", "water", "snowice"]
MIN_LAND_PCT = 50.0      # below this a target cell is called sea and left NaN

# ─────────────────────────────────────────────────────
# CONFIG — NDVI
# ─────────────────────────────────────────────────────
NDVI_SCALE = 0.001
NDVI_FILL = 65535
NDVI_SEMIMONTHS = 24
NDVI_GROWING_SEMIMONTHS = 12   # of 24: the greenest half of the year
NDVI_MIN_SEMIMONTHS = 8        # of 24, matching the global build

divider = feat.divider


# ─────────────────────────────────────────────────────
# LAND COVER — 300 m to the predictor grid
# ─────────────────────────────────────────────────────
def block_mean(array, factor):
    """Mean of each factor x factor block. Exact integer decimation only."""
    height, width = array.shape
    return (array.reshape(height // factor, factor, width // factor, factor)
            .mean(axis=(1, 3)))


def aggregate_landcover(grid):
    """Area-mean the 14 PFT variables onto the grid, then normalise to land.

    Fractions come out as "percent of the non-sea part of the cell", which is
    the global block's convention. Normalising by the partition sum rather than
    dividing by the cell area keeps a half-coastal cell describing the land in
    it instead of reporting every cover type at half strength.
    """
    variables = sorted({v for group in PFT_COMPONENTS.values() for v in group})
    paths = {v: os.path.join(PFT_DIR, f"pft_{v}.tif") for v in variables}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"PFT variables not on disk: {missing}")

    with rasterio.open(paths[variables[0]]) as src:
        native_width, native_height = src.width, src.height
        native_transform = src.transform
    factor = int(round(abs(grid["transform"].a) / abs(native_transform.a)))
    if factor < 1 or native_width % factor or native_height % factor:
        raise SystemExit(
            f"300 m PFT ({native_width}x{native_height}) does not decimate "
            f"exactly by {factor} onto the {grid['profile']} grid")

    covered_width = native_width // factor
    covered_height = native_height // factor
    print(f"land cover       300 m {native_width}x{native_height} -> "
          f"{factor}x{factor} area-mean -> {covered_width}x{covered_height}")
    if covered_width < grid["width"] or covered_height < grid["height"]:
        short = grid["width"] - covered_width
        print(f"                 the download stops "
              f"{short} column(s) short of the grid's east edge "
              f"({grid['width']}); those cells stay NaN and are handled by the "
              f"missingness indicator")

    # One variable at a time: 5520x3780 int8 is 21 MB, and holding all 14 plus
    # their float accumulators at once is pointless when each is used once.
    native_means = {}
    for name in variables:
        with rasterio.open(paths[name]) as src:
            band = src.read(1).astype("float32")
        native_means[name] = block_mean(band, factor)
        del band

    layers = {}
    for name, components in PFT_COMPONENTS.items():
        layers[name] = sum(native_means[c] for c in components)
    land_pct = sum(layers[name] for name in PFT_PARTITION)

    is_land = land_pct >= MIN_LAND_PCT
    out = {}
    for name in PFT_COMPONENTS:
        with np.errstate(invalid="ignore", divide="ignore"):
            normalised = 100.0 * layers[name] / np.where(land_pct > 0,
                                                         land_pct, np.nan)
        out[f"landcover_{name}_frac"] = np.where(is_land, normalised, np.nan)
    # Kept because the global block publishes it and it is not a vegetation
    # measure: it is how much of the cell is not sea.
    out["landcover_land_frac"] = np.where(land_pct > 0, land_pct, np.nan)

    for name in out:
        out[name] = pad_to_grid(out[name].astype("float32"), grid)
    print(f"                 {int(is_land.sum()):,} land cells of "
          f"{is_land.size:,} in the covered window; partition sum "
          f"{np.nanmin(land_pct[is_land]):.1f}-{np.nanmax(land_pct[is_land]):.1f}%")
    return out


def pad_to_grid(array, grid):
    """Place a smaller, origin-aligned array into a full-grid array of NaN."""
    if array.shape == (grid["height"], grid["width"]):
        return array
    full = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    full[:array.shape[0], :array.shape[1]] = array
    return full


# ─────────────────────────────────────────────────────
# NDVI — 1/12 degree climatology to the predictor grid
# ─────────────────────────────────────────────────────
def ndvi_statistics(grid):
    """Semi-monthly climatology at 1/12 deg, four summaries, then onto the grid.

    One annual mean cannot separate an evergreen stand from a pasture that is
    bare for half the year, so the same four statistics the global block uses
    are kept. All four are computed before any spatial resampling.
    """
    paths = [os.path.join(DOWNLOAD_DIR, name) for name in NDVI_FILES]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit(f"no NDVI files in {DOWNLOAD_DIR}")

    with rasterio.open(paths[0]) as src:
        native_transform, native_crs = src.transform, src.crs
        shape = (src.height, src.width)
    total = np.zeros((NDVI_SEMIMONTHS,) + shape, dtype="float32")
    count = np.zeros((NDVI_SEMIMONTHS,) + shape, dtype="uint8")
    for path in paths:
        with rasterio.open(path) as src:
            raw = src.read()
        if len(raw) != NDVI_SEMIMONTHS:
            raise SystemExit(f"{path} has {len(raw)} bands, expected "
                             f"{NDVI_SEMIMONTHS}")
        good = raw != NDVI_FILL
        total += np.where(good, raw, 0).astype("float32") * NDVI_SCALE
        count += good
        del raw, good

    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    del total, count

    observed = np.isfinite(clim).sum(axis=0)
    usable = observed >= NDVI_MIN_SEMIMONTHS
    with np.errstate(invalid="ignore"):
        mean_native = np.nanmean(clim, axis=0)
        peak_native = np.nanmax(clim, axis=0)
        trough_native = np.nanmin(clim, axis=0)
        ranked = np.sort(np.where(np.isfinite(clim), clim, -np.inf), axis=0)
        green = np.where(np.isfinite(ranked[-NDVI_GROWING_SEMIMONTHS:]),
                         ranked[-NDVI_GROWING_SEMIMONTHS:], np.nan)
        growing_native = np.nanmean(green, axis=0)

    print(f"ndvi             {len(paths)} year(s), {shape[1]}x{shape[0]} at "
          f"1/12 deg, {int(usable.sum()):,} of {usable.size:,} cells have "
          f">= {NDVI_MIN_SEMIMONTHS} of 24 composites")

    out = {}
    for name, native in [("observed_mean", mean_native),
                         ("peak", peak_native),
                         ("growing_season_mean", growing_native),
                         ("seasonal_amplitude", peak_native - trough_native)]:
        field = np.where(usable, native, np.nan).astype("float32")
        out[f"ndvi_{name}"] = put_on_grid(field, native_transform, native_crs,
                                          grid)
    return out


def put_on_grid(array, src_transform, src_crs, grid):
    """Resample an array onto the profile grid.

    Nearest when going finer, area-average when going coarser. Nearest on the
    way up is the point: NDVI is 9.3 km data and a 1 km cell inherits its
    parent's value unchanged rather than acquiring interpolated detail that the
    source never measured.
    """
    finer = abs(grid["transform"].a) < abs(src_transform.a)
    out = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    reproject(source=np.ascontiguousarray(array, dtype="float32"),
              destination=out,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=grid["transform"], dst_crs=grid["crs"],
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.nearest if finer else Resampling.average)
    return out


# ─────────────────────────────────────────────────────
# SOIL MOISTURE — the one non-greenness layer
# ─────────────────────────────────────────────────────
def soil_moisture(grid):
    """Read the global 18.5 km soil-moisture layers onto the grid."""
    out = {}
    for name, path in GLOBAL_SOILMOISTURE.items():
        if not os.path.exists(path):
            print(f"                 {path} missing, skipped")
            continue
        band, _ = feat.read_onto_grid(path, grid, Resampling.nearest)
        band = np.asarray(band, dtype="float32")
        band[band < feat.NODATA_FLOOR] = np.nan
        out[name] = band
    if out:
        first = next(iter(out.values()))
        print(f"soil moisture    {len(out)} layer(s) from the global 18.5 km "
              f"block (0.25 deg source), "
              f"{100 * np.isfinite(first).mean():.1f}% of grid cells covered")
    return out


# ─────────────────────────────────────────────────────
# WRITING
# ─────────────────────────────────────────────────────
ROLE_TAGS = {
    "landcover": ("CIRCULAR CANDIDATE — current land cover. Tested as a "
                  "predictor by deep_sdm_satellite_experiment.py only."),
    "ndvi": ("CIRCULAR CANDIDATE — current greenness. Tested as a predictor "
             "by deep_sdm_satellite_experiment.py only."),
    "soilmoisture": ("DEFENSIBLE CANDIDATE — microwave retrieval, not a "
                     "greenness proxy. Coarsest layer in the block."),
}


def write_block(layers, grid, output_dir, provenance):
    """Write every layer as a single-band float32 GeoTIFF on the grid."""
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(layers):
        group = name.split("_")[0]
        path = os.path.join(output_dir, f"{name}.tif")
        with rasterio.open(path, "w", driver="GTiff", height=grid["height"],
                           width=grid["width"], count=1, dtype="float32",
                           crs=grid["crs"], transform=grid["transform"],
                           nodata=np.nan, compress="deflate",
                           tiled=True) as dst:
            dst.write(layers[name], 1)
            dst.update_tags(
                experiment_layer=name,
                experiment_role=ROLE_TAGS.get(group, "experiment only"),
                experiment_only="true",
                not_a_production_predictor="true",
                source=provenance.get(group, "see MANIFEST.json"),
                grid=grid["profile"])
        coverage = 100 * np.isfinite(layers[name]).mean()
        print(f"   {name:<34s} {coverage:5.1f}% covered  {path}")


MARKER_TEXT = """\
# Do not train on anything in this directory

Same rule as `country_data/FRA/satellite_download/`, and for the same reason.
This directory only differs in being on the 1 km predictor grid, which makes it
*more* dangerous, not less: these files can be stacked with the predictors
without anything complaining about shape.

They exist for one controlled experiment, `deep_sdm_satellite_experiment.py`,
which measures what happens when satellite layers are used as predictors and in
particular whether any gain survives on land with no existing tree cover. The
production path (`deep_sdm_training.py` + `sdm_features.py`) must never read
them: `sdm_features.assert_no_vegetation_layers()` blocks this directory twice,
on the path token `satellite` and on this marker, and only the explicit
`ALLOW_VEGETATION_PREDICTORS_EXPERIMENT` opt-in lifts the block.

`landcover_tree_frac.tif` and `landcover_cropland_frac.tif` have a second,
legitimate job: they define the tree-cover strata the experiment evaluates on.
Used that way they are evaluation metadata, not inputs.
"""


def write_manifest(layers, grid, output_dir, elapsed):
    manifest = {
        "block": "satellite predictors, EXPERIMENT ONLY",
        "written_by": "satellite_predictor_block.py",
        "consumed_by": "deep_sdm_satellite_experiment.py",
        "on_predictor_grid": True,
        "production_predictor": False,
        "why_it_exists": (
            "The project excluded vegetation layers from the predictor set on "
            "measured grounds (vegetation alone AUC 0.791 vs 0.711 for the "
            "whole environmental set, but +0.001 on cleared farmland). This "
            "block lets that finding be re-tested on the France 1 km data "
            "rather than assumed."),
        "profile": grid["profile"],
        "grid": {"width": grid["width"], "height": grid["height"],
                 "crs": grid["crs"], "transform": list(grid["transform"])[:6]},
        "layers": sorted(layers),
        "coverage_pct": {name: round(float(100 * np.isfinite(layers[name]).mean()), 2)
                         for name in sorted(layers)},
        "honest_resolution": {
            "landcover_*": "~1 km, area-mean of 3x3 native 300 m ESA CCI PFT",
            "ndvi_*": "~9.3 km, 1/12 deg PKU GIMMS, nearest onto 1 km",
            "soilmoisture_*": "~28 km, 0.25 deg ESA CCI, via the 18.5 km block",
        },
        "deliberately_excluded": {
            "planting_exclusion_mask": "output filter; bit 16 is a thresholded "
                                       "closed-canopy-forest label",
            "ndvi_observed_semimonths": "composite count, metadata not signal",
            "soilmoisture_months_observed": "retrieval gaps track canopy "
                                            "density, so it is a back-door "
                                            "tree-cover layer",
            "landcover_majority_class": "categorical class codes carry a false "
                                        "ordering as one linear column",
        },
        "sources": {
            "landcover": "ESA CCI Land Cover PFT v2.0.81, epoch 2020, 300 m",
            "ndvi": "PKU GIMMS NDVI V1.2, 2021-2022, 1/12 deg, semi-monthly",
            "soilmoisture": "ESA CCI Soil Moisture v09.2 COMBINED, 0.25 deg",
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(os.path.join(output_dir, "MANIFEST.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    with open(os.path.join(output_dir, feat.DO_NOT_TRAIN_MARKER), "w") as handle:
        handle.write(MARKER_TEXT)
    return manifest


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def output_dir_for(profile):
    return os.path.join(OUTPUT_ROOT, f"FRA_{profile.split('_')[-1]}")


def build(profile):
    grid = feat.load_grid(profile)
    output_dir = output_dir_for(profile)
    print(f"profile          {profile}  "
          f"({feat.GRID_PROFILES[profile]['label']})")
    print(f"grid             {grid['width']}x{grid['height']} @ "
          f"{grid['cell']:.10f} deg, origin "
          f"({grid['transform'].c}, {grid['transform'].f})")
    print(f"output           {output_dir}/")

    started = time.time()
    layers = {}
    divider("1. LAND COVER  -  ESA CCI PFT, 300 m native")
    layers.update(aggregate_landcover(grid))
    divider("2. NDVI  -  PKU GIMMS, 1/12 deg native")
    layers.update(ndvi_statistics(grid))
    divider("3. SOIL MOISTURE  -  ESA CCI, 0.25 deg native")
    layers.update(soil_moisture(grid))

    divider("4. WRITING")
    write_block(layers, grid, output_dir, {
        "landcover": "ESA CCI Land Cover PFT v2.0.81, 2020, 300 m",
        "ndvi": "PKU GIMMS NDVI V1.2, 2021-2022, 1/12 deg",
        "soilmoisture": "ESA CCI Soil Moisture v09.2, 0.25 deg"})
    elapsed = time.time() - started
    write_manifest(layers, grid, output_dir, elapsed)

    divider("5. CHECKS")
    catalog = [{"name": n, "group": "satellite", "kind": "linear",
                "path": os.path.join(output_dir, f"{n}.tif")}
               for n in sorted(layers)]
    feat.check_grid(catalog, grid)
    try:
        feat.assert_no_vegetation_layers(catalog)
        print("GUARD FAILED     the vegetation guard did not block this block")
        raise SystemExit(1)
    except SystemExit as stop:
        if str(stop) == "1":
            raise
        print("guard            still blocks every layer written here "
              "(path token + DO_NOT_TRAIN marker)")

    tree = layers["landcover_tree_frac"]
    crop = layers["landcover_cropland_frac"]
    finite = np.isfinite(tree)
    print(f"tree cover       mean {np.nanmean(tree):.1f}%, "
          f"{100 * (tree[finite] <= 10).mean():.1f}% of land cells at or "
          f"below 10%")
    print(f"cropland         mean {np.nanmean(crop):.1f}%, "
          f"{100 * (crop[finite] >= 50).mean():.1f}% of land cells at or "
          f"above 50%")
    print(f"\nelapsed          {elapsed:.1f}s")
    return layers


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="resample France satellite downloads onto the predictor "
                    "grid, for the satellite-as-predictor experiment only")
    parser.add_argument("--profile", default="fra_30s",
                        choices=["fra_10m", "fra_30s"])
    args = parser.parse_args(argv)

    divider("SATELLITE PREDICTOR BLOCK  -  EXPERIMENT ONLY, NOT PRODUCTION")
    print("These layers measure where trees already grow. They are built here "
          "so that the")
    print("decision to exclude them can be re-tested on the France data, not "
          "so they can be")
    print("used. Nothing in the production path reads this directory.\n")
    build(args.profile)


if __name__ == "__main__":
    main()
