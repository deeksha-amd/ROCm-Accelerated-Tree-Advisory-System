"""
Sampling-effort surface and target-group background points

Two products, on the shared 2160x1080 / 1-6-degree grid:

  sampling_effort/plant_effort_10m.tif   how many vascular-plant (Tracheophyta)
                                         GBIF records exist in each cell —
                                         a map of where botanists have looked
  sampling_effort/background_points.csv  pseudo-absences drawn with probability
                                         proportional to that surface

WHY UNIFORM BACKGROUND SAMPLING IS WRONG HERE
---------------------------------------------
A presence-only model learns the contrast between where the species was
recorded and where the background points are. If background is drawn uniformly
over land, that contrast includes everything that determines whether anyone
went looking — not just whether the tree can grow.

GBIF presences cluster along roads, near cities and inside rich countries.
This project already measured the consequence: built-up fraction came out as
the strongest single separator of presence from uniform background, at 0.246 —
ahead of every climate, soil and terrain variable. Nothing about tree ecology
makes "near a city" the top predictor of tree habitat. That number is survey
effort, and a model fed uniform background will spend its capacity learning it,
then recommend planting in car parks.

The target-group fix (Phillips et al. 2009) is to draw background from the same
sampling process that produced the presences. Anyone who recorded an oak was a
botanist recording vascular plants, so the density of ALL Tracheophyta records
is a direct estimate of botanical survey effort. Sampling background in
proportion to it means presence and background share the same bias, and the
contrast between them is what is left: environment.

This replaces `generate_pseudo_absences()` in xgboost_training.py, which samples
uniformly over valid land pixels. That function is deliberately left in place —
run this script with --compare to see the difference it makes before switching.

HOW THE SURFACE IS BUILT WITHOUT DOWNLOADING 500 MILLION RECORDS
----------------------------------------------------------------
There are 523 million Tracheophyta occurrence records with usable coordinates.
Counting them per cell through the occurrence API would need one query per cell
(2.3 million queries) or an adaptive quadtree (tens of thousands). Instead this
reads GBIF's own pre-aggregated map service: /v2/map/occurrence/density returns
Mapbox Vector Tiles of per-pixel record counts. At zoom 3 that is 128 tiles,
about 50 MB and a couple of minutes for the whole planet, and the clipped sums
reproduce the occurrence-search counts to within a fraction of a percent.
gbif_api.py holds the tile reader and the verification notes.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

import gbif_api
from gbif_api import (GbifClient, OCCURRENCE_FILTERS, TRACHEOPHYTA_KEY,
                      density_tiles, tile_grid, tiles_covering)
from download_species import COUNTRY, COUNTRY_BOUNDS, DEFAULT_BOUNDS, paths

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUTPUT_DIR = "sampling_effort"
TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"      # the 10 arcmin grid
LAND_RASTER = "satellite/landcover_land_frac_10m.tif"
BUILTUP_RASTER = "satellite/landcover_builtup_frac_10m.tif"   # for --compare

# ── WHICH GRID THE EFFORT SURFACE IS BUILT ON ──
#
# This used to be TEMPLATE and nothing else, which meant every country's
# background was drawn at 18.5 km while its predictors were at 1 km. A
# target-group correction on a grid 400x coarser than the model only cancels
# bias BETWEEN 18.5 km blocks; inside one, the surveyors still walked the roads
# and the village edges and the model can still learn "near a town" as habitat.
# That was measured, not assumed: presence cells averaged 9.93% built-up
# against 2.80% for background (21.9% vs 5.3% on treeless land), and built-up
# fraction alone was still worth +0.008 AUC overall and +0.016 on treeless
# land. The supplied background also collapsed to 2,373 distinct cells on the
# 1 km grid against 69,540 presence cells, so the trainer was resampling it.
#
# "30s" is now the default for a country that has a prepared data tree: the
# grid is read from country_data/<ISO3>/country_grid.json, the same authority
# every other country block adopts, so effort lands cell-for-cell on the
# predictors. "10m" keeps the old global-template behaviour exactly, because
# the fra_10m training profile still uses it and the two resolutions have to
# stay comparable.
RESOLUTIONS = ("30s", "10m")
DEFAULT_RESOLUTION = "30s"
COUNTRY_DATA_ROOT = "country_data"

# The target group. Tracheophyta = vascular plants: every tree, and every
# botanist's other records. Kingdom Plantae would drag in mosses and algae,
# whose recorders are a different community visiting different places.
TARGET_GROUP_KEY = TRACHEOPHYTA_KEY
TARGET_GROUP_NAME = "Tracheophyta"

# Restricting the tile pyramid to one country makes a much finer zoom
# affordable: the world at zoom 6 is 8,192 tiles, France is 24. Zoom 6 pixels
# are 0.0055 deg (~600 m), so roughly 1,100 tile pixels fall inside each
# 1/6-degree target cell and the per-cell effort total is well resolved rather
# than being one coarse tile pixel smeared across several cells.
ZOOM = 6

# At 30 arcsec a cell is 0.00833 deg (~930 m), so zoom 6's 600 m tile pixels
# are barely finer than the target and the surface would be blocky at 600 m.
# Zoom 8 pixels are 0.00137 deg (~150 m), ~37 per target cell, which is the
# closest affordable analogue of the 1,100-per-cell margin the coarse grid
# enjoys. Cost scales as 4^zoom: France is 345 tiles at zoom 8 against 24 at
# zoom 6, and the lower 48 is 3,108.
ZOOM_BY_RESOLUTION = {"10m": 6, "30s": 8}

# Background pool for one country; the trainer subsamples per species from it.
# The fine grid gets more points because the pool it is drawn from is 400x
# larger: 20,000 points over ~930,000 eligible 1 km cells in France would be
# thinner than the 69,540 cells the presences occupy, which is the imbalance
# that made the trainer discard the supplied background in the first place.
N_BACKGROUND = 20_000
N_BACKGROUND_BY_RESOLUTION = {"10m": 20_000, "30s": 50_000}

# ── SPARSE COUNTS AT 1 km, AND WHY THE WEIGHT IS SMOOTHED ──
#
# At 18.5 km France holds ~2,600 cells and ~92 M vascular-plant records, so
# every cell has tens of thousands and the raw count is a precise measurement.
# At 1 km the same records spread over ~930,000 cells. The MEAN is still ~100,
# but the distribution is extremely skewed and a large share of cells hold
# zero or one record -- so the raw count stops being a measurement of effort
# and becomes a Poisson draw from it, with most of the variance being noise.
#
# Sampling straight from raw counts at 1 km would therefore do the thing this
# fix exists to prevent: it would degenerate towards a near-binary "has a
# record or not" mask, which concentrates background on exactly the roadside
# and village cells that carry the bias, and is worse than the coarse version.
#
# So at 30 arcsec the SAMPLING WEIGHT is a smoothed intensity, not the raw
# count. Survey effort is a latent continuous field -- how much botanical
# attention a neighbourhood gets -- and the right estimator of a field sampled
# this thinly is a kernel smoother. Smoothing over a few km keeps the real
# structure (cities, reserves, transport corridors, the Alps being walked more
# than the Beauce) while averaging out the Poisson noise that separates one
# roadside cell from its neighbour.
#
# The RAW counts are still written to plant_effort_<C>_30s.tif unchanged, so
# that file stays an honest record-count map and anything reading it keeps the
# meaning it had. The smoothed field goes to a second file and is what the
# draw uses.
#
# HOW WIDE, AND WHY IT IS NOT A CONSTANT
#
# Smoothing is not free. It blurs away exactly the town-and-roadside structure
# that the presences have and that the background is supposed to reproduce, so
# too much of it makes the correction WORSE than no smoothing at all. Measured
# on France, presence cells against background cells on 1 km built-up fraction
# (the support the model actually sees):
#
#     kernel     zero cells   top 1% of      background   gap to    AUC
#     sigma      in-country   weight holds   built-up     presence
#     ------     ----------   ------------   ----------   -------   -----
#     uniform            --             --        2.38%     +5.87      --
#     0 km            22.7%          54.3%        5.78%     +2.47   0.533
#     1 km             0.0%          14.0%        5.11%     +3.14   0.551
#     2 km             0.0%           8.4%        4.87%     +3.38   0.556
#     5 km             0.0%           6.6%        4.60%     +3.65   0.559
#     (old 18.5 km grid, for reference)           5.10%     +3.16   0.557
#
# Every km of extra smoothing moves the background further from the presences.
# So the rule is: smooth only where the counts are too sparse to be an
# estimate at all, and no further. France has records in 77% of its 1 km cells
# -- there the raw count already IS the effort field and the EFFORT_FLOOR is
# what keeps the unrecorded quarter reachable. A country with three times
# fewer records over twelve times the area is a different case, and the ladder
# below widens the kernel for it automatically instead of asking a person to
# remember to.
SMOOTH_KM = None                # None = adaptive; a number forces that sigma
SMOOTH_LADDER_KM = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)
SMOOTH_ZERO_SHARE_TARGET = 0.25  # widen until at most this share of cells are 0
SMOOTH_PASSES = 3               # box passes; 3 is a good Gaussian approximation

# A cell with zero plant records is unsurveyed, not uninhabitable. Pure
# proportional sampling would give it probability exactly zero and so would
# never place a background point anywhere nobody has ever botanised — which
# quietly tells the model that unsurveyed ground is suitable. This additive
# floor, in units of records, leaves such cells reachable but rare.
EFFORT_FLOOR = 0.5

# ...but "0.5 records" is only a small number relative to a cell that holds
# tens of thousands, which is what an 18.5 km cell holds and what this constant
# was chosen against. On the 1 km grid the same 0.5 can be LARGER than a
# typical cell's intensity, and then the floor stops being a floor and becomes
# the signal: measured on the USA, the smoothed median is 0.42 records/cell, so
# an absolute 0.5 floor would make an unsurveyed cell just as likely as a
# median one and flatten the correction back to uniform.
#
# So at 30 arcsec the floor is set RELATIVE to the surface it is flooring: an
# unsurveyed cell is this many times less likely than a median surveyed one,
# whatever the country's record density. 20 is close to what France lands on
# with an absolute 0.5 (18x), so the country that was actually measured keeps
# the behaviour that produced the measurement, and the sparse ones stop being
# governed by a constant calibrated for a 400x coarser grid.
EFFORT_FLOOR_MEDIAN_RATIO = 20.0

# 1.0 = strict target-group sampling. Below 1.0 flattens the surface towards
# uniform, which is the knob to turn if background ends up too concentrated in
# western Europe to cover the environmental space at all.
EFFORT_POWER = 1.0

MIN_LAND_PERCENT = 1.0          # landcover_land_frac is 0-100

# Whether to refuse to put background in a cell that holds a presence.
#
# False, and deliberately so. It was True, which is defensible globally but
# wrong here: France holds only ~2,600 cells at 18.5 km and the 69 species
# between them occupy 2,513 of them, so excluding any cell with a presence
# left 86 usable cells out of 2,599. It is also wrong in principle for a
# multi-species file — a cell where oak grows is a perfectly good background
# cell for pine. Background represents the available environment, and in
# target-group background and MaxEnt practice it may coincide with presences.
# Per-species exclusion, if wanted at all, belongs in the trainer, which knows
# which species it is fitting.
EXCLUDE_PRESENCE_CELLS = False

RANDOM_SEED = 42

# sdm_features.py looks for background at a fixed, country-free path (its
# BACKGROUND_CANDIDATES list), and falls back to a uniform sampler when it is
# missing. The country-scoped file stays the master copy; this is the one the
# trainer reads.
CANONICAL_BACKGROUND = os.path.join(OUTPUT_DIR, "background_points.csv")


def country_files(country=None, resolution="10m"):
    """Output paths, carrying the country AND the resolution.

    The 10m names are exactly what they always were, so France's existing
    plant_effort_FR_10m.tif, background_FR.csv and MANIFEST_FR.json are neither
    renamed nor overwritten when a 1 km surface is built beside them. That
    matters here beyond tidiness: another agent's training results were
    produced from those files and have to stay reproducible.
    """
    country = (country or COUNTRY).upper()
    if resolution == "10m":
        effort = f"plant_effort_{country}_10m.tif"
        background = f"background_{country}.csv"
        manifest = f"MANIFEST_{country}.json"
    else:
        effort = f"plant_effort_{country}_{resolution}.tif"
        background = f"background_{country}_{resolution}.csv"
        manifest = f"MANIFEST_{country}_{resolution}.json"
    return {
        "effort": os.path.join(OUTPUT_DIR, effort),
        "smoothed": os.path.join(OUTPUT_DIR,
                                 effort.replace(".tif", "_smoothed.tif")),
        "background": os.path.join(OUTPUT_DIR, background),
        "manifest": os.path.join(OUTPUT_DIR, manifest),
        "presence": os.path.join(paths(country)["dir"],
                                 f"gbif_trees_{country}_thinned.csv"),
    }


def country_bounds(country=None):
    return COUNTRY_BOUNDS.get((country or COUNTRY).upper(), DEFAULT_BOUNDS)


def country_iso3(iso2):
    """ISO3 for a 2-letter GBIF country code, resolved from Natural Earth.

    Not a hardcoded table: country_clip.py already downloads admin-0 for the
    polygon masks and that file carries both code systems, so DE -> DEU works
    as well as ES -> ESP without anyone maintaining a list. Falls back to the
    2-letter code itself, which lets an unknown country fail loudly on a
    missing country_grid.json rather than silently at the wrong extent.
    """
    from country_clip import ensure_boundaries

    wanted = iso2.upper()
    with open(ensure_boundaries(), encoding="utf-8") as handle:
        collection = json.load(handle)
    for feature in collection["features"]:
        props = feature["properties"]
        for key in ("ISO_A2", "ISO_A2_EH", "WB_A2"):
            if str(props.get(key, "")).upper() == wanted:
                for out_key in ("ADM0_A3", "ISO_A3", "ISO_A3_EH", "SOV_A3"):
                    code = props.get(out_key)
                    if code not in (None, "", "-99", -99):
                        return str(code).upper()
    return wanted


# ─────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────
def read_template(path):
    with rasterio.open(path) as src:
        return {"width": src.width, "height": src.height,
                "transform": src.transform, "crs": src.crs}


def target_grid(country=None, resolution=DEFAULT_RESOLUTION, template=None):
    """The grid the effort surface is built on, plus the country polygon.

    resolution="10m" reproduces the old behaviour bit for bit: the global
    climate template, no polygon, the bounding box from COUNTRY_BOUNDS.

    resolution="30s" reads country_data/<ISO3>/country_grid.json -- the file
    country_climate_rasters.py writes and every other country block adopts --
    so the effort surface lands cell-for-cell on the predictor stack instead of
    400 predictor cells to one effort cell. It also carries the Natural Earth
    polygon, which replaces the "a cell counts as this country if it holds a
    record attributed to it" rule. That rule is tolerable at 18.5 km, where
    France's 92 M plant records leave almost no unrecorded cell, and actively
    destructive at 1 km, where most cells hold nothing and the rule would
    confine background to precisely the surveyed cells whose bias we are
    trying to cancel.
    """
    if template:
        return read_template(template)
    if resolution == "10m":
        return read_template(TEMPLATE)
    if resolution not in RESOLUTIONS:
        raise SystemExit(f"--resolution must be one of {RESOLUTIONS}")

    from country_clip import country_grid as clip_country_grid

    iso3 = country_iso3(country or COUNTRY)
    path = os.path.join(COUNTRY_DATA_ROOT, iso3, "country_grid.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"No {path}. The 30 arcsec effort surface is built on the "
            f"country's own grid, so prepare the rasters first:\n"
            f"    python3 prepare_country.py {iso3} --stage grid")

    with open(path, encoding="utf-8") as handle:
        saved = json.load(handle)

    from affine import Affine
    grid = {"width": saved["width"], "height": saved["height"],
            "transform": Affine(*saved["transform"]),
            "crs": rasterio.crs.CRS.from_string(saved["crs"]),
            "iso3": iso3, "bounds": tuple(saved["bounds"]),
            "res_deg": saved["res_deg"]}

    # The polygon, taken through the same call every other block uses so the
    # border cannot drift between the effort surface and the predictors.
    reference = clip_country_grid(
        iso3, res_deg=saved["res_deg"], margin_deg=0.0,
        bbox=tuple(saved["bbox"]) if saved.get("bbox") else tuple(saved["bounds"]))
    grid["geometry"] = reference["geometry"]
    return grid


def grid_bounds(grid, country=None):
    """(lat_min, lat_max, lon_min, lon_max) for tile fetching and box tests."""
    if "bounds" in grid:
        west, south, east, north = grid["bounds"]
        return (south, north, west, east)
    return country_bounds(country)


def polygon_mask(grid):
    """Burn the country polygon onto the grid, or None if there is no polygon."""
    if "geometry" not in grid:
        return None
    from country_clip import country_mask as rasterize_country
    return rasterize_country({"geometry": grid["geometry"],
                              "height": grid["height"],
                              "width": grid["width"],
                              "transform": grid["transform"]}).astype(bool)


# ─────────────────────────────────────────────────────
# SMOOTHING
# ─────────────────────────────────────────────────────
def _box1d(array, half, axis):
    """Running mean of width 2*half+1 along one axis, edges replicated."""
    if half < 1:
        return array
    n = array.shape[axis]
    pad = [(0, 0)] * array.ndim
    pad[axis] = (half, half)
    padded = np.pad(array, pad, mode="edge")
    cumulative = np.cumsum(padded, axis=axis)
    zeros = np.zeros_like(np.take(cumulative, [0], axis=axis))
    cumulative = np.concatenate([zeros, cumulative], axis=axis)
    hi = [slice(None)] * array.ndim
    lo = [slice(None)] * array.ndim
    hi[axis] = slice(2 * half + 1, 2 * half + 1 + n)
    lo[axis] = slice(0, n)
    return (cumulative[tuple(hi)] - cumulative[tuple(lo)]) / (2 * half + 1)


def smooth_intensity(counts, grid, smooth_km=SMOOTH_KM,
                     passes=SMOOTH_PASSES):
    """Estimate the latent effort field from sparse per-cell counts.

    Three box passes approximate a Gaussian by the central limit theorem, and
    a box filter is a couple of cumulative sums, so this needs no scipy. For n
    passes of width w the variance is n*(w^2-1)/12, which inverts to the width
    below.

    Rows are smoothed with a wider kernel than columns because a degree of
    longitude is shorter than a degree of latitude away from the equator, and
    the kernel is meant to be a distance in km, not in cells.
    """
    if smooth_km <= 0:
        return counts.astype("float64")

    cell_deg = abs(grid["transform"].a)
    mid_lat = grid["transform"].f - grid["height"] / 2 * abs(grid["transform"].e)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * max(np.cos(np.deg2rad(mid_lat)), 0.1)

    sigma_rows = smooth_km / (km_per_deg_lat * cell_deg)
    sigma_cols = smooth_km / (km_per_deg_lon * cell_deg)

    def half_width(sigma):
        width = np.sqrt(12.0 * sigma ** 2 / passes + 1.0)
        return max(0, int(round((width - 1) / 2)))

    half_rows, half_cols = half_width(sigma_rows), half_width(sigma_cols)
    out = counts.astype("float64")
    for _ in range(passes):
        out = _box1d(out, half_rows, axis=0)
        out = _box1d(out, half_cols, axis=1)
    return out


def choose_smoothing(counts, mask, grid, ladder=SMOOTH_LADDER_KM,
                     target=SMOOTH_ZERO_SHARE_TARGET):
    """Smallest kernel on the ladder that leaves at most `target` zero cells.

    Sparsity is what smoothing is for; anything beyond that is blur. See the
    table in CONFIG -- on France every extra km moved the background away from
    the presences, so overshooting is not a harmless safety margin, it is the
    failure mode. Returns (sigma_km, rows) with the whole ladder for the log,
    so the choice is visible rather than buried.
    """
    rows = []
    chosen = ladder[-1]
    for sigma in ladder:
        field = smooth_intensity(counts, grid, sigma) if sigma > 0 else counts
        values = field[mask]
        zero_share = float((values <= 0).mean()) if values.size else 1.0
        ordered = np.sort(values)[::-1]
        top1 = (float(ordered[:max(1, len(ordered) // 100)].sum()
                      / max(ordered.sum(), 1e-9)))
        rows.append((sigma, zero_share, top1))
        if zero_share <= target:
            chosen = sigma
            break
    return chosen, rows


def write_raster(array, path, grid, tags):
    with rasterio.open(path, "w", driver="GTiff", height=grid["height"],
                       width=grid["width"], count=1, dtype="float32",
                       crs=grid["crs"], transform=grid["transform"],
                       nodata=np.nan, compress="deflate", tiled=True) as dst:
        dst.write(array.astype("float32"), 1)
        dst.update_tags(**tags)


# ─────────────────────────────────────────────────────
# EFFORT SURFACE
# ─────────────────────────────────────────────────────
def accumulate(counts, lon, lat, weight, grid, pixel_lon, pixel_lat):
    """Add one tile's pixel counts onto the target grid, area-weighted.

    The tile pixel grid and the target grid do not nest: at zoom 3 the ratio is
    8192/2160 = 3.7926 in both axes. Assigning each tile pixel wholly to
    whichever cell its centre lands in would be exactly conservative but would
    give neighbouring cells catchments of 9, 12 or 16 tile pixels — a periodic
    +-20% modulation stamped across the map.

    So each tile pixel is treated as the box it really is and its count is
    split between the cells it overlaps, in proportion to the overlap area.
    Since the tile pixel is smaller than a cell, a box touches at most 2x2
    cells, and because lat/lon grids are separable the weights are just the
    1-D overlaps multiplied. Total records are still conserved exactly.
    """
    cell_lon = abs(grid["transform"].a)
    cell_lat = abs(grid["transform"].e)
    west, north = grid["transform"].c, grid["transform"].f

    # Fractional cell coordinates of each tile pixel's edges.
    x0 = (lon - pixel_lon / 2 - west) / cell_lon
    x1 = (lon + pixel_lon / 2 - west) / cell_lon
    y0 = (north - lat - pixel_lat / 2) / cell_lat
    y1 = (north - lat + pixel_lat / 2) / cell_lat

    col0 = np.floor(x0).astype("int64")
    row0 = np.floor(y0).astype("int64")
    # Share of the box falling in the first cell; the remainder goes to the next
    frac_col = np.clip((col0 + 1 - x0) / (x1 - x0), 0.0, 1.0)
    frac_row = np.clip((row0 + 1 - y0) / (y1 - y0), 0.0, 1.0)

    for d_row, w_row in ((0, frac_row), (1, 1.0 - frac_row)):
        for d_col, w_col in ((0, frac_col), (1, 1.0 - frac_col)):
            share = weight * w_row * w_col
            row = row0 + d_row
            col = col0 + d_col
            # Longitude wraps; latitude does not (the poles have no cells).
            col = np.mod(col, grid["width"])
            keep = (row >= 0) & (row < grid["height"]) & (share > 0)
            np.add.at(counts, (row[keep], col[keep]), share[keep])


def build_effort_surface(client, grid, zoom=ZOOM, country=None, bounds=None):
    """Fetch the density tiles covering one country, rasterise onto the grid.

    The `country` filter is passed to the map service too, so the surface
    counts records attributed to that country rather than every record inside
    its bounding box. That keeps effort from neighbouring countries out of the
    weights along the borders.
    """
    country = (country or COUNTRY).upper()
    bounds = bounds or country_bounds(country)
    xs, ys = tiles_covering(zoom, bounds)
    n_tiles = len(xs) * len(ys)
    pixel_deg = 180.0 / (2 ** zoom) / 512      # nominal; tiles report extent

    cell_deg = abs(grid["transform"].a)
    print(f"Target group:   {TARGET_GROUP_NAME} (taxonKey={TARGET_GROUP_KEY}) "
          f"in {country}")
    print(f"Tile pyramid:   EPSG:4326 zoom {zoom}, {len(xs)}x{len(ys)} "
          f"= {n_tiles} tiles over {bounds}, ~{pixel_deg:.5f} deg pixels")
    print(f"Target cell:    {cell_deg:.7f} deg, "
          f"{(cell_deg / pixel_deg) ** 2:.1f} tile pixels per cell")
    if pixel_deg > cell_deg:
        # accumulate() splits each tile pixel across at most 2x2 target cells,
        # which is only valid while the pixel is the smaller box. A coarser
        # tile than the target would need a different area-weighting, and
        # would be resampling invented detail in any case.
        raise SystemExit(
            f"zoom {zoom} gives {pixel_deg:.6f} deg tile pixels, coarser than "
            f"the {cell_deg:.6f} deg target cell. Raise --zoom.")

    counts = np.zeros((grid["height"], grid["width"]), dtype="float64")
    n_pixels = 0

    progress = tqdm(total=n_tiles, desc="Density tiles", unit="tile")
    for lon, lat, weight in density_tiles(client, zoom, progress=progress,
                                          bounds=bounds, country=country,
                                          taxonKey=TARGET_GROUP_KEY):
        accumulate(counts, lon, lat, weight, grid, pixel_deg, pixel_deg)
        n_pixels += len(lon)
    progress.close()

    print(f"Tile pixels:    {n_pixels:,}")
    print(f"Records mapped: {counts.sum():,.0f}")
    return counts, n_pixels


def verify_surface(client, counts, grid, country=None, n_boxes=6, bounds=None):
    """Cross-check the raster against the occurrence API in a few boxes.

    The boxes tile the country's bounding box, so the check covers the area the
    surface is actually used over rather than a fixed list of world regions.
    """
    country = (country or COUNTRY).upper()
    lat_min, lat_max, lon_min, lon_max = bounds or country_bounds(country)
    boxes = []
    for i in range(n_boxes):
        lat0 = lat_min + (lat_max - lat_min) * i / n_boxes
        lat1 = lat_min + (lat_max - lat_min) * (i + 1) / n_boxes
        boxes.append((round(lat0, 2), round(lat1, 2),
                      round(lon_min, 2), round(lon_max, 2),
                      f"{lat0:.1f}-{lat1:.1f}N"))

    cell = abs(grid["transform"].a)
    print("\nCross-check against /occurrence/search counts:")
    ratios = []
    for lat0, lat1, lon0, lon1, name in boxes:
        row0 = int((grid["transform"].f - lat1) / cell)
        row1 = int((grid["transform"].f - lat0) / cell)
        col0 = int((lon0 - grid["transform"].c) / cell)
        col1 = int((lon1 - grid["transform"].c) / cell)
        raster_total = counts[row0:row1, col0:col1].sum()
        api_total = client.count(taxonKey=TARGET_GROUP_KEY, country=country,
                                 decimalLatitude=f"{lat0},{lat1}",
                                 decimalLongitude=f"{lon0},{lon1}",
                                 **OCCURRENCE_FILTERS)
        ratio = raster_total / max(api_total, 1)
        ratios.append(ratio)
        print(f"  {name:14s} raster={raster_total:>13,.0f} "
              f"api={api_total:>13,} ratio={ratio:.3f}")
    return ratios


def run_surface(client, grid, zoom=ZOOM, verify=True, country=None,
                resolution="10m", smooth_km=SMOOTH_KM):
    country = (country or COUNTRY).upper()
    files = country_files(country, resolution)
    bounds = grid_bounds(grid, country)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 62)
    print(f"SAMPLING-EFFORT SURFACE — {country} @ {resolution}")
    print("=" * 62)
    print(f"Grid:           {grid['width']} x {grid['height']}, "
          f"{abs(grid['transform'].a):.7f} deg cells")

    counts, n_pixels = build_effort_surface(client, grid, zoom, country, bounds)
    ratios = (verify_surface(client, counts, grid, country, bounds=bounds)
              if verify else [])

    # Report on the cells that will actually be eligible for background --
    # inside the country AND on land. The window around a country contains a
    # lot of foreign land, and the surface is country-filtered, so foreign
    # cells are legitimately zero and would otherwise be counted as "sparse".
    land = load_land_mask(grid, country)
    inside = polygon_mask(grid)
    if land is not None and inside is not None:
        land = land & inside
    surveyed = int((counts > 0).sum())
    land_cells = int(land.sum()) if land is not None else 0

    write_raster(counts, files["effort"], grid, {
        "description": f"GBIF {TARGET_GROUP_NAME} record count per cell, "
                       f"{country}",
        "units": "occurrence records",
        "target_group": TARGET_GROUP_NAME,
        "taxon_key": str(TARGET_GROUP_KEY),
        "country": country,
        "source": "GBIF /v2/map/occurrence/density MVT tiles",
        "tile_zoom": str(zoom),
        "records_total": f"{counts.sum():.0f}",
        "purpose": "target-group background sampling weight",
    })

    # The sampling weight. See the SMOOTH_KM note in CONFIG.
    if smooth_km is None and land is not None and land_cells:
        smooth_km, ladder = choose_smoothing(counts, land, grid)
        print(f"\nKernel choice (smallest that leaves <= "
              f"{100 * SMOOTH_ZERO_SHARE_TARGET:.0f}% of in-country cells "
              f"at zero):")
        for sigma, zero_share, top1 in ladder:
            mark = "<-" if sigma == smooth_km else "  "
            print(f"  {mark} sigma {sigma:>4.1f} km   zero cells "
                  f"{100 * zero_share:>5.1f}%   top 1% of weight holds "
                  f"{100 * top1:>5.1f}%")
    elif smooth_km is None:
        smooth_km = 0.0

    smoothed = smooth_intensity(counts, grid, smooth_km)
    if smooth_km > 0:
        write_raster(smoothed, files["smoothed"], grid, {
            "description": f"smoothed GBIF {TARGET_GROUP_NAME} intensity, "
                           f"{country}",
            "units": "records per cell, kernel-smoothed",
            "smooth_km": str(smooth_km),
            "smooth_passes": str(SMOOTH_PASSES),
            "country": country,
            "purpose": "sampling weight for target-group background",
            "note": "the raw counts are the sibling file without _smoothed",
        })
    elif os.path.exists(files["smoothed"]):
        # A previous run may have written one at a wider setting. Leaving it
        # would silently override this run's decision, because run_background
        # prefers the smoothed file wherever it exists.
        os.remove(files["smoothed"])
        print(f"  removed a stale {files['smoothed']} (this run smooths none)")

    if land is not None and land_cells:
        raw_zero = 100.0 * float((counts[land] <= 0).mean())
        smooth_zero = 100.0 * float((smoothed[land] <= 0).mean())
        if smooth_km > 0:
            print(f"\nSparsity in-country: {raw_zero:.1f}% of cells hold no "
                  f"record raw -> {smooth_zero:.1f}% after a "
                  f"{smooth_km:g} km kernel")
        else:
            print(f"\nSparsity in-country: {raw_zero:.1f}% of cells hold no "
                  f"record; no kernel applied, the floor carries them")
        positive = smoothed[land][smoothed[land] > 0]
        if len(positive):
            median = float(np.median(positive))
            floor = effort_floor(smoothed[land], resolution)
            print(f"Weight scale:     median surveyed cell {median:.2f} "
                  f"records, floor {floor:.4g}, so an unsurveyed cell is "
                  f"{median / floor:.0f}x less likely than a median one")

    print(f"\nCells with any plant record: {surveyed:,} of "
          f"{grid['width'] * grid['height']:,}")
    if land_cells:
        on_land = int(((counts > 0) & land).sum())
        print(f"In-country cells surveyed:   {on_land:,} of {land_cells:,} "
              f"({100 * on_land / land_cells:.1f}%)")
        share = np.sort(counts[land].ravel())[::-1]
        total = max(share.sum(), 1)
        print(f"Effort concentration:        top 1% of in-country cells hold "
              f"{100 * share[:max(1, len(share) // 100)].sum() / total:.1f}% "
              f"of all plant records")
    print(f"Raster: {files['effort']} "
          f"({os.path.getsize(files['effort']) / 1e6:.1f} MB)")

    with open(files["manifest"], "w") as handle:
        json.dump({
            "country": country,
            "country_bounds": country_bounds(country),
            "resolution": resolution,
            "target_grid": {"width": grid["width"], "height": grid["height"],
                            "crs": str(grid["crs"]),
                            "transform": list(grid["transform"])[:6],
                            "template": (TEMPLATE if resolution == "10m" else
                                         f"country_data/{grid.get('iso3')}/"
                                         "country_grid.json")},
            "effort_raster": files["effort"],
            "smoothed_raster": files["smoothed"] if smooth_km > 0 else None,
            "smooth_km": smooth_km,
            "smooth_passes": SMOOTH_PASSES,
            "country_mask": ("Natural Earth polygon" if "geometry" in grid
                             else "bounding box AND cells holding records"),
            "target_group": TARGET_GROUP_NAME,
            "taxon_key": TARGET_GROUP_KEY,
            "tile_zoom": zoom,
            "tile_pixels_read": n_pixels,
            "records_total": float(counts.sum()),
            "cells_surveyed": surveyed,
            "verification_ratios": [round(r, 4) for r in ratios],
            "occurrence_filters": OCCURRENCE_FILTERS,
        }, handle, indent=2)
    return counts


# ─────────────────────────────────────────────────────
# BACKGROUND POINTS
# ─────────────────────────────────────────────────────
def load_land_mask(grid, country=None):
    """Boolean land mask on `grid`, or None if nothing suitable exists.

    On the 10 arcmin grid this is the global land-fraction layer, unchanged.
    On a country grid that layer is the wrong shape and 400x too coarse, so
    land is taken from the country's own 30 arcsec climate block: a cell has a
    bioclim value or it does not, and that is the mask the model is applied
    under anyway, which makes it the honest definition of "somewhere a
    recommendation could be made".
    """
    if "iso3" in grid:
        candidate = os.path.join(COUNTRY_DATA_ROOT, grid["iso3"],
                                 "climate_30s", "wc2.1_30s_bio_1.tif")
        if not os.path.exists(candidate):
            return None
        with rasterio.open(candidate) as src:
            band = src.read(1)
            nodata = src.nodata
        land = np.isfinite(band) & (band > -1e30)
        if nodata is not None and np.isfinite(nodata):
            land &= band != nodata
        return land

    if not os.path.exists(LAND_RASTER):
        return None
    with rasterio.open(LAND_RASTER) as src:
        land = src.read(1)
    if land.shape != (grid["height"], grid["width"]):
        return None
    return np.nan_to_num(land, nan=0.0) >= MIN_LAND_PERCENT


def effort_floor(values, resolution="10m"):
    """The additive floor for one surface. See EFFORT_FLOOR_MEDIAN_RATIO."""
    if resolution == "10m":
        return EFFORT_FLOOR
    positive = values[values > 0]
    if not positive.size:
        return EFFORT_FLOOR
    return float(np.median(positive)) / EFFORT_FLOOR_MEDIAN_RATIO


def weighted_sample_without_replacement(weights, n, rng):
    """Draw n distinct indices with probability proportional to `weights`.

    The exponential race (Efraimidis-Spirakis): give every candidate a key
    -log(U)/w and take the n smallest. Exact, one pass, no scipy.
    """
    keys = rng.exponential(size=len(weights)) / weights
    return np.argpartition(keys, n - 1)[:n]


def country_mask(grid, effort, country=None):
    """Cells that count as this country for background sampling.

    The bounding box alone would reach into Spain and Germany, and there is no
    border polygon here. The country-filtered effort surface supplies the
    border instead: a cell holds records attributed to this country or it does
    not. Combined with the land mask that is a good approximation.

    The cost is that a French cell where nobody has ever recorded a vascular
    plant is excluded rather than given the EFFORT_FLOOR probability. In
    France that is close to free — 92 million plant records leave almost no
    unrecorded cells — but in a sparsely recorded country it would matter, and
    there a real border polygon would be needed.
    """
    polygon = polygon_mask(grid)
    if polygon is not None:
        # A real border. The "holds a record attributed to this country" rule
        # below is a stand-in for one, and at 1 km it would be fatal: most
        # cells hold no record at all, so it would confine background to the
        # surveyed cells whose bias is the entire thing being corrected.
        return polygon

    lat_min, lat_max, lon_min, lon_max = country_bounds(country)
    cell = abs(grid["transform"].a)

    rows = np.arange(grid["height"])
    cols = np.arange(grid["width"])
    lat = grid["transform"].f - (rows + 0.5) * cell
    lon = grid["transform"].c + (cols + 0.5) * cell
    box = (((lat >= lat_min) & (lat <= lat_max))[:, None] &
           ((lon >= lon_min) & (lon <= lon_max))[None, :])

    return box & (effort > 0)


def run_background(grid, n_points=N_BACKGROUND, presence_file=None,
                   country=None, canonical=True, resolution="10m",
                   power=EFFORT_POWER):
    country = (country or COUNTRY).upper()
    files = country_files(country, resolution)
    presence_file = presence_file or files["presence"]

    print("=" * 62)
    print(f"TARGET-GROUP BACKGROUND POINTS — {country} @ {resolution}")
    print("=" * 62)

    if not os.path.exists(files["effort"]):
        raise SystemExit(f"No effort surface at {files['effort']}. "
                         f"Run: python3 sampling_effort.py --surface "
                         f"--resolution {resolution}")

    with rasterio.open(files["effort"]) as src:
        raw_effort = np.nan_to_num(src.read(1).astype("float64"), nan=0.0)

    # Weight by the smoothed field where one was written; fall back to the raw
    # counts, which is what the 18.5 km grid has always used and is correct
    # there because its cells hold tens of thousands of records each.
    if os.path.exists(files["smoothed"]):
        with rasterio.open(files["smoothed"]) as src:
            effort = np.nan_to_num(src.read(1).astype("float64"), nan=0.0)
        print(f"Weighting by:   {files['smoothed']} (kernel-smoothed)")
    else:
        effort = raw_effort
        print(f"Weighting by:   {files['effort']} (raw counts)")

    land = load_land_mask(grid, country)
    if land is None:
        # Without a land mask, "has climate data" is the next best definition
        # of land, and it is the mask the model will be applied under anyway.
        with rasterio.open(TEMPLATE) as src:
            band = src.read(1)
            nodata = src.nodata
        land = np.isfinite(band) & (band != nodata if nodata is not None else True)
        print(f"No {LAND_RASTER}; using the climate template's valid cells "
              f"as the land mask")

    inside = country_mask(grid, effort, country)
    land = land & inside
    print(f"Country cells:  {int(land.sum()):,} land cells inside {country}")
    eligible = land.copy()

    presences = None
    if EXCLUDE_PRESENCE_CELLS and os.path.exists(presence_file):
        presences = pd.read_csv(presence_file)
        cell = abs(grid["transform"].a)
        rows = np.floor((grid["transform"].f - presences["latitude"]) / cell) \
                 .astype("int64")
        cols = np.floor((presences["longitude"] - grid["transform"].c) / cell) \
                 .astype("int64")
        inside = ((rows >= 0) & (rows < grid["height"]) &
                  (cols >= 0) & (cols < grid["width"]))
        eligible[rows[inside], cols[inside]] = False
        print(f"Presence file:  {presence_file} ({len(presences):,} records)")
        print(f"Excluded cells: {int(land.sum() - eligible.sum()):,} "
              f"already hold a presence")
    elif EXCLUDE_PRESENCE_CELLS:
        print(f"No presence file at {presence_file}; background may land on "
              f"presence cells")

    rows, cols = np.nonzero(eligible)
    values = effort[rows, cols]
    floor = effort_floor(values, resolution)
    weights = (values + floor) ** power
    rng = np.random.default_rng(RANDOM_SEED)
    positive = values[values > 0]
    if positive.size:
        print(f"Effort floor:   {floor:.4g} records "
              f"(median surveyed cell {np.median(positive):.4g}, so an "
              f"unsurveyed cell is {np.median(positive) / floor:.0f}x less "
              f"likely)")

    # Sampling without replacement only weights by effort while the pool is
    # much larger than the sample. One country at 18.5 km has ~2,600 cells, so
    # asking for more points than that and drawing without replacement would
    # return every cell exactly once — a uniform sample, which is precisely
    # the bias this file exists to remove. With replacement, effort shows up
    # as repeated cells and the row density stays proportional to effort.
    with_replacement = n_points > len(rows) // 2
    if with_replacement:
        pick = rng.choice(len(rows), size=n_points, replace=True,
                          p=weights / weights.sum())
        print(f"Drawing {n_points:,} points WITH replacement from "
              f"{len(rows):,} cells, so effort weighting survives")
    else:
        pick = weighted_sample_without_replacement(weights, n_points, rng)

    cell = abs(grid["transform"].a)
    lat = grid["transform"].f - (rows[pick] + 0.5) * cell
    lon = grid["transform"].c + (cols[pick] + 0.5) * cell

    frame = pd.DataFrame({
        "latitude": lat, "longitude": lon,
        "grid_row": rows[pick], "grid_col": cols[pick],
        "plant_records": raw_effort[rows[pick], cols[pick]],
        "effort_intensity": effort[rows[pick], cols[pick]],
        "weight": weights[pick],
    }).reset_index(drop=True)
    # Not sorted by cell: sdm_features.load_background_file subsamples with a
    # stride (frame[::step]), which on cell-sorted rows would take a
    # geographically systematic slice and throw the effort weighting away
    # again. Shuffled, any stride is still a fair sample.
    frame = frame.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frame.to_csv(files["background"], index=False)
    if canonical:
        frame.to_csv(CANONICAL_BACKGROUND, index=False)

    distinct = frame.groupby(["grid_row", "grid_col"], sort=False).ngroups
    print(f"\nEligible cells: {len(rows):,}")
    print(f"Background:     {len(frame):,} points over {distinct:,} distinct "
          f"cells, {floor:.4g} record floor, effort^{power}")
    print(f"Surveyed share: "
          f"{100 * (frame['plant_records'] > 0).mean():.1f}% of background "
          f"points fall in a cell that has plant records "
          f"({100 * (raw_effort[rows, cols] > 0).mean():.1f}% of eligible "
          f"cells do)")
    print(f"Written to:     {files['background']}")
    if canonical:
        print(f"                {CANONICAL_BACKGROUND} "
              f"(what sdm_features.py reads)")
    else:
        print(f"                {CANONICAL_BACKGROUND} left alone "
              "(--no-canonical)")
    return frame


# ─────────────────────────────────────────────────────
# COMPARISON: does it actually remove the survey-bias signal?
# ─────────────────────────────────────────────────────
def auc(positive, negative):
    """Mann-Whitney U / AUC. numpy only; sklearn is not installed."""
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = np.empty(len(combined), dtype="float64")
    order = np.argsort(combined, kind="stable")
    sorted_values = combined[order]
    # average ranks within ties, or the AUC of a discrete variable is wrong
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rank_sum = ranks[:len(positive)].sum()
    n_pos, n_neg = len(positive), len(negative)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def sample_raster(path, lat, lon, grid=None):
    """Read `path` at the given coordinates, using THAT RASTER'S transform.

    It used to index with the caller's grid, which silently assumed every
    layer being compared shared it. That was true while everything was on the
    10 arcmin grid and is false the moment a 1 km background is scored against
    the 10 arcmin built-up layer -- it would have read a point 20 cells away.
    Using each file's own transform is what lets old and new background be
    measured against one common yardstick, which is the only way the
    comparison means anything.
    """
    with rasterio.open(path) as src:
        band = src.read(1)
        transform, nodata = src.transform, src.nodata
    res_x, res_y = transform.a, -transform.e
    rows = np.floor((transform.f - np.asarray(lat, dtype="float64")) / res_y)
    cols = np.floor((np.asarray(lon, dtype="float64") - transform.c) / res_x)
    rows = rows.astype("int64")
    cols = cols.astype("int64")
    out = np.full(len(rows), np.nan)
    inside = ((rows >= 0) & (rows < band.shape[0]) &
              (cols >= 0) & (cols < band.shape[1]))
    out[inside] = band[rows[inside], cols[inside]]
    out[out < -1e30] = np.nan
    if nodata is not None and np.isfinite(nodata):
        out[out == nodata] = np.nan
    return out


# ─────────────────────────────────────────────────────
def builtup_1km(country, iso3):
    """300 m built-up fraction block-averaged onto the country's 1 km grid.

    THE yardstick, and the reason it exists: at 300 m a presence is a person's
    recorded position while a background point is a cell centre, so the two are
    measured at different spatial supports and part of any gap is that
    mismatch rather than bias. The model never sees 300 m -- every predictor it
    reads is a 1 km cell value -- so the honest question is whether presence
    CELLS and background CELLS differ, and that is what this measures.

    Cached beside the effort surfaces. It is a diagnostic and is deliberately
    written outside country_data/, where a built-up layer could be mistaken for
    a predictor; vegetation and built-up layers are circular for this problem.
    """
    source = os.path.join(COUNTRY_DATA_ROOT, iso3, "satellite_download",
                          "landcover_pft_2020", "pft_built.tif")
    if not os.path.exists(source):
        return None
    out_path = os.path.join(OUTPUT_DIR,
                            f"builtup_diagnostic_{country.upper()}_30s.tif")
    if os.path.exists(out_path):
        return out_path

    from rasterio.warp import Resampling, reproject

    grid = target_grid(country, "30s")
    destination = np.zeros((grid["height"], grid["width"]), dtype="float32")
    with rasterio.open(source) as src:
        reproject(rasterio.band(src, 1), destination,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  resampling=Resampling.average)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_raster(destination, out_path, grid, {
        "description": "ESA CCI PFT built-up fraction, 300 m averaged to 1 km",
        "purpose": "DIAGNOSTIC ONLY -- measuring survey bias in background "
                   "points. Never a predictor: built-up and vegetation layers "
                   "are circular for this problem.",
        "country": country.upper(),
    })
    return out_path


def builtup_layers(country=None, grid=None):
    """Yardsticks for the built-up comparison.

    The 10 arcmin global layer is the one the earlier experiment used to get
    "presence 9.93% built-up against background 2.80%", so it is kept -- a new
    number is only comparable if it is measured the same way. The 1 km version
    is the one to read, for the reason in builtup_1km().
    """
    layers = {}
    iso3 = (grid or {}).get("iso3") or (country_iso3(country) if country else None)
    if country and iso3:
        fine = builtup_1km(country, iso3)
        if fine:
            layers["built-up 1km (model support)"] = fine
    if os.path.exists(BUILTUP_RASTER):
        layers["built-up 18.5km (legacy yardstick)"] = BUILTUP_RASTER
    return layers


def uniform_reference(country, n_points=50_000):
    """Coordinates of a uniform draw over the country, as the do-nothing baseline.

    Without it the old and new numbers float free: "gap 3.16 versus 2.47" says
    nothing until you know that doing nothing gives 5.87.
    """
    try:
        grid = target_grid(country, "30s")
    except SystemExit:
        return None
    land = load_land_mask(grid, country)
    polygon = polygon_mask(grid)
    if land is None or polygon is None:
        return None
    rows, cols = np.nonzero(land & polygon)
    rng = np.random.default_rng(RANDOM_SEED)
    pick = rng.choice(len(rows), size=min(n_points, len(rows)), replace=False)
    cell = abs(grid["transform"].a)
    return (grid["transform"].f - (rows[pick] + 0.5) * cell,
            grid["transform"].c + (cols[pick] + 0.5) * cell)


def run_score(presence_file, background_files, country=None, grid=None):
    """Score presences and one or more background files on built-up fraction.

    This is the cleanest evidence that a background set absorbs survey bias:
    if it does, its built-up fraction looks like the presences' and the AUC
    separating them sits near 0.5. If it does not, the model has a "near a
    town" signal left to learn and will learn it.
    """
    print("=" * 70)
    print(f"BUILT-UP SEPARATION — presences vs background")
    print("=" * 70)

    presences = pd.read_csv(presence_file)
    print(f"Presences:  {presence_file} ({len(presences):,} records)")
    layers = builtup_layers(country, grid)
    reference = uniform_reference(country)
    if not layers:
        raise SystemExit("No built-up raster available to score against")

    results = {}
    for label, path in layers.items():
        p = sample_raster(path, presences["latitude"], presences["longitude"])
        print(f"\n{label}   ({path})")
        print(f"  {'background set':<44} {'presence':>9} {'bg':>9} "
              f"{'gap':>8} {'AUC':>7}")
        print(f"  {'(presences)':<44} {np.nanmean(p):>9.2f} "
              f"{'-':>9} {'-':>8} {'-':>7}")
        if reference is not None:
            u = sample_raster(path, reference[0], reference[1])
            print(f"  {'UNIFORM over the country (no correction)':<44} "
                  f"{np.nanmean(p):>9.2f} {np.nanmean(u):>9.2f} "
                  f"{np.nanmean(p) - np.nanmean(u):>8.2f} {auc(p, u):>7.3f}")
        for background_file in background_files:
            if not os.path.exists(background_file):
                print(f"  {background_file:<44} MISSING")
                continue
            background = pd.read_csv(background_file)
            b = sample_raster(path, background["latitude"],
                              background["longitude"])
            gap = float(np.nanmean(p) - np.nanmean(b))
            score = auc(p, b)
            results.setdefault(label, {})[background_file] = {
                "presence_mean": float(np.nanmean(p)),
                "background_mean": float(np.nanmean(b)),
                "gap": gap, "auc": float(score),
                "separation": abs(float(score) - 0.5),
                "n_background": int(len(background)),
            }
            name = os.path.basename(background_file)
            print(f"  {name:<44} {np.nanmean(p):>9.2f} {np.nanmean(b):>9.2f} "
                  f"{gap:>8.2f} {score:>7.3f}")
    print("\nA smaller gap and an AUC nearer 0.500 mean less of the model's "
          "signal is\nsurvey effort. This is an honesty fix: removing real "
          "bias should LOWER\nheadline AUC, by roughly 0.01-0.02, not raise "
          "it.")
    return results


def run_compare(grid, presence_file=None, n_points=N_BACKGROUND,
                country=None, resolution="10m"):
    """Uniform vs effort-weighted background, measured on built-up fraction.

    The claim being tested is the one in the module docstring: uniform
    background makes built-up fraction look like the best predictor of tree
    habitat. If target-group background works, the presence-vs-background AUC
    on built-up fraction should fall towards 0.5 — no separation — while the
    genuine environmental variables keep theirs.
    """
    country = (country or COUNTRY).upper()
    files = country_files(country, resolution)
    presence_file = presence_file or files["presence"]

    print("=" * 62)
    print(f"UNIFORM vs TARGET-GROUP BACKGROUND — {country} @ {resolution}")
    print("=" * 62)

    if not os.path.exists(presence_file):
        raise SystemExit(f"No presences at {presence_file}")
    if not os.path.exists(BUILTUP_RASTER):
        raise SystemExit(f"Needs {BUILTUP_RASTER} (run satellite_rasters.py)")

    presences = pd.read_csv(presence_file)
    with rasterio.open(files["effort"]) as src:
        effort = np.nan_to_num(src.read(1).astype("float64"), nan=0.0)

    # Both background sets are drawn from the same country cells, so the only
    # difference measured is uniform vs effort-weighted, not extent.
    land = load_land_mask(grid, country) & country_mask(grid, effort, country)
    rows, cols = np.nonzero(land)
    rng = np.random.default_rng(RANDOM_SEED)
    cell = abs(grid["transform"].a)

    n_points = min(n_points, len(rows) // 2)
    uniform_pick = rng.choice(len(rows), size=n_points, replace=False)
    weights = ((effort[rows, cols]
                + effort_floor(effort[rows, cols], resolution)) ** EFFORT_POWER)
    tgb_pick = weighted_sample_without_replacement(weights, n_points, rng)

    def coords(pick):
        return (grid["transform"].f - (rows[pick] + 0.5) * cell,
                grid["transform"].c + (cols[pick] + 0.5) * cell)

    uniform_lat, uniform_lon = coords(uniform_pick)
    tgb_lat, tgb_lon = coords(tgb_pick)

    layers = dict(builtup_layers(country, grid))
    layers["annual mean temp"] = TEMPLATE
    print(f"Presences: {len(presences):,}   background: {n_points:,} each\n")
    print(f"{'variable':22s} {'presence':>10s} {'uniform bg':>11s} "
          f"{'TGB bg':>10s} {'AUC unif':>9s} {'AUC TGB':>8s}")
    results = {}
    for label, path in layers.items():
        if not os.path.exists(path):
            continue
        p = sample_raster(path, presences["latitude"], presences["longitude"])
        u = sample_raster(path, uniform_lat, uniform_lon)
        t = sample_raster(path, tgb_lat, tgb_lon)
        auc_u, auc_t = auc(p, u), auc(p, t)
        results[label] = {"presence_mean": float(np.nanmean(p)),
                          "uniform_mean": float(np.nanmean(u)),
                          "tgb_mean": float(np.nanmean(t)),
                          "auc_uniform": float(auc_u), "auc_tgb": float(auc_t)}
        print(f"{label:22s} {np.nanmean(p):>10.3f} {np.nanmean(u):>11.3f} "
              f"{np.nanmean(t):>10.3f} {auc_u:>9.3f} {auc_t:>8.3f}")

    built = (results.get("built-up 1km (model support)")
             or results.get("built-up 18.5km (legacy yardstick)"))
    if built:
        print(f"\nBuilt-up separation |AUC-0.5|: uniform "
              f"{abs(built['auc_uniform'] - 0.5):.3f} -> target-group "
              f"{abs(built['auc_tgb'] - 0.5):.3f}")
        print("Closer to zero means less of the model's signal is survey "
              "effort. The previous analysis measured 0.246 on uniform "
              "background.")
    return results


# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build the GBIF sampling-effort surface and draw "
                    "target-group background points.")
    parser.add_argument("--surface", action="store_true",
                        help="download density tiles and write the effort raster")
    parser.add_argument("--background", action="store_true",
                        help="draw background points from the effort raster")
    parser.add_argument("--compare", action="store_true",
                        help="measure uniform vs target-group background bias")
    parser.add_argument("--score", nargs="+", metavar="BACKGROUND_CSV",
                        default=None,
                        help="score presences against existing background "
                             "files on built-up fraction; the old-vs-new "
                             "evidence that the bias correction works")
    parser.add_argument("--resolution", choices=RESOLUTIONS,
                        default=DEFAULT_RESOLUTION,
                        help="grid to build on: 30s (~1 km, the country's own "
                             "grid, matches the predictors) or 10m (~18.5 km, "
                             f"the global template). Default {DEFAULT_RESOLUTION}")
    parser.add_argument("--template", default=None,
                        help="explicit grid template raster, overriding "
                             "--resolution")
    parser.add_argument("--effort-power", type=float, default=EFFORT_POWER,
                        help="exponent on the effort weight. 1.0 is strict "
                             "target-group sampling; below 1.0 flattens "
                             "towards uniform, which is the knob for a country "
                             "whose recording is so city-concentrated that "
                             "strict weighting overshoots and makes background "
                             "MORE urban than the presences")
    parser.add_argument("--smooth-km", type=float, default=None,
                        help="kernel sigma in km for the sampling weight; "
                             "default is adaptive at 30s (smallest on "
                             f"{SMOOTH_LADDER_KM} leaving <= "
                             f"{100 * SMOOTH_ZERO_SHARE_TARGET:.0f}% zero "
                             "cells) and 0 at 10m, where cells already hold "
                             "tens of thousands of records")
    parser.add_argument("--zoom", type=int, default=None,
                        help="tile zoom, cost scales as 4^zoom "
                             f"(default {ZOOM_BY_RESOLUTION})")
    parser.add_argument("--n-background", type=int, default=None,
                        help=f"background points (default "
                             f"{N_BACKGROUND_BY_RESOLUTION})")
    parser.add_argument("--presences", default=None,
                        help="presence CSV used to exclude cells and compare")
    parser.add_argument("--country", default=COUNTRY,
                        help=f"ISO country code (default {COUNTRY})")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the API cross-check of the raster")
    parser.add_argument("--no-canonical", action="store_true",
                        help="write only the country-scoped background file "
                             "and leave sampling_effort/background_points.csv "
                             "alone. The canonical copy is what sdm_features.py "
                             "reads, so a second country would otherwise "
                             "silently repoint a training run at itself")
    args = parser.parse_args()

    country = args.country.upper()
    resolution = "10m" if args.template else args.resolution
    zoom = args.zoom or ZOOM_BY_RESOLUTION[resolution]
    n_background = args.n_background or N_BACKGROUND_BY_RESOLUTION[resolution]
    # The coarse grid needs no smoothing and never had any: one 18.5 km cell
    # holds tens of thousands of records, so the raw count already IS the
    # effort field rather than a noisy sample of it.
    smooth_km = (args.smooth_km if args.smooth_km is not None
                 else (SMOOTH_KM if resolution == "30s" else 0.0))
    if smooth_km is not None and smooth_km < 0:
        smooth_km = None            # -1 forces the adaptive ladder back on

    if not (args.surface or args.background or args.compare or args.score):
        parser.print_help()
        return

    if args.score:
        files = country_files(country, resolution)
        run_score(args.presences or files["presence"], args.score, country)
        return

    grid = target_grid(country, resolution, args.template)

    if args.surface:
        run_surface(GbifClient(), grid, zoom, not args.no_verify, country,
                    resolution, smooth_km)
    if args.background:
        run_background(grid, n_background, args.presences, country,
                       canonical=not args.no_canonical, resolution=resolution,
                       power=args.effort_power)
    if args.compare:
        run_compare(grid, args.presences, n_background, country, resolution)


if __name__ == "__main__":
    main()
