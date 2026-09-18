"""
Build the country-scoped TERRAIN block at 30 arcsec (~1 km) from GEDTM30 v1.2
at its native 1 arcsec: elevation, ruggedness, SLOPE and ASPECT.

WHAT GAP THIS CLOSES
    topography/ ships elevation and nothing else. Slope and aspect were derived
    and validated in validate_topography_alignment.py but never made it into
    production, so the model currently cannot tell a cliff from a plain at the
    same height. This script produces them.

    It does not touch topography/. Everything lands under
    country_data/<ISO>/topography_30s/.

DERIVE AT NATIVE RESOLUTION, AGGREGATE AFTERWARDS
    This is the whole point, and it is the one thing that is easy to get wrong.
    DATA_OVERVIEW.md states it as assumption 6: averaging fine data up is fine
    for elevation and FALSE for slope. Averaging a DEM over a cell erases every
    ridge and valley smaller than that cell, so differentiating the smoothed
    surface reports a nearly flat world -- validate_topography_alignment.py
    measured the coarsen-first route under-reporting mean slope by roughly an
    order of magnitude. So slope and aspect are computed per native 1 arcsec
    pixel first, and only the RESULT is aggregated.

    Aspect needs more than that, because it is circular: 350 deg and 10 deg are
    20 deg apart and both face north, but their arithmetic mean is due south.
    It is therefore decomposed into northness = cos(aspect) and
    eastness = sin(aspect), each weighted by sin(slope) so that flat ground --
    where aspect is undefined and numerically random -- contributes nothing
    rather than noise.

    slope_aspect() and block_reduce() are imported from
    validate_topography_alignment.py rather than reimplemented. That
    implementation includes the latitude correction most people skip (a degree
    of longitude is 111,320 * cos(lat) m, so a constant dx makes high-latitude
    terrain look artificially steep) and it was cross-checked against
    GEDTM30's own published slope layer to r = 0.999.

WHY A WARP AND NOT A BLOCK REDUCTION
    GEDTM30's grid is anchored at -180.00125, which is 4.5 native pixels off
    the whole-degree line, so target cell edges fall exactly on native pixel
    CENTRES rather than on native pixel edges. A naive 30 x 30 block reduction
    would therefore be half a native pixel out of register on every edge.

    So each native field is reduced onto the target grid with GDAL's warper,
    which weights partial pixels correctly. That also buys the reductions that
    a block average cannot express: min and max for the elevation range, and
    the third quartile for a steep-ground indicator.

    Two of the outputs need a weighted ratio rather than a mean, and the trick
    is that a ratio of two averages over the same cell equals the ratio of
    their sums, so numerator and weight are reduced separately and divided:
        northness = avg(sin(slope) * cos(aspect)) / avg(sin(slope))
    Standard deviation is done the same way, from avg(x) and avg(x^2).

COST, HONESTLY
    Nothing is downloaded in bulk: the source is a 432 GB Cloud-Optimised
    GeoTIFF and only the country's windows travel, tile by tile. But the DEM
    has to be read at 1 arcsec, which is 900 native pixels per target cell, so
    this is the slow block -- expect hours for a large country, and a few GB of
    HTTP range traffic. Tiles that fall entirely outside the country polygon
    are skipped, and progress is checkpointed so an interrupted run resumes
    instead of starting over.

OUTPUTS  (country_data/<ISO>/topography_30s/)
    elev_mean.tif elev_min.tif elev_max.tif elev_range.tif elev_std.tif
    slope_mean.tif slope_q3.tif slope_max.tif
    northness.tif eastness.tif aspect_strength.tif
    MANIFEST.json

USAGE
    python3 country_topography_rasters.py IND
    python3 country_topography_rasters.py IND --tile-deg 1.0
    python3 country_topography_rasters.py IND --all-tiles   # do not skip sea
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from tqdm import tqdm

import validate_topography_alignment as vta
from country_clip import RES_30S, country_grid, country_mask

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUT_ROOT = "country_data"
BLOCK_NAME = "topography_30s"
TARGET_RES = RES_30S

# ── GEDTM30 v1.2: bare-earth DTM, 1 arcsec, 85N-65S, CC-BY-4.0 ──
# Bare-earth matters here more than accuracy does. Copernicus DEM and the SRTM
# family are surface models that sit on the treetops, and their error tracks
# canopy density -- which would leak the thing being predicted into a
# predictor, and into slope and aspect as derivatives of it.
GEDTM30_BASE = "https://s3.opengeohub.org/global/dtm"
GEDTM30_DTM_URL = (f"{GEDTM30_BASE}/v1.2/"
                   "gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_"
                   "v1.2.tif")
GEDTM30_NATIVE_RES = 1.0 / 3600.0        # 1 arcsec
GEDTM30_NODATA_ABOVE = 1e30
GEDTM30_NORTH_LIMIT = 85.0
GEDTM30_SOUTH_LIMIT = -65.0

# Tile size in degrees. 0.5 deg is 1800 x 1800 native pixels, ~26 MB as
# float64, and the derivation holds several such arrays at once. 1.0 deg
# quadruples that; do not go much higher.
TILE_DEG = 0.5
HALO_PX = 2          # native pixels of overlap, so the Horn 3x3 kernel has
                     # real neighbours at a tile edge instead of NaN

CHECKPOINT_EVERY = 25        # tiles
NODATA = -3.4e38             # project convention: valid cells are > -1e30

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "200000000",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

# name -> (description, unit)
LAYERS = {
    "elev_mean": ("mean bare-earth elevation", "m"),
    "elev_min": ("lowest native pixel in the cell", "m"),
    "elev_max": ("highest native pixel in the cell", "m"),
    "elev_range": ("elev_max - elev_min, relief within the cell", "m"),
    "elev_std": ("standard deviation of elevation, ruggedness", "m"),
    "slope_mean": ("mean slope, from native 1 arcsec", "degrees"),
    "slope_q3": ("third quartile of slope, steep-ground indicator", "degrees"),
    "slope_max": ("steepest native pixel in the cell", "degrees"),
    "northness": ("cos(aspect) weighted by sin(slope); +1 = pole-facing, "
                  "shaded, cooler and moister", "-1..1"),
    "eastness": ("sin(aspect) weighted by sin(slope); +1 = east-facing",
                 "-1..1"),
    "aspect_strength": ("|(northness, eastness)|; 0 = no dominant facing",
                        "0..1"),
}

SEP = "=" * 70


# ─────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────
def load_country_grid(iso, margin_deg, area_frac, max_gap_deg):
    """The authoritative country grid, from country_grid.json where it exists.

    country_climate_rasters.py writes that file, and every block reading it is
    what guarantees the blocks share one grid rather than merely agreeing about
    what grid they should share.
    """
    path = os.path.join(OUT_ROOT, iso, "country_grid.json")
    grid = country_grid(iso, res_deg=TARGET_RES, margin_deg=margin_deg,
                        area_frac=area_frac, max_gap_deg=max_gap_deg)

    if not os.path.exists(path):
        print(f"  note: {path} not found, so the grid is derived from the "
              "country polygon.\n        Run country_climate_rasters.py first "
              "to pin the grid to WorldClim's extent.")
        return grid

    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    if abs(saved["res_deg"] - TARGET_RES) > 1e-12:
        raise RuntimeError(f"{path} is at {saved['res_deg']} deg, this block "
                           f"is {TARGET_RES}")
    grid.update(transform=Affine(*saved["transform"]), width=saved["width"],
                height=saved["height"], bounds=tuple(saved["bounds"]),
                grid_source=os.path.basename(path))
    print(f"  grid adopted from {path}")
    return grid


def tiles_for(grid, tile_deg, inside, all_tiles):
    """Target-grid sub-windows to process, as (row0, col0, rows, cols).

    Tiles are whole numbers of target cells, so every tile's destination
    transform is exactly the country transform translated by an integer -- the
    reduction can never smear across a cell boundary.
    """
    step = int(round(tile_deg / TARGET_RES))
    out = []
    for row0 in range(0, grid["height"], step):
        for col0 in range(0, grid["width"], step):
            rows = min(step, grid["height"] - row0)
            cols = min(step, grid["width"] - col0)
            if not all_tiles and not inside[row0:row0 + rows,
                                            col0:col0 + cols].any():
                continue
            out.append((row0, col0, rows, cols))
    return out


# ─────────────────────────────────────────────────────
# REDUCTION
# ─────────────────────────────────────────────────────
def reduce_native(native, native_transform, crs, dest_transform, shape,
                  resampling):
    """Reduce one native-resolution field onto a target sub-grid.

    GDAL's warper is used rather than a block average because GEDTM30's grid
    sits 4.5 native pixels off the whole-degree line, so target cell edges fall
    on native pixel centres and partial pixels have to be weighted properly.
    """
    out = np.full(shape, np.nan, dtype="float32")
    reproject(source=np.ascontiguousarray(native, dtype="float32"),
              destination=out,
              src_transform=native_transform, src_crs=crs, src_nodata=np.nan,
              dst_transform=dest_transform, dst_crs=crs, dst_nodata=np.nan,
              resampling=resampling)
    return out


def derive_tile(dem, native_transform, crs, dest_transform, shape):
    """Every terrain aggregate for one tile, derived at native resolution."""
    res_lon, res_lat = native_transform.a, -native_transform.e
    lat_centres = native_transform.f - (np.arange(dem.shape[0]) + 0.5) * res_lat

    slope, aspect = vta.slope_aspect(dem, lat_centres, res_lon, res_lat)

    # Aspect is undefined on flat ground, where sin(slope) is 0 and the
    # direction carries no weight. Represent that as a zero contribution with
    # zero weight, which is different from absent data.
    finite = np.isfinite(slope)
    weight = np.where(finite, np.sin(np.deg2rad(slope)), np.nan)
    angle = np.deg2rad(aspect)
    has_aspect = np.isfinite(aspect)
    northing = np.where(has_aspect, np.cos(angle) * weight,
                        np.where(finite, 0.0, np.nan))
    easting = np.where(has_aspect, np.sin(angle) * weight,
                       np.where(finite, 0.0, np.nan))

    def reduce_to(field, resampling=Resampling.average):
        return reduce_native(field, native_transform, crs, dest_transform,
                             shape, resampling)

    elev_mean = reduce_to(dem)
    elev_sq = reduce_to(dem * dem)
    elev_min = reduce_to(dem, Resampling.min)
    elev_max = reduce_to(dem, Resampling.max)

    with np.errstate(invalid="ignore"):
        variance = np.maximum(elev_sq - elev_mean * elev_mean, 0.0)

    weight_mean = reduce_to(weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        northness = np.where(weight_mean > 1e-9,
                             reduce_to(northing) / weight_mean, 0.0)
        eastness = np.where(weight_mean > 1e-9,
                            reduce_to(easting) / weight_mean, 0.0)
    # Keep sea absent rather than calling it flat.
    northness = np.where(np.isfinite(weight_mean), northness, np.nan)
    eastness = np.where(np.isfinite(weight_mean), eastness, np.nan)

    return {
        "elev_mean": elev_mean,
        "elev_min": elev_min,
        "elev_max": elev_max,
        "elev_range": elev_max - elev_min,
        "elev_std": np.sqrt(variance),
        "slope_mean": reduce_to(slope),
        "slope_q3": reduce_to(slope, Resampling.q3),
        "slope_max": reduce_to(slope, Resampling.max),
        "northness": northness,
        "eastness": eastness,
        "aspect_strength": np.hypot(northness, eastness),
    }


# ─────────────────────────────────────────────────────
# TILE LOOP
# ─────────────────────────────────────────────────────
def read_native_window(src, west, south, east, north):
    """A native 1 arcsec window with a halo, plus its exact transform."""
    window = from_bounds(west, south, east, north,
                         src.transform).round_offsets().round_lengths()
    window = rasterio.windows.Window(
        window.col_off - HALO_PX, window.row_off - HALO_PX,
        window.width + 2 * HALO_PX, window.height + 2 * HALO_PX)

    # boundless=True lets the halo hang off the edge of the mosaic, filled with
    # nodata, instead of failing at the antimeridian or at 85N.
    dem = src.read(1, window=window, boundless=True,
                   fill_value=src.nodata or 0).astype("float64")
    dem[dem > GEDTM30_NODATA_ABOVE] = np.nan
    return dem, src.window_transform(window)


def build(iso, margin_deg, area_frac, max_gap_deg, tile_deg, all_tiles,
          resume):
    iso = iso.upper()
    out_dir = os.path.join(OUT_ROOT, iso, BLOCK_NAME)
    checkpoint = os.path.join(out_dir, ".progress.npz")

    print(SEP)
    print("COUNTRY TERRAIN BLOCK  --  30 arcsec from GEDTM30 v1.2 at 1 arcsec")
    print(SEP)
    print(f"country     {iso}")
    print(f"source      {GEDTM30_DTM_URL}")
    print("            bare-earth DTM, CC-BY-4.0, read through windows only")
    print(f"output      {out_dir}/")

    grid = load_country_grid(iso, margin_deg, area_frac, max_gap_deg)
    inside = country_mask(grid).astype(bool)
    west, south, east, north = grid["bounds"]

    print(f"  extent    {west:.4f} {south:.4f} {east:.4f} {north:.4f}")
    print(f"  size      {grid['width']} x {grid['height']} at 30 arcsec")
    print(f"  native    {int(round(TARGET_RES / GEDTM30_NATIVE_RES))} x "
          f"{int(round(TARGET_RES / GEDTM30_NATIVE_RES))} = "
          f"{int(round(TARGET_RES / GEDTM30_NATIVE_RES)) ** 2} native pixels "
          "per target cell")

    if north > GEDTM30_NORTH_LIMIT or south < GEDTM30_SOUTH_LIMIT:
        print(f"  WARNING   GEDTM30 covers {GEDTM30_NORTH_LIMIT}N to "
              f"{GEDTM30_SOUTH_LIMIT}S; ground outside that will be nodata")

    tiles = tiles_for(grid, tile_deg, inside, all_tiles)
    total_tiles = ((grid["height"] + int(round(tile_deg / TARGET_RES)) - 1)
                   // int(round(tile_deg / TARGET_RES))) * \
                  ((grid["width"] + int(round(tile_deg / TARGET_RES)) - 1)
                   // int(round(tile_deg / TARGET_RES)))
    print(f"  tiles     {len(tiles)} of {total_tiles} at {tile_deg} deg "
          + ("(all)" if all_tiles else "(sea-only tiles skipped)"))

    stack = {name: np.full((grid["height"], grid["width"]), np.nan,
                           dtype="float32") for name in LAYERS}
    done = set()

    if resume and os.path.exists(checkpoint):
        saved = np.load(checkpoint, allow_pickle=False)
        if saved["shape"].tolist() == [grid["height"], grid["width"]]:
            for name in LAYERS:
                if name in saved:
                    stack[name] = saved[name]
            done = {tuple(t) for t in saved["done"].tolist()}
            print(f"  resumed   {len(done)} tiles already done")
        else:
            print("  note: checkpoint is for a different grid, ignoring it")

    remaining = [t for t in tiles if t not in done]
    if not remaining:
        print("  every tile is already done")

    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(f"/vsicurl/{GEDTM30_DTM_URL}") as src:
            if abs(src.transform.a - GEDTM30_NATIVE_RES) > 1e-12:
                raise RuntimeError(
                    f"GEDTM30 pixel is {src.transform.a}, expected "
                    f"{GEDTM30_NATIVE_RES}; the source grid has changed")
            crs = src.crs

            for index, tile in enumerate(
                    tqdm(remaining, desc="terrain tiles", unit="tile"), 1):
                row0, col0, rows, cols = tile
                dest_transform = grid["transform"] * Affine.translation(col0,
                                                                        row0)
                t_west = grid["transform"].c + col0 * TARGET_RES
                t_north = grid["transform"].f - row0 * TARGET_RES
                t_east = t_west + cols * TARGET_RES
                t_south = t_north - rows * TARGET_RES

                dem, native_transform = read_native_window(
                    src, t_west, t_south, t_east, t_north)
                if not np.isfinite(dem).any():
                    done.add(tile)
                    continue

                result = derive_tile(dem, native_transform, crs,
                                     dest_transform, (rows, cols))
                for name, array in result.items():
                    stack[name][row0:row0 + rows, col0:col0 + cols] = array
                done.add(tile)

                if index % CHECKPOINT_EVERY == 0:
                    np.savez(checkpoint,
                             shape=np.array([grid["height"], grid["width"]]),
                             done=np.array(sorted(done)), **stack)

    elapsed = time.time() - started
    print(f"\n  {len(remaining)} tiles in {elapsed / 60:.1f} min")

    print("\nwriting layers")
    written = []
    for name, (description, unit) in LAYERS.items():
        path = os.path.join(out_dir, f"{name}.tif")
        array = stack[name]
        data = np.where(np.isfinite(array), array, NODATA).astype("float32")
        with rasterio.open(path, "w", driver="GTiff", width=grid["width"],
                           height=grid["height"], count=1, dtype="float32",
                           crs=grid["crs"], transform=grid["transform"],
                           nodata=NODATA, compress="deflate", predictor=2,
                           tiled=True, blockxsize=256, blockysize=256,
                           BIGTIFF="IF_SAFER") as dst:
            dst.write(data, 1)
            dst.update_tags(
                COUNTRY=iso, DESCRIPTION=description, UNIT=unit,
                RESOLUTION="30 arcsec (~1 km)",
                SOURCE="GEDTM30 v1.2 (Ho et al. 2025, PeerJ 13:e19673), "
                       "CC-BY-4.0",
                NATIVE_RESOLUTION="1 arcsec (~30 m)",
                METHOD="slope and aspect computed per native pixel (Horn 3x3, "
                       "latitude-corrected), then reduced to 30 arcsec with "
                       "GDAL's warper; aspect decomposed into sin/cos "
                       "components weighted by sin(slope), never averaged as "
                       "degrees",
                DERIVED_BY="country_topography_rasters.py")
            dst.set_band_description(1, f"{description} ({unit})")
        written.append(path)

        valid = np.isfinite(array) & inside
        if valid.any():
            print(f"  {name:<16} mean {np.nanmean(array[valid]):9.3f}  "
                  f"range {np.nanmin(array[valid]):9.2f} .. "
                  f"{np.nanmax(array[valid]):9.2f}  "
                  f"coverage {100.0 * valid.sum() / inside.sum():5.1f}%")
        else:
            print(f"  {name:<16} NO VALID DATA inside the country")

    slope_mean = stack["slope_mean"]
    sane = np.isfinite(slope_mean) & inside
    if sane.any():
        print(f"\n  sanity: mean slope {np.nanmean(slope_mean[sane]):.2f} deg. "
              "Deriving this from an\n  already-coarsened DEM would report "
              "roughly an order of magnitude less, which is\n  why the native "
              "loop exists.")

    manifest = {
        "country": iso,
        "block": BLOCK_NAME,
        "resolution": "30 arcsec (1/120 deg, ~1 km)",
        "grid": {"crs": "EPSG:4326",
                 "transform": list(grid["transform"])[:6],
                 "width": grid["width"], "height": grid["height"],
                 "bounds": list(grid["bounds"]),
                 "defined_by": grid.get("grid_source", "country polygon")},
        "source": {
            "name": "GEDTM30 v1.2",
            "url": GEDTM30_DTM_URL,
            "native_resolution": "1 arcsec (~30 m)",
            "coverage": "85N to 65S",
            "licence": "CC-BY-4.0",
            "why": ("bare-earth digital TERRAIN model; surface models such as "
                    "Copernicus DEM and SRTM sit on the canopy and their "
                    "error tracks tree cover, which would leak the response "
                    "variable into elevation, slope and aspect alike"),
        },
        "method": {
            "slope": "Horn (1981) 3x3 at native 1 arcsec with a per-row "
                     "dx = res * 111320 * cos(lat), then reduced",
            "aspect": "decomposed into northness = cos(aspect) and "
                      "eastness = sin(aspect), weighted by sin(slope); a mean "
                      "of aspect DEGREES would be invalid",
            "reduction": "GDAL warper, because GEDTM30's grid is 4.5 native "
                         "pixels off the whole-degree line so a block "
                         "reduction would be half a pixel out of register",
            "validated_in": "validate_topography_alignment.py, which "
                            "cross-checked this slope against GEDTM30's own "
                            "published slope layer",
        },
        "tiles_processed": len(done),
        "layers": sorted(os.path.basename(p) for p in written),
        "caveats": [
            "Aspect is intrinsically a hillside-scale variable. Even at 1 km a "
            "cell in steep country contains slopes facing every direction, so "
            "they partly cancel; aspect_strength records how much. Expect "
            "northness and eastness to rank below slope and elev_std.",
            "GEDTM30 is a 2006-2015 product and is treated as static.",
        ],
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if os.path.exists(checkpoint):
        os.remove(checkpoint)

    total = sum(os.path.getsize(p) for p in written)
    print("\n" + SEP)
    print(f"DONE  --  {len(written)} layers, {total / 1e6:.1f} MB")
    print(SEP)
    print(f"Next: python3 verify_country_alignment.py {iso}")
    return out_dir


# ─────────────────────────────────────────────────────
# NAMES — satisfy sdm_features.discover_terrain without editing it
# ─────────────────────────────────────────────────────
# discover_terrain() matches TERRAIN_AUTODISCOVER keys as bare substrings of the
# filename, so the names this script writes natively are misread three ways:
#
#   aspect_strength.tif  matches "aspect" -> consumed as circular DEGREES and
#                        expanded through cos/sin, but it is a 0-1 magnitude.
#                        cos/sin of ~0.4 is two near-constant columns of noise.
#   slope_{mean,q3,max}  all match "slope" -> three layers all named "slope",
#                        because the key becomes the column name, not the file.
#   northness, eastness  match no key at all -> silently dropped, so the aspect
#                        signal never reaches the network.
#
# The fix is naming, not code: expose exactly one "*slope*" and exactly one
# "*aspect*", and keep the trigger substrings out of every other filename. The
# aspect layer is written in DEGREES 0-360 (not radians) so the consumer's own
# cos/sin expansion is correct, and it is the vector-mean direction recovered
# from the already slope-weighted components -- degrees are never averaged.
RENAMES = {
    "slope_mean": "slope",                        # the one "*slope*" layer
    "slope_q3": "steepness_q3",
    "slope_max": "steepness_max",
    "aspect_strength": "terrain_vector_strength",
}


def finalize_names(out_dir):
    """Derive aspect.tif in degrees, then rename to the discoverable set."""
    print("\nnames for auto-discovery")
    north_path = os.path.join(out_dir, "northness.tif")
    if not os.path.exists(north_path):
        print(f"  nothing to do -- {north_path} is missing")
        return []

    with rasterio.open(north_path) as src:
        profile, north = src.profile, src.read(1, masked=True)
    with rasterio.open(os.path.join(out_dir, "eastness.tif")) as src:
        east = src.read(1, masked=True)

    # atan2(sin, cos) inverts the decomposition exactly, so this is the circular
    # mean bearing, weighted by sin(slope) the same way the components were.
    aspect = np.degrees(np.arctan2(east.filled(np.nan),
                                   north.filled(np.nan))) % 360.0
    aspect = np.where(np.isfinite(aspect), aspect, NODATA).astype("float32")

    path = os.path.join(out_dir, "aspect.tif")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(aspect, 1)
        dst.update_tags(
            DESCRIPTION="slope-weighted circular mean aspect",
            UNIT="degrees clockwise from north, 0-360",
            NOT_RADIANS="degrees, so a consumer expanding this through cos/sin "
                        "as circular_degrees is correct",
            METHOD="atan2(eastness, northness); the components were averaged as "
                   "vectors weighted by sin(slope), so no degree value was ever "
                   "averaged directly",
            DERIVED_BY="country_topography_rasters.py")
        dst.set_band_description(1, "aspect (degrees clockwise from north)")
    print(f"  aspect.tif           degrees 0-360, vector mean of "
          f"northness/eastness")

    for old, new in RENAMES.items():
        src_path = os.path.join(out_dir, f"{old}.tif")
        if os.path.exists(src_path):
            os.replace(src_path, os.path.join(out_dir, f"{new}.tif"))
            print(f"  {old + '.tif':<21}-> {new}.tif")

    triggers = ("slope", "aspect", "tpi", "curvature")
    for key in triggers:
        hits = [n for n in sorted(os.listdir(out_dir))
                if n.endswith(".tif") and key in n.lower()]
        if len(hits) > 1:
            print(f"  WARNING: {len(hits)} files match '{key}': {hits}. The "
                  "consumer would give them all one column name.")
    return sorted(n for n in os.listdir(out_dir) if n.endswith(".tif"))


# ─────────────────────────────────────────────────────
# 10 ARCMIN — the same terrain on the existing block's grid
# ─────────────────────────────────────────────────────
# The 30 arcsec grid nests 20x20 inside the 10 arcmin one, so a plain block mean
# lands exactly on the coarse cell with no resampling. Averaging slope here is
# sound in a way that deriving it from a coarsened DEM is not: the gradient was
# still measured per 1 arcsec pixel, and this only pools those measurements.
# Aspect is pooled as vectors and only then turned back into degrees.
COARSE_FACTOR = 20


def aggregate_to_10m(fine_dir, coarse_dir):
    """Pool the 30 arcsec terrain block onto the 10 arcmin grid."""
    print("\n10 arcmin terrain (pooled from the 30 arcsec block)")
    os.makedirs(coarse_dir, exist_ok=True)

    def pooled(name):
        with rasterio.open(os.path.join(fine_dir, f"{name}.tif")) as src:
            array = src.read(1, masked=True).astype("float64").filled(np.nan)
            profile = src.profile
        h = array.shape[0] // COARSE_FACTOR * COARSE_FACTOR
        w = array.shape[1] // COARSE_FACTOR * COARSE_FACTOR
        blocks = array[:h, :w].reshape(h // COARSE_FACTOR, COARSE_FACTOR,
                                        w // COARSE_FACTOR, COARSE_FACTOR)
        with warnings.catch_warnings():        # all-sea blocks are legitimately
            warnings.simplefilter("ignore")    # empty; nanmean warns on those
            return np.nanmean(blocks, axis=(1, 3)), profile

    slope, fine_profile = pooled("slope")
    north, _ = pooled("northness")
    east, _ = pooled("eastness")
    elev, _ = pooled("elev_mean")
    rugged, _ = pooled("elev_std")
    aspect = np.degrees(np.arctan2(east, north)) % 360.0

    fine = fine_profile["transform"]
    profile = dict(fine_profile)
    profile.update(height=slope.shape[0], width=slope.shape[1],
                   transform=Affine(fine.a * COARSE_FACTOR, 0.0, fine.c,
                                             0.0, fine.e * COARSE_FACTOR, fine.f))

    # One "*slope*" file and one "*aspect*" file, for the same substring reason
    # finalize_names() exists. Elevation keeps the global block's filename so it
    # matches the consumer's explicit entry rather than being auto-discovered.
    layers = {"gedtm30_v1.2_elev_10m": (elev, "elevation", "m"),
              "slope_10m": (slope, "mean slope", "degrees"),
              "aspect_10m": (aspect, "slope-weighted circular mean aspect",
                             "degrees clockwise from north, 0-360"),
              "northness_10m": (north, "cos(aspect), slope-weighted", "-1..1"),
              "eastness_10m": (east, "sin(aspect), slope-weighted", "-1..1"),
              "elev_std_10m": (rugged, "elevation std, ruggedness", "m")}

    for name, (array, description, unit) in layers.items():
        path = os.path.join(coarse_dir, f"{name}.tif")
        data = np.where(np.isfinite(array), array, NODATA).astype("float32")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(DESCRIPTION=description, UNIT=unit,
                            RESOLUTION="10 arcmin",
                            METHOD=f"{COARSE_FACTOR}x{COARSE_FACTOR} block mean "
                                   "of the 30 arcsec block, whose slope and "
                                   "aspect were derived at 1 arcsec; aspect "
                                   "pooled as sin/cos vectors, never degrees",
                            DERIVED_BY="country_topography_rasters.py")
            dst.set_band_description(1, f"{description} ({unit})")
        print(f"  {name + '.tif':<30} {profile['width']} x {profile['height']}")
    return sorted(layers)


def main():
    parser = argparse.ArgumentParser(
        description="Derive the 30 arcsec country terrain block, including the "
                    "slope and aspect that never shipped.")
    parser.add_argument("iso", help="3-letter country code")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--area-frac", type=float, default=1.0)
    parser.add_argument("--max-gap", type=float, default=10.0)
    parser.add_argument("--tile-deg", type=float, default=TILE_DEG,
                        help=f"tile size in degrees (default {TILE_DEG}); "
                             "larger is fewer HTTP reads but more memory")
    parser.add_argument("--all-tiles", action="store_true",
                        help="process tiles that lie entirely outside the "
                             "country polygon too")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any checkpoint and start over")
    parser.add_argument("--names-only", action="store_true",
                        help="skip the terrain build; only derive aspect.tif, "
                             "apply the auto-discovery renames and pool to 10 "
                             "arcmin, for an existing block")
    args = parser.parse_args()

    iso = args.iso.upper()
    out_dir = os.path.join(OUT_ROOT, iso, BLOCK_NAME)
    coarse_dir = os.path.join(OUT_ROOT, iso, "gedtm30")

    if args.names_only:
        if not os.path.isdir(out_dir):
            print(f"ERROR: {out_dir} does not exist; run the build first")
            sys.exit(1)
        finalize_names(out_dir)
        aggregate_to_10m(out_dir, coarse_dir)
        return

    if build(iso, args.margin, args.area_frac, args.max_gap,
             args.tile_deg, args.all_tiles, not args.no_resume) is None:
        sys.exit(1)
    finalize_names(out_dir)
    aggregate_to_10m(out_dir, coarse_dir)


if __name__ == "__main__":
    main()
