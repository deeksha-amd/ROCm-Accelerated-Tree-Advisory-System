"""
Build the country-scoped SOIL block at 30 arcsec (~1 km): 11 properties at two
depths, from sources that are genuinely finer than the target grid.

WHY THIS IS A REAL GAIN AND NOT JUST MORE PIXELS
    soil_data/aligned_10m/ was built from SoilGrids' 5 km AGGREGATED product,
    because at an 18.5 km target there was no point paying for anything finer.
    At 1 km that reasoning inverts: 5 km data upsampled to 1 km would be four
    empty pixels for every real one. So this block reads the native products
    instead -- SoilGrids 2.0 at 250 m and OpenLandMap-soildb at 120 m, both
    comfortably finer than a 900 m cell, so aggregating them is averaging real
    measurements rather than interpolating coarse ones.

    Neither is downloaded in bulk. SoilGrids' tiled COGs and OpenLandMap's
    multi-gigabyte mosaics are read through a window, so a country costs a few
    MB per layer.

SOURCE HIERARCHY, UNCHANGED FROM THE GLOBAL BLOCK
    Keeping the same provenance rules means the country block is comparable
    with the global one rather than a different dataset that happens to share
    a name.

    OpenLandMap-soildb (2026 release, 2020-2022 data) is primary for ph_h2o,
        sand, silt, clay, bdod, soc and socd. It is the newest product and it
        is native EPSG:4326, so no reprojection is needed at all.
    SoilGrids 2.0 supplies cec, nitrogen and cfvo, which OpenLandMap does not
        publish, and backfills OpenLandMap's deliberate gaps -- it excludes hot
        deserts and stops at 76N, which cost ~12% of land cells globally.
    drainage is DERIVED from sand and clay, not measured. It is a relative
        ranking index, and derive_drainage() in soil_rasters.py is reused so
        the country and global versions cannot drift apart.

    source_flag_30s.tif records which source won per cell (1 = OpenLandMap,
    2 = SoilGrids). As in the global block it describes the seven OpenLandMap
    properties, and reading it for cec, nitrogen or cfvo would be misleading,
    since those are SoilGrids everywhere by construction.

PROJECTIONS
    SoilGrids 2.0 is served in Interrupted Goode Homolosine, not EPSG:4326, so
    it must be warped rather than sliced. Both sources therefore go through one
    WarpedVRT that lands directly on the country grid with area-weighted
    averaging, which does the reprojection and the aggregation in a single step
    and cannot leave a half-pixel registration error behind.

    Averaging is done in float32, not in the sources' uint8/int16 storage
    types: averaging pH stored as pH x 10 in uint8 and rounding back would
    throw away most of the precision the averaging just bought.

DEPTHS
    0-30cm is the topsoil holding the nutrients and feeder roots that decide
    whether a sapling establishes; 30-60cm is the subsoil holding structural
    roots and the moisture reserve a mature tree draws on in drought.
    SoilGrids publishes thinner intervals, composed here by thickness-weighted
    mean: 0-30 from 0-5, 5-15 and 15-30; 30-60 directly.

OUTPUTS  (country_data/<ISO>/soil_30s/)
    <property>_<depth>_30s.tif   for 11 properties x 2 depths
    source_flag_30s.tif
    MANIFEST.json

USAGE
    python3 country_soil_rasters.py IND
    python3 country_soil_rasters.py IND --no-backfill
    python3 country_soil_rasters.py IND --depths 0-30cm
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from tqdm import tqdm

import soil_rasters as sr
from country_clip import RES_30S, country_grid, country_mask

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUT_ROOT = "country_data"
BLOCK_NAME = "soil_30s"
TARGET_RES = RES_30S

DEPTHS = ("0-30cm", "30-60cm")

# SoilGrids 2.0 at NATIVE 250 m, not the 5 km aggregate the global block used.
#
# The 250 m product is served as a .vrt over thousands of small tiles with no
# overviews, so a country read is dominated by HTTP round trips: measured at
# ~9 s per layer for a 2.5 x 4.5 degree window, which scales with area and
# becomes the slowest part of the whole pipeline for a large country.
#
# SG_RESOLUTIONS therefore offers the 1 km aggregate as a documented fast path.
# It is a single global GeoTIFF per layer (80-210 MB, range-readable), so it is
# far cheaper to window. At a 900 m target the two carry nearly the same
# information -- 250 m averaged 3.6x versus 1 km resampled slightly -- but
# averaging the native product is the more defensible of the two, so it stays
# the default and the fast path is opt-in.
SG_BASE = "https://files.isric.org/soilgrids/latest"
SG_RESOLUTIONS = {
    "250m": ("native, tiled COGs behind a .vrt", "{p}/{p}_{part}_mean.vrt"),
    "1000m": ("1 km aggregate, one global COG per layer",
              "{p}/{p}_{part}_mean_1000.tif"),
}
SG_DEFAULT_RESOLUTION = "250m"

# OpenLandMap at 120 m. The 30 m release exists but is ~60 GB per layer and a
# 900 m target cannot use the detail, so 120 m is the right rung.
OLM_RES = "120m"

# Reuse the global block's definitions so the two cannot drift apart.
OLM_LAYERS = sr.OLM_LAYERS
OLM_DEPTH_TAG = sr.OLM_DEPTH_TAG
SG_PROPERTIES = sr.SG_PROPERTIES
SG_DEPTH_PARTS = sr.SG_DEPTH_PARTS
SG_ONLY = sr.SG_ONLY
SOURCE_OPENLANDMAP = sr.SOURCE_OPENLANDMAP
SOURCE_SOILGRIDS = sr.SOURCE_SOILGRIDS
PERIOD = sr.PERIOD

NODATA = np.float32(np.nan)

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "200000000",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

SEP = "=" * 70


# ─────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────
def load_country_grid(iso, margin_deg, area_frac, max_gap_deg):
    """The authoritative country grid, from country_grid.json where it exists."""
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
    grid.update(transform=Affine(*saved["transform"]), width=saved["width"],
                height=saved["height"], bounds=tuple(saved["bounds"]),
                grid_source=os.path.basename(path))
    print(f"  grid adopted from {path}")
    return grid


# ─────────────────────────────────────────────────────
# READING
# ─────────────────────────────────────────────────────
def read_onto_grid(url, grid, scale=None):
    """Warp and area-average one remote raster straight onto the country grid.

    One WarpedVRT does the reprojection and the aggregation together. Doing
    them separately would mean resampling twice, and each resampling is another
    chance to end up half a cell out of register.

    dtype="float32" forces the averaging into floating point. The sources store
    scaled integers -- pH as pH x 10 in uint8, texture as g/kg in int16 -- and
    averaging in those types would round away the precision.
    """
    with rasterio.open(f"/vsicurl/{url}") as src:
        factor = scale if scale is not None else (src.scales[0] or 1.0)
        with WarpedVRT(src, crs=grid["crs"], transform=grid["transform"],
                       width=grid["width"], height=grid["height"],
                       dtype="float32", nodata=float("nan"),
                       resampling=Resampling.average) as vrt:
            array = vrt.read(1)

    array = array.astype("float32")
    array[~np.isfinite(array)] = np.nan
    return array * np.float32(factor)


def olm_url(name, depth):
    folder, stem, version, _ = OLM_LAYERS[name]
    return (f"{sr.OLM_S3}/{folder}/{stem}_m_{OLM_RES}_"
            f"{OLM_DEPTH_TAG[depth]}_{PERIOD}_g_epsg.4326_{version}.tif")


def sg_url(sg_name, part, resolution=SG_DEFAULT_RESOLUTION):
    tail = SG_RESOLUTIONS[resolution][1].format(p=sg_name, part=part)
    folder = "data" if resolution == "250m" else f"data_aggregated/{resolution}"
    return f"{SG_BASE}/{folder}/{tail}"


def fetch_openlandmap(depths, grid):
    """Every OpenLandMap property/depth on the country grid."""
    jobs = [(name, depth) for name in OLM_LAYERS for depth in depths]
    layers = {}
    for name, depth in tqdm(jobs, desc="OpenLandMap 120m", unit="layer"):
        layers[(name, depth)] = read_onto_grid(olm_url(name, depth), grid)
    return layers


def fetch_soilgrids(depths, grid, properties,
                    resolution=SG_DEFAULT_RESOLUTION):
    """Every SoilGrids property on the country grid, depths composed.

    The thinner source intervals are combined by thickness-weighted mean, so a
    cell with 0-5 and 5-15 present but 15-30 missing still gets a value from
    what exists rather than being dropped.
    """
    jobs = [(sg_name, depth) for sg_name in properties for depth in depths]
    layers = {}
    for sg_name, depth in tqdm(jobs, desc=f"SoilGrids {resolution}",
                               unit="layer"):
        our_name, divisor, _ = SG_PROPERTIES[sg_name]
        stack, weights = [], []
        for part, thickness in SG_DEPTH_PARTS[depth]:
            stack.append(read_onto_grid(sg_url(sg_name, part, resolution),
                                        grid, scale=1.0 / divisor))
            weights.append(thickness)

        stacked = np.stack(stack)
        w = np.asarray(weights, dtype="float32")[:, None, None]
        present = np.isfinite(stacked)
        wsum = (w * present).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            layers[(our_name, depth)] = np.where(
                wsum > 0, np.nansum(stacked * w, axis=0) / np.maximum(wsum, 1e-6),
                np.nan).astype("float32")
    return layers


# ─────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────
def build(iso, margin_deg, area_frac, max_gap_deg, depths, backfill,
          sg_resolution=SG_DEFAULT_RESOLUTION):
    iso = iso.upper()
    out_dir = os.path.join(OUT_ROOT, iso, BLOCK_NAME)

    print(SEP)
    print("COUNTRY SOIL BLOCK  --  30 arcsec (~1 km)")
    print(SEP)
    print(f"country     {iso}")
    print(f"primary     OpenLandMap-soildb {OLM_RES}, native EPSG:4326, "
          f"epoch {PERIOD[:4]}-{PERIOD[9:13]}")
    print(f"secondary   SoilGrids 2.0 at {sg_resolution} "
          f"({SG_RESOLUTIONS[sg_resolution][0]}; Homolosine, warped)")
    print(f"depths      {', '.join(depths)}")
    print(f"output      {out_dir}/")

    grid = load_country_grid(iso, margin_deg, area_frac, max_gap_deg)
    inside = country_mask(grid).astype(bool)
    print(f"  size      {grid['width']} x {grid['height']} at 30 arcsec, "
          f"{int(inside.sum()):,} cells in country")

    with rasterio.Env(**GDAL_ENV):
        print("\n[1/4] OpenLandMap-soildb (windowed reads, nothing downloaded "
              "whole)")
        olm_layers = fetch_openlandmap(depths, grid)

        properties = sorted(SG_PROPERTIES if backfill else SG_ONLY)
        print(f"\n[2/4] SoilGrids 2.0, {len(properties)} properties"
              + ("" if backfill else " (SoilGrids-only properties; "
                                     "no backfill)"))
        sg_layers = fetch_soilgrids(depths, grid, properties,
                                    sg_resolution)

    print("\n[3/4] merging -- newest source wins, SoilGrids fills its gaps")
    merged, provenance = sr.merge_sources(olm_layers, sg_layers)
    sr.derive_drainage(merged, provenance, depths)

    print("\n[4/4] writing")
    units = dict(
        [(name, OLM_LAYERS[name][3]) for name in OLM_LAYERS]
        + [(our, unit) for our, _, unit in SG_PROPERTIES.values()]
        + [("drainage", "index 1-7 (1=very poor, 7=excessive)")])

    os.makedirs(out_dir, exist_ok=True)
    written, coverage = [], []
    for (prop, depth), array in sorted(merged.items()):
        flag = provenance[(prop, depth)]
        n_olm = int((flag == SOURCE_OPENLANDMAP).sum())
        n_sg = int((flag == SOURCE_SOILGRIDS).sum())

        path = os.path.join(out_dir, f"{prop}_{depth}_30s.tif")
        with rasterio.open(path, "w", driver="GTiff", width=grid["width"],
                           height=grid["height"], count=1, dtype="float32",
                           crs=grid["crs"], transform=grid["transform"],
                           nodata=NODATA, compress="deflate", predictor=2,
                           tiled=True, blockxsize=256, blockysize=256,
                           BIGTIFF="IF_SAFER") as dst:
            dst.write(array.astype("float32"), 1)
            dst.update_tags(
                COUNTRY=iso, soil_property=prop, soil_depth=depth,
                soil_units=units.get(prop, "?"),
                RESOLUTION="30 arcsec (~1 km)",
                soil_source=("OpenLandMap-soildb 2020-2022 at 120 m, "
                             f"SoilGrids 2.0 at {sg_resolution} where "
                             "absent"),
                soil_cells_openlandmap=str(n_olm),
                soil_cells_soilgrids=str(n_sg),
                DERIVED_BY="country_soil_rasters.py")
            dst.set_band_description(1, f"{prop} {depth} "
                                        f"({units.get(prop, '?')})")
        written.append(path)

        valid = np.isfinite(array) & inside
        share = 100.0 * valid.sum() / max(int(inside.sum()), 1)
        coverage.append(share)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(array[valid]) if valid.any() else float("nan")
        print(f"  {prop + ' ' + depth:<22} mean {mean:9.3f} "
              f"{units.get(prop, '?'):<28} coverage {share:5.1f}%")

    flag_path = os.path.join(out_dir, "source_flag_30s.tif")
    any_flag = provenance[("ph_h2o", depths[0])]
    with rasterio.open(flag_path, "w", driver="GTiff", width=grid["width"],
                       height=grid["height"], count=1, dtype="uint8",
                       crs=grid["crs"], transform=grid["transform"],
                       nodata=0, compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256) as dst:
        dst.write(any_flag.astype("uint8"), 1)
        dst.update_tags(
            COUNTRY=iso,
            DESCRIPTION="1 = OpenLandMap-soildb, 2 = SoilGrids 2.0, 0 = "
                        "neither",
            CAVEAT="describes the seven OpenLandMap properties only; cec, "
                   "nitrogen and cfvo are SoilGrids everywhere because "
                   "OpenLandMap does not publish them",
            DERIVED_BY="country_soil_rasters.py")
    written.append(flag_path)

    n_olm = int((any_flag == SOURCE_OPENLANDMAP).sum())
    n_sg = int((any_flag == SOURCE_SOILGRIDS).sum())
    print(f"\n  provenance: {n_olm:,} cells OpenLandMap, {n_sg:,} cells "
          "backfilled from SoilGrids")
    print(f"  coverage:   min {min(coverage):.1f}%, median "
          f"{float(np.median(coverage)):.1f}% of in-country cells")

    manifest = {
        "country": iso,
        "block": BLOCK_NAME,
        "resolution": "30 arcsec (1/120 deg, ~1 km)",
        "grid": {"crs": "EPSG:4326",
                 "transform": list(grid["transform"])[:6],
                 "width": grid["width"], "height": grid["height"],
                 "bounds": list(grid["bounds"]),
                 "defined_by": grid.get("grid_source", "country polygon")},
        "depths": list(depths),
        "sources": {
            "OpenLandMap-soildb": {
                "resolution": f"{OLM_RES} native, EPSG:4326",
                "epoch": "2020-2022",
                "properties": sorted(OLM_LAYERS),
                "role": "primary",
                "licence": "CC-BY 4.0",
            },
            "SoilGrids 2.0": {
                "resolution": f"{sg_resolution} "
                              f"({SG_RESOLUTIONS[sg_resolution][0]}), "
                              "Interrupted Goode Homolosine, warped to "
                              "EPSG:4326",
                "epoch": "2020 release",
                "properties": sorted(SG_ONLY) + (
                    ["+ backfill for the OpenLandMap properties"]
                    if backfill else []),
                "role": "properties OpenLandMap does not model, plus gap "
                        "backfill",
                "licence": "CC-BY 4.0",
            },
        },
        "derived": {"drainage": "1 + 6 * clip((sand - clay + 100) / 200); a "
                                "relative ranking index, not a measurement"},
        "backfill_enabled": backfill,
        "layers": sorted(os.path.basename(p) for p in written),
        "caveats": [
            "The country block reads the NATIVE 250 m / 120 m products, while "
            "soil_data/aligned_10m/ was built from the 5 km aggregate. Values "
            "will therefore differ slightly from a block-reduction of the "
            "global block; that is the point, not an error.",
            "nitrogen and organic carbon shift over decades, so treating them "
            "as static is weaker than it is for texture and stones.",
            "source_flag_30s.tif describes the OpenLandMap properties only.",
        ],
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = sum(os.path.getsize(p) for p in written)
    print("\n" + SEP)
    print(f"DONE  --  {len(written)} layers, {total / 1e6:.1f} MB")
    print(SEP)
    print(f"Next: python3 verify_country_alignment.py {iso}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build the 30 arcsec country soil block from native-"
                    "resolution SoilGrids and OpenLandMap.")
    parser.add_argument("iso", help="3-letter country code")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--area-frac", type=float, default=1.0)
    parser.add_argument("--max-gap", type=float, default=10.0)
    parser.add_argument("--depths", nargs="+", default=list(DEPTHS),
                        choices=sorted(SG_DEPTH_PARTS))
    parser.add_argument("--sg-resolution", default=SG_DEFAULT_RESOLUTION,
                        choices=sorted(SG_RESOLUTIONS),
                        help="SoilGrids resolution; 250m is native and the "
                             "default, 1000m is the much faster aggregate")
    parser.add_argument("--no-backfill", action="store_true",
                        help="do not fill OpenLandMap's desert and polar gaps "
                             "from SoilGrids")
    args = parser.parse_args()

    if build(args.iso, args.margin, args.area_frac, args.max_gap,
             tuple(args.depths), not args.no_backfill,
             args.sg_resolution) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
