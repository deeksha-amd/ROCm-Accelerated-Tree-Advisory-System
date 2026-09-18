"""
Download SOIL rasters and align them to the WorldClim 10m grid

Primary source : OpenLandMap-soildb v20250204 / v20250523  (newest global soil
                 product available; described in Hengl et al., ESSD 18, 989-1013,
                 published 06 Feb 2026).  Native EPSG:4326, 30 m / 120 m COGs,
                 time-explicit, latest epoch 2020-2022.
Gap-filler     : SoilGrids 2.0 (ISRIC, 2020) for CEC, total nitrogen and coarse
                 fragments only -- OpenLandMap-soildb does not model these and
                 no newer global product does either.

The 120 m OpenLandMap mosaics are ~5 GB each, so we never download them whole.
They are Cloud-Optimized GeoTIFFs, so we read one level of their internal
overview pyramid over HTTP range requests (a few MB per layer) and average that
onto the 2160x1080 WorldClim grid.
"""

import os
import json
import sys
import requests
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from repo_paths import data

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUTPUT_DIR = data("soil_data")
TEMPLATE = data("climate_current", "wc2.1_10m_bio_1.tif")   # defines the target grid

MODE = "sample"      # "sample" = 2 depths, 9 properties  (~150 MB, a few minutes)
                     # "full"   = 3 depths, 9 properties  (~250 MB)

# Depth intervals. OpenLandMap-soildb uses 0-30 / 30-60 / 60-100 cm.
# 0-30 cm  = fine-feeder-root and nutrient zone -> drives establishment
# 30-60 cm = structural rooting zone            -> drives anchorage + dry-season water
# 60-100cm = deep water reserve                 -> only matters for mature/dryland trees
DEPTHS_SAMPLE = ["0-30cm", "30-60cm"]
DEPTHS_FULL = ["0-30cm", "30-60cm", "60-100cm"]

# Epoch to use from the time-explicit OpenLandMap product (newest available).
PERIOD = "20200101_20221231"

# ── OpenLandMap-soildb (newest product, native EPSG:4326) ──
OLM_S3 = "https://s3.opengeohub.org/global-soil"
OLM_V_CORE = "v20250204"      # pH, bulk density, organic carbon
OLM_V_TEXTURE = "v20250523"   # sand / silt / clay (re-released May 2025)
OLM_RES = "120m"              # "120m" (~5 GB/layer) or "30m" (~60 GB/layer);
                              # we only ever read overviews, so 120m is plenty
                              # for a 10-arcminute target grid

# name -> (folder, file stem, version, unit after scaling)
OLM_LAYERS = {
    "ph_h2o":  ("global_soil_props_v20250204_mosaics",
                "ph.h2o_iso.10390.2021.index", OLM_V_CORE, "pH"),
    "soc":     ("global_soil_props_v20250204_mosaics",
                "oc_iso.10694.1995.wpml", OLM_V_CORE, "g/kg"),
    "socd":    ("global_soil_props_v20250204_mosaics",
                "oc_iso.10694.1995.mg.cm3", OLM_V_CORE, "kg/m3"),
    "bdod":    ("global_soil_props_v20250204_mosaics",
                "bd.core_iso.11272.2017.g.cm3", OLM_V_CORE, "g/cm3"),
    "clay":    ("global_soil_props_v20250523",
                "clay.tot_iso.11277.2020.wpct", OLM_V_TEXTURE, "%"),
    "sand":    ("global_soil_props_v20250523",
                "sand.tot_iso.11277.2020.wpct", OLM_V_TEXTURE, "%"),
    "silt":    ("global_soil_props_v20250523",
                "silt.tot_iso.11277.2020.wpct", OLM_V_TEXTURE, "%"),
}
OLM_DEPTH_TAG = {"0-30cm": "b0cm..30cm",
                 "30-60cm": "b30cm..60cm",
                 "60-100cm": "b60cm..100cm"}

# ── SoilGrids 2.0, 5 km aggregated ──
# Used for two jobs:
#   1. properties OpenLandMap-soildb does not model at all (CEC, nitrogen,
#      coarse fragments) -- no newer global product provides these either;
#   2. backfilling the places OpenLandMap-soildb deliberately leaves empty,
#      namely hot deserts and everything above 76N / below 56S.
SG_BASE = "https://files.isric.org/soilgrids/latest/data_aggregated/5000m"
SG_PROPERTIES = {        # SoilGrids name -> (our name, divisor, unit)
    "cec":      ("cec", 10.0, "cmol(c)/kg"),        # mmol(c)/kg -> cmol(c)/kg
    "nitrogen": ("nitrogen", 100.0, "g/kg"),        # cg/kg      -> g/kg
    "cfvo":     ("cfvo", 10.0, "vol%"),             # cm3/dm3    -> vol%
    "phh2o":    ("ph_h2o", 10.0, "pH"),             # pH*10      -> pH
    "sand":     ("sand", 10.0, "%"),                # g/kg       -> %
    "silt":     ("silt", 10.0, "%"),
    "clay":     ("clay", 10.0, "%"),
    "bdod":     ("bdod", 100.0, "g/cm3"),           # cg/cm3     -> g/cm3
    "soc":      ("soc", 10.0, "g/kg"),              # dg/kg      -> g/kg
    "ocd":      ("socd", 10.0, "kg/m3"),            # hg/m3      -> kg/m3
}
SG_ONLY = {"cec", "nitrogen", "cfvo"}   # nothing newer maps these globally

BACKFILL_WITH_SOILGRIDS = True   # fill OpenLandMap's desert and polar gaps
                                 # with SoilGrids; set False to keep a single
                                 # source at the cost of ~67,000 land cells
# SoilGrids uses thinner intervals; these compose into our target depths by
# thickness-weighted averaging.
SG_DEPTH_PARTS = {
    "0-30cm":   [("0-5cm", 5), ("5-15cm", 10), ("15-30cm", 15)],
    "30-60cm":  [("30-60cm", 30)],
    "60-100cm": [("60-100cm", 40)],
}

# Read the overview level whose pixel is at least this much finer than the
# 0.1666667 deg target, so that averaging has something to average.
OVERSAMPLE = 2.0


def download_file(url, output_path, attempts=5):
    """Download with progress bar.

    files.isric.org drops long connections fairly often, so an interrupted
    transfer is resumed with a Range request rather than restarted.
    """
    head = requests.head(url, timeout=60)
    if head.status_code != 200:
        print(f"ERROR: could not download (status {head.status_code})")
        print(f"URL tried: {url}")
        return False
    total_size = int(head.headers.get('content-length', 0))

    desc = os.path.basename(output_path)[:38]
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
        for attempt in range(attempts):
            done = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            if total_size and done >= total_size:
                return True
            pbar.n = done
            pbar.refresh()

            headers = {"Range": f"bytes={done}-"} if done else {}
            try:
                r = requests.get(url, stream=True, timeout=120, headers=headers)
                with open(output_path, 'ab' if done else 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            except requests.exceptions.RequestException as exc:
                print(f"  retry {attempt + 1}/{attempts} after {type(exc).__name__}")
                continue

            if not total_size or os.path.getsize(output_path) >= total_size:
                return True

    print(f"ERROR: incomplete download after {attempts} attempts: {url}")
    return False


# ─────────────────────────────────────────────────────
# GRID ALIGNMENT
# ─────────────────────────────────────────────────────
def read_template(path):
    """Return the target grid definition taken from a WorldClim raster."""
    with rasterio.open(path) as src:
        return {"width": src.width, "height": src.height,
                "transform": src.transform, "crs": src.crs}


def regrid(src, grid, scale):
    """Average one open raster onto the target grid, returning real units.

    Reads an internal overview rather than full resolution, so only a few MB
    travel over the network for a multi-gigabyte source.
    """
    target_res = abs(grid["transform"].a)
    factor = 1
    for ov in sorted(src.overviews(1) or [1]):
        if src.res[0] * ov <= target_res / OVERSAMPLE:
            factor = ov
    out_h, out_w = src.height // factor, src.width // factor

    band = src.read(1, out_shape=(out_h, out_w))
    src_transform = src.transform * Affine.scale(src.width / out_w,
                                                 src.height / out_h)

    # Move to float and mark nodata as NaN *before* averaging, so that the
    # nodata sentinel (255 / 32767 / -32768) never pollutes a cell mean.
    data = band.astype("float32")
    if src.nodata is not None:
        data[band == src.nodata] = np.nan
    data *= scale

    out = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    reproject(source=data, destination=out,
              src_transform=src_transform, src_crs=src.crs, src_nodata=np.nan,
              dst_transform=grid["transform"], dst_crs=grid["crs"],
              dst_nodata=np.nan, resampling=Resampling.average, num_threads=4)
    return out


def write_aligned(array, path, grid, tags):
    """Write a float32 layer on the target grid with NaN as nodata."""
    with rasterio.open(path, "w", driver="GTiff", height=grid["height"],
                       width=grid["width"], count=1, dtype="float32",
                       crs=grid["crs"], transform=grid["transform"],
                       nodata=np.nan, compress="deflate", tiled=True) as dst:
        dst.write(array, 1)
        dst.update_tags(**tags)


# ─────────────────────────────────────────────────────
# SOURCE 1: OpenLandMap-soildb  (streamed, never fully downloaded)
# ─────────────────────────────────────────────────────
def olm_url(name, depth):
    folder, stem, version, _ = OLM_LAYERS[name]
    fname = (f"{stem}_m_{OLM_RES}_{OLM_DEPTH_TAG[depth]}_{PERIOD}"
             f"_g_epsg.4326_{version}.tif")
    return f"{OLM_S3}/{folder}/{fname}"


def fetch_openlandmap(depths, grid):
    """Stream each OpenLandMap COG overview onto the target grid."""
    layers = {}
    jobs = [(n, d) for n in OLM_LAYERS for d in depths]

    for name, depth in tqdm(jobs, desc="OpenLandMap-soildb", unit="layer"):
        with rasterio.open("/vsicurl/" + olm_url(name, depth)) as src:
            scale = src.scales[0] if src.scales else 1.0
            layers[(name, depth)] = regrid(src, grid, scale)
    return layers


# ─────────────────────────────────────────────────────
# SOURCE 2: SoilGrids 2.0 -- CEC / nitrogen / coarse fragments only
# ─────────────────────────────────────────────────────
def fetch_soilgrids(depths, grid, raw_dir, properties):
    """Download the small 5 km SoilGrids files and compose target depths."""
    layers = {}

    needed = sorted({(sg_name, part)
                     for sg_name in properties
                     for depth in depths
                     for part, _ in SG_DEPTH_PARTS[depth]})
    for sg_name, part in needed:
        fname = f"{sg_name}_{part}_mean_5000.tif"
        # download_file checks the size on disk first, so this both skips
        # files already present and resumes ones left half-written.
        if not download_file(f"{SG_BASE}/{sg_name}/{fname}",
                             os.path.join(raw_dir, fname)):
            raise RuntimeError(f"could not retrieve {fname}")

    for sg_name in properties:
        our_name, divisor, _ = SG_PROPERTIES[sg_name]
        for depth in depths:
            stack, weights = [], []
            for part, thickness in SG_DEPTH_PARTS[depth]:
                path = os.path.join(raw_dir, f"{sg_name}_{part}_mean_5000.tif")
                with rasterio.open(path) as src:
                    stack.append(regrid(src, grid, 1.0 / divisor))
                weights.append(thickness)

            # Thickness-weighted mean across the SoilGrids sub-intervals
            stacked = np.stack(stack)
            w = np.array(weights, dtype="float32")[:, None, None]
            wsum = (w * np.isfinite(stacked)).sum(axis=0)
            with np.errstate(invalid="ignore"):
                layers[(our_name, depth)] = np.where(
                    wsum > 0, np.nansum(stacked * w, axis=0) / wsum, np.nan
                ).astype("float32")
    return layers


# ─────────────────────────────────────────────────────
# MERGE: newest source wins, SoilGrids fills its holes
# ─────────────────────────────────────────────────────
SOURCE_OPENLANDMAP = 1
SOURCE_SOILGRIDS = 2


def merge_sources(olm_layers, sg_layers):
    """Prefer OpenLandMap-soildb, fall back to SoilGrids where it is empty."""
    merged, provenance = {}, {}

    for key in set(olm_layers) | set(sg_layers):
        primary = olm_layers.get(key)
        backup = sg_layers.get(key)

        if primary is None:
            merged[key] = backup
            flag = np.where(np.isfinite(backup), SOURCE_SOILGRIDS, 0)
        elif backup is None:
            merged[key] = primary
            flag = np.where(np.isfinite(primary), SOURCE_OPENLANDMAP, 0)
        else:
            use_primary = np.isfinite(primary)
            merged[key] = np.where(use_primary, primary, backup).astype("float32")
            flag = np.where(use_primary, SOURCE_OPENLANDMAP,
                            np.where(np.isfinite(backup), SOURCE_SOILGRIDS, 0))
        provenance[key] = flag.astype("uint8")

    return merged, provenance


# ─────────────────────────────────────────────────────
# DERIVED: drainage
# ─────────────────────────────────────────────────────
def derive_drainage(merged, provenance, depths):
    """Texture-based drainage index, 1 (very poor) to 7 (excessive).

    No global product maps drainage class directly, so we approximate it the
    way soil surveys do: water moves faster as sand rises and slower as clay
    rises. This is a relative index for ranking sites, not a measured property.
    """
    for depth in depths:
        sand = merged.get(("sand", depth))
        clay = merged.get(("clay", depth))
        if sand is None or clay is None:
            continue

        index = 1.0 + 6.0 * np.clip((sand - clay + 100.0) / 200.0, 0.0, 1.0)
        index[~np.isfinite(sand) | ~np.isfinite(clay)] = np.nan
        merged[("drainage", depth)] = index.astype("float32")
        provenance[("drainage", depth)] = provenance[("sand", depth)]


def main():
    print("=" * 50)
    print("SOIL RASTER DOWNLOADER")
    print("=" * 50)

    depths = DEPTHS_SAMPLE if MODE == "sample" else DEPTHS_FULL
    print(f"Mode:       {MODE}")
    print(f"Depths:     {', '.join(depths)}")
    print(f"Primary:    OpenLandMap-soildb {OLM_V_CORE}/{OLM_V_TEXTURE} "
          f"({OLM_RES}, EPSG:4326, epoch 2020-2022)")
    print(f"Gap-filler: SoilGrids 2.0 5 km  (CEC, nitrogen, coarse fragments)")

    aligned_dir = os.path.join(OUTPUT_DIR, "aligned_10m")
    raw_dir = os.path.join(OUTPUT_DIR, "soilgrids_5km")
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    grid = read_template(TEMPLATE)
    print(f"\nTarget grid: {grid['width']}x{grid['height']} {grid['crs']} "
          f"pixel {abs(grid['transform'].a):.10f} deg")

    print("\nStreaming OpenLandMap-soildb overviews (range requests only)...")
    olm_layers = fetch_openlandmap(depths, grid)

    sg_properties = sorted(SG_PROPERTIES if BACKFILL_WITH_SOILGRIDS
                           else SG_ONLY)
    print(f"\nDownloading SoilGrids 2.0 5 km layers "
          f"({len(sg_properties)} properties)...")
    sg_layers = fetch_soilgrids(depths, grid, raw_dir, sg_properties)

    print("\nMerging sources (OpenLandMap-soildb first, SoilGrids in its gaps)...")
    merged, provenance = merge_sources(olm_layers, sg_layers)

    print("Deriving drainage index from texture...")
    derive_drainage(merged, provenance, depths)

    print("Writing aligned layers...")
    written = {}
    units = dict(
        [(n, OLM_LAYERS[n][3]) for n in OLM_LAYERS]
        + [(our, unit) for our, _, unit in SG_PROPERTIES.values()]
        + [("drainage", "index 1-7 (1=very poor, 7=excessive)")]
    )
    for (prop, depth), array in sorted(merged.items()):
        flag = provenance[(prop, depth)]
        n_olm = int((flag == SOURCE_OPENLANDMAP).sum())
        n_sg = int((flag == SOURCE_SOILGRIDS).sum())
        out = os.path.join(aligned_dir, f"{prop}_{depth}_10m.tif")
        write_aligned(array, out, grid, {
            "soil_property": prop,
            "soil_depth": depth,
            "soil_units": units.get(prop, "?"),
            "soil_source": ("OpenLandMap-soildb 2020-2022, "
                            "SoilGrids 2.0 where absent"),
            "soil_cells_openlandmap": str(n_olm),
            "soil_cells_soilgrids": str(n_sg),
        })
        written[f"{prop}_{depth}"] = out

    # One provenance raster so the modeller can see which source fed each cell.
    flag_path = os.path.join(aligned_dir, "source_flag_10m.tif")
    with rasterio.open(flag_path, "w", driver="GTiff", height=grid["height"],
                       width=grid["width"], count=1, dtype="uint8",
                       crs=grid["crs"], transform=grid["transform"], nodata=0,
                       compress="deflate", tiled=True) as dst:
        dst.write(provenance[("ph_h2o", depths[0])], 1)
        dst.update_tags(description="0 = no soil data, "
                                    "1 = OpenLandMap-soildb, "
                                    "2 = SoilGrids 2.0 backfill")

    manifest = {
        "target_grid": {"width": grid["width"], "height": grid["height"],
                        "crs": str(grid["crs"]),
                        "transform": list(grid["transform"])[:6],
                        "template": TEMPLATE},
        "depths": depths,
        "layers": sorted(written),
        "sources": {
            "OpenLandMap-soildb": {
                "version": f"{OLM_V_CORE} (core) / {OLM_V_TEXTURE} (texture)",
                "published": "2026-02-06 (ESSD 18, 989-1013)",
                "doi": "10.5194/essd-18-989-2026",
                "data_doi": "10.5281/zenodo.15470431",
                "epoch": "2020-2022",
                "native_crs": "EPSG:4326",
                "native_resolution": OLM_RES,
                "licence": "CC-BY 4.0",
                "properties": sorted(OLM_LAYERS),
            },
            "SoilGrids 2.0": {
                "version": "2.0",
                "published": "2020 (paper: SOIL 7, 217-240, 2021)",
                "doi": "10.5194/soil-7-217-2021",
                "epoch": "historical WoSIS profiles, March 2020 snapshot",
                "native_crs": "Homolosine (+proj=igh)",
                "native_resolution": "250 m (5 km aggregate used here)",
                "licence": "CC-BY 4.0",
                "properties": sg_properties,
                "role": ("CEC, nitrogen and coarse fragments outright; "
                         "elsewhere only where OpenLandMap-soildb is empty"
                         if BACKFILL_WITH_SOILGRIDS else
                         "CEC, nitrogen and coarse fragments only"),
            },
        },
        "source_flag": {"raster": "aligned_10m/source_flag_10m.tif",
                        "0": "no soil data", "1": "OpenLandMap-soildb",
                        "2": "SoilGrids 2.0 backfill"},
    }
    with open(os.path.join(OUTPUT_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Wrote {len(written)} aligned layers to {aligned_dir}/")
    for key in sorted(written):
        print(f"  {key}")
    print(f"\nManifest: {OUTPUT_DIR}/MANIFEST.json")
    print("Next: run validate_soil_alignment.py")


if __name__ == "__main__":
    main()
