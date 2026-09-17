"""
Download SATELLITE land-cover filters and align them to the WorldClim 10m grid.

These layers are site filters for recommend.py. They are NOT XGBoost predictors.
NDVI / tree cover measure trees that already exist, so training on them is circular.

Source : ESA WorldCover 10 m, 2021 v200 (Zanaga et al.). Public Cloud-Optimized
         GeoTIFFs on S3 — 3×3° tiles, class map (not a 120 GB download).
Method : stream one overview pyramid per tile over HTTP range requests (a few
         hundred KB, not the 10 m pixels) and average class membership onto the
         2160×1080 WorldClim grid. Same trick as soil_rasters.py.

Output filenames match what recommend.py already looks up:

    satellite/water_fraction_10m.tif
    satellite/snow_ice_fraction_10m.tif
    satellite/built_up_fraction_10m.tif
    satellite/cropland_fraction_10m.tif
    satellite/tree_cover_fraction_10m.tif     # UI context only — do not train
    satellite/land_fraction_10m.tif
    satellite/planting_exclusion_mask_10m.tif # water + snow/ice + wetland

Values are fractions in [0, 1]. Unmapped ocean cells are treated as water.

Usage
-----
python satellite_rasters.py              # global (~10 min, 8 workers)
python satellite_rasters.py --smoke      # 4 demo tiles, then recommend.py
python satellite_rasters.py --workers 12
python satellite_rasters.py --bbox -10,40,10,60
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject
from tqdm import tqdm

TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"
OUTPUT_DIR = "satellite"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
OVERSAMPLE = 2.0

GRID_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v100/2020/esa_worldcover_2020_grid.geojson"
)
URL_2021 = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
URL_2020 = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
    "/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile}_Map.tif"
)

# WorldCover discrete classes. 0 = nodata (ocean / unmapped).
TREE = (10, 95)          # tree cover + mangrove
CROP = (40,)
BUILT = (50,)
SNOW = (70,)
WATER = (80,)
WETLAND = (90,)
LAND = (10, 20, 30, 40, 50, 60, 90, 95, 100)

LAYER_SPECS = (
    ("water_fraction_10m.tif", WATER),
    ("snow_ice_fraction_10m.tif", SNOW),
    ("built_up_fraction_10m.tif", BUILT),
    ("cropland_fraction_10m.tif", CROP),
    ("tree_cover_fraction_10m.tif", TREE),
    ("land_fraction_10m.tif", LAND),
    ("planting_exclusion_mask_10m.tif", SNOW + WATER + WETLAND),
)

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_CACHEMAX": 128,
    "VSI_CACHE": True,
    "VSI_CACHE_SIZE": 268435456,
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_NUM_THREADS": 1,
}

TILE_RE = re.compile(r"^([NS])(\d+)([EW])(\d+)$")

# --smoke: London, Portland, Lyon, mid-Pacific (ocean → exclusion)
SMOKE_TILES = ("N51W003", "N45W123", "N45E003", "N00W150")


def read_template(path):
    with rasterio.open(path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "transform": src.transform,
            "crs": src.crs,
        }


def write_aligned(array, path, grid, tags):
    with rasterio.open(
        path, "w", driver="GTiff",
        height=grid["height"], width=grid["width"], count=1,
        dtype="float32", crs=grid["crs"], transform=grid["transform"],
        nodata=np.nan, compress="deflate", tiled=True,
    ) as dst:
        dst.write(array, 1)
        dst.update_tags(**tags)


def tile_square(ll_tile):
    """3×3° square from WorldCover lower-left tile id (e.g. N51W003)."""
    m = TILE_RE.match(ll_tile)
    if not m:
        raise ValueError(f"bad tile id: {ll_tile}")
    lat = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
    return lon, lat, lon + 3.0, lat + 3.0


def intersects(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def download_grid(path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        with open(path) as f:
            return json.load(f)
    print(f"Downloading WorldCover tile index → {path}")
    req = urllib.request.Request(GRID_URL, headers={"User-Agent": "hackathon-tree-poc/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return json.loads(data.decode("utf-8"))


def list_tiles(grid_geojson, bbox=None, smoke=False):
    if smoke:
        return list(SMOKE_TILES)
    tiles = []
    for feat in grid_geojson["features"]:
        ll = feat["properties"]["ll_tile"]
        if bbox is not None and not intersects(tile_square(ll), bbox):
            continue
        tiles.append(ll)
    return sorted(set(tiles))


def dest_window(bounds, grid, pad=1):
    """Dest pixels covering a geographic box, padded, clipped to the template."""
    left, bottom, right, top = bounds
    win = from_bounds(left, bottom, right, top, transform=grid["transform"])
    win = win.round_offsets(pixel_precision=1e-6).round_lengths(pixel_precision=1e-6)
    col = max(int(win.col_off) - pad, 0)
    row = max(int(win.row_off) - pad, 0)
    width = int(win.width) + 2 * pad
    height = int(win.height) + 2 * pad
    width = min(width, grid["width"] - col)
    height = min(height, grid["height"] - row)
    if width <= 0 or height <= 0:
        return None
    return Window(col, row, width, height)


def _open_tile(tile):
    last_err = None
    for template in (URL_2021, URL_2020):
        url = "/vsicurl/" + template.format(tile=tile)
        try:
            src = rasterio.open(url)
            return src, template
        except Exception as exc:
            last_err = exc
    raise last_err


def process_tile(tile, grid):
    """Return (tile, row, col, arrays[n_layers,h,w] or None, error).

    arrays is None with row/col set and error == 'unmapped' when ESA has no
    land tile there (ocean square) — fill as water in the parent process.
    """
    with rasterio.Env(**GDAL_ENV):
        try:
            src, used = _open_tile(tile)
        except Exception as exc:
            msg = str(exc).lower()
            missing = any(
                s in msg for s in ("404", "does not exist", "not recognized", "no such file")
            )
            if missing:
                win = dest_window(tile_square(tile), grid)
                if win is None:
                    return tile, None, None, None, None
                h, w = int(win.height), int(win.width)
                arrays = np.zeros((len(LAYER_SPECS), h, w), dtype="float32")
                names = [n for n, _ in LAYER_SPECS]
                arrays[names.index("water_fraction_10m.tif")] = 1.0
                arrays[names.index("planting_exclusion_mask_10m.tif")] = 1.0
                return tile, int(win.row_off), int(win.col_off), arrays, None
            return tile, None, None, None, f"open failed: {traceback.format_exc(limit=1)}"

        try:
            bounds = tile_square(tile)
            win = dest_window(bounds, grid)
            if win is None:
                return tile, None, None, None, None

            target_res = abs(grid["transform"].a)
            factor = 1
            for ov in sorted(src.overviews(1) or [1]):
                if src.res[0] * ov <= target_res / OVERSAMPLE:
                    factor = ov
            out_h = max(1, src.height // factor)
            out_w = max(1, src.width // factor)
            band = src.read(1, out_shape=(out_h, out_w))
            src_transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
            src_crs = src.crs
            nodata = src.nodata
        finally:
            src.close()

        valid = band != 0
        if nodata is not None:
            valid &= band != nodata

        src_stack = np.empty((len(LAYER_SPECS), band.shape[0], band.shape[1]), dtype="float32")
        src_stack[:] = np.nan
        for i, (_, codes) in enumerate(LAYER_SPECS):
            src_stack[i, valid] = np.isin(band[valid], codes).astype("float32")

        dst_transform = window_transform(win, grid["transform"])
        dst_stack = np.full(
            (len(LAYER_SPECS), int(win.height), int(win.width)), np.nan, dtype="float32"
        )
        reproject(
            source=src_stack, destination=dst_stack,
            src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
            dst_transform=dst_transform, dst_crs=grid["crs"], dst_nodata=np.nan,
            resampling=Resampling.average, num_threads=1,
        )
        return tile, int(win.row_off), int(win.col_off), dst_stack, None


def _worker(payload):
    tile, grid = payload
    try:
        return process_tile(tile, grid)
    except Exception:
        return tile, None, None, None, f"worker crash: {traceback.format_exc(limit=2)}"


def parse_bbox(text):
    if not text:
        return None
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--bbox must be min_lon,min_lat,max_lon,max_lat")
    return tuple(parts)


def accumulate(tiles, grid, workers):
    n_layers = len(LAYER_SPECS)
    sums = np.zeros((n_layers, grid["height"], grid["width"]), dtype="float64")
    counts = np.zeros((grid["height"], grid["width"]), dtype="float64")
    failed = []

    payloads = [(tile, grid) for tile in tiles]
    if workers <= 1:
        iterator = (_worker(p) for p in payloads)
        done_iter = iterator
        n = len(payloads)
        for result in tqdm(done_iter, total=n, desc="WorldCover tiles", unit="tile"):
            failed.extend(_ingest(result, sums, counts))
        return sums, counts, failed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, p) for p in payloads]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="WorldCover tiles", unit="tile"):
            failed.extend(_ingest(fut.result(), sums, counts))
    return sums, counts, failed


def _ingest(result, sums, counts):
    tile, row, col, arrays, err = result
    if err:
        return [(tile, err)]
    if arrays is None:
        return []
    h, w = arrays.shape[1], arrays.shape[2]
    sl = (slice(row, row + h), slice(col, col + w))
    finite = np.isfinite(arrays[0])
    counts[sl] += finite
    for i in range(arrays.shape[0]):
        chunk = np.nan_to_num(arrays[i], nan=0.0)
        sums[i][sl] += np.where(finite, chunk, 0.0)
    return []


def fractions_from_accum(sums, counts, fill_unmapped_as_water):
    """Tile-average fractions.

    Full global run: cells ESA never mapped are ocean → water/exclusion = 1.
    Smoke / bbox runs: leave the rest as NaN so recommend.py skips those pins
    instead of pretending Paris is underwater.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mapped = counts > 0
        out = {}
        for i, (name, _) in enumerate(LAYER_SPECS):
            frac = np.full(counts.shape, np.nan, dtype="float32")
            frac[mapped] = (sums[i][mapped] / counts[mapped]).astype("float32")
            out[name] = frac

    if fill_unmapped_as_water:
        uncovered = ~mapped
        out["water_fraction_10m.tif"][uncovered] = 1.0
        out["planting_exclusion_mask_10m.tif"][uncovered] = 1.0
        out["snow_ice_fraction_10m.tif"][uncovered] = 0.0
        out["built_up_fraction_10m.tif"][uncovered] = 0.0
        out["cropland_fraction_10m.tif"][uncovered] = 0.0
        out["tree_cover_fraction_10m.tif"][uncovered] = 0.0
        out["land_fraction_10m.tif"][uncovered] = 0.0
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Align ESA WorldCover to the WorldClim 10m grid as recommend.py filters."
    )
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel tile fetches (HTTP-bound). 1 = no multiprocessing.")
    p.add_argument("--bbox", type=str, default=None,
                   help="min_lon,min_lat,max_lon,max_lat — only tiles that intersect")
    p.add_argument("--smoke", action="store_true",
                   help="Four demo tiles (London, Portland, Lyon, Pacific). Fast check.")
    p.add_argument("--max-tiles", type=int, default=0,
                   help="Process at most N tiles (debug).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("=" * 50)
    print("SATELLITE FILTER RASTERS  (not for training)")
    print("=" * 50)
    print("Source: ESA WorldCover 10 m 2021 v200 (fallback: 2020 v100)")
    print("Role:   post-score filters for recommend.py")
    print("Do not add these files to collect_predictor_paths().")

    if not os.path.exists(TEMPLATE):
        raise SystemExit(f"Missing template {TEMPLATE} — run from hackathon_2026/")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    grid = read_template(TEMPLATE)
    print(
        f"\nTarget grid: {grid['width']}x{grid['height']} {grid['crs']} "
        f"pixel {abs(grid['transform'].a):.10f} deg"
    )

    geo = download_grid(os.path.join(RAW_DIR, "esa_worldcover_2020_grid.geojson"))
    tiles = list_tiles(geo, bbox=parse_bbox(args.bbox), smoke=args.smoke)
    if args.max_tiles:
        tiles = tiles[: args.max_tiles]
    print(f"Tiles to stream: {len(tiles)}  workers={args.workers}")
    if not tiles:
        raise SystemExit("No WorldCover tiles in this bbox.")

    sums, counts, failed = accumulate(tiles, grid, max(1, args.workers))
    fill_ocean = not (args.smoke or args.bbox)
    layers = fractions_from_accum(sums, counts, fill_unmapped_as_water=fill_ocean)
    if not fill_ocean:
        print("Partial run: unmapped cells stay NaN (recommend.py will skip them).")

    n_land = int((layers["land_fraction_10m.tif"] > 0).sum())
    n_excl = int((layers["planting_exclusion_mask_10m.tif"] > 0.5).sum())
    print(f"\nMapped cells with some land: {n_land:,}")
    print(f"Cells excluded (>50% water/ice/wetland or unmapped): {n_excl:,}")

    tags_base = {
        "satellite_source": "ESA WorldCover 10m 2021 v200 (2020 v100 fallback)",
        "satellite_doi": "10.5281/zenodo.5571936",
        "satellite_licence": "CC-BY 4.0",
        "satellite_role": "recommend.py site filter — do not train XGBoost on vegetation",
        "satellite_grid": "WorldClim 10 arc-minute 2160x1080",
    }
    for name, array in layers.items():
        path = os.path.join(OUTPUT_DIR, name)
        write_aligned(array, path, grid, {**tags_base, "satellite_layer": name})
        print(f"  wrote {path}")

    manifest = {
        "target_grid": {
            "width": grid["width"], "height": grid["height"],
            "crs": str(grid["crs"]),
            "transform": list(grid["transform"])[:6],
            "template": TEMPLATE,
        },
        "source": {
            "product": "ESA WorldCover 10m",
            "year": "2021 (v200), fallback 2020 (v100)",
            "native_resolution": "10 m (overviews streamed, never the full mosaic)",
            "licence": "CC-BY 4.0",
            "classes": {
                "10": "tree cover", "40": "cropland", "50": "built-up",
                "70": "snow/ice", "80": "permanent water", "90": "wetland",
                "95": "mangrove (counted as tree)",
            },
        },
        "n_tiles": len(tiles),
        "n_failed": len(failed),
        "failed_tiles": [t for t, _ in failed[:50]],
        "layers": [name for name, _ in LAYER_SPECS],
        "training": "NEVER — vegetation layers are circular with the SDM target",
        "recommend": "python recommend.py --lat 51.51 --lon -0.13",
    }
    man_path = os.path.join(OUTPUT_DIR, "MANIFEST.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {man_path}")
    if failed:
        print(f"WARNING: {len(failed)} tiles failed (see MANIFEST.json)")
        for tile, err in failed[:8]:
            print(f"  {tile}: {err.splitlines()[-1][:160]}")

    print("\nNext:")
    print("  python recommend.py --lat 51.51 --lon -0.13 --goal shade")
    print("  python recommend.py --lat 0 --lon -150          # ocean → blocked")
    print("=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
