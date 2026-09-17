"""USA 1 km climate suitability maps: today vs 2050 (BIO delta only).

Scores a bbox of the CONUS 30-arcsec grid with a saved data/models_usa_30s booster.
Does not load the full 5 GB stack — only the window covering --bbox.
Does not retrain. Satellite is not a feature.

    python future_climate_usa_30s.py          # once, 19 BIO layers
    python suitability_maps_usa_30s.py --species "Quercus virginiana"
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from poc.recommend import bio_number, load_booster
from poc.suitability_maps import (
    colorize_delta,
    colorize_p,
    png_data_uri,
    score_grid,
    write_png,
)
from repo_paths import data, usa_maps
from xgboost_training_usa_30s import MODEL_DIR, collect_predictor_paths, species_slug

OUT_DIR = usa_maps()
FUTURE_30S_DIR = data("country_data", "USA", "climate_future_2050_30s")
TEMPLATE = data("country_data", "USA", "climate_30s", "wc2.1_30s_bio_1.tif")
# South-central US: live oak, Austin, Gulf coast — 1 km grain is visible.
DEFAULT_BBOX = "-107,24.5,-88,37"


def parse_bbox(text):
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--bbox must be min_lon,min_lat,max_lon,max_lat")
    return tuple(parts)


def _clean_band(band, nodata):
    out = band.astype(np.float32, copy=False)
    if nodata is not None and np.isfinite(nodata):
        out = np.where(out == nodata, np.nan, out)
    return np.where(np.abs(out) > 1e30, np.nan, out)


def window_from_bbox(src, bbox):
    west, south, east, north = bbox
    win = from_bounds(west, south, east, north, transform=src.transform)
    win = win.round_offsets().round_lengths()
    full = Window(0, 0, src.width, src.height)
    win = win.intersection(full)
    if win.width <= 0 or win.height <= 0:
        raise SystemExit("bbox does not overlap the USA 30s grid")
    return win


def read_stack_window(layer_paths, bbox):
    first = layer_paths[0][1]
    with rasterio.open(first) as src:
        win = window_from_bbox(src, bbox)
        transform = src.window_transform(win)
        profile = src.profile.copy()
        height, width = int(win.height), int(win.width)
        crs = src.crs
    names = []
    bands = []
    for name, path in layer_paths:
        with rasterio.open(path) as src:
            band = _clean_band(src.read(1, window=win), src.nodata)
            if band.shape != (height, width):
                raise SystemExit(f"{path} window {band.shape} != {(height, width)}")
        names.append(name)
        bands.append(band)
    stack = np.stack(bands, axis=0)
    profile.update(
        height=height,
        width=width,
        transform=transform,
        crs=crs,
        count=1,
        dtype="float32",
        nodata=np.nan,
    )
    west, north = transform * (0, 0)
    east, south = transform * (width, height)
    return {
        "names": names,
        "stack": stack,
        "transform": transform,
        "profile": profile,
        "bounds": (float(min(west, east)), float(min(south, north)),
                   float(max(west, east)), float(max(south, north))),
    }


def apply_future_bio(stack, names, bbox):
    """Copy stack; swap BIO layers from the 1 km 2050 delta files."""
    out = stack.copy()
    n_swapped = 0
    for i, name in enumerate(names):
        k = bio_number(name)
        if k is None:
            continue
        path = os.path.join(FUTURE_30S_DIR, f"wc2.1_30s_bio_{k}.tif")
        if not os.path.isfile(path):
            return None, f"missing {path}"
        with rasterio.open(path) as src:
            win = window_from_bbox(src, bbox)
            band = _clean_band(src.read(1, window=win), src.nodata)
        if band.shape != out[i].shape:
            return None, f"BIO{k} window mismatch"
        out[i] = band
        n_swapped += 1
    if n_swapped != 19:
        return None, f"swapped {n_swapped} BIO layers, expected 19"
    return out, None


def write_map_html(species, stem, bbox, out_dir, have_2050):
    west, south, east, north = bbox
    now_uri = png_data_uri(os.path.join(out_dir, f"{stem}_now.png"))
    fut_block = ""
    layers_js = '{"Today p": now}'
    if have_2050:
        fut_uri = png_data_uri(os.path.join(out_dir, f"{stem}_2050.png"))
        delta_uri = png_data_uri(os.path.join(out_dir, f"{stem}_delta.png"))
        fut_block = f"""
  <figure><img src="{fut_uri}" alt="2050"/><figcaption>2050 (delta-downscaled ssp245). Same color scale.</figcaption></figure>
  <figure><img src="{delta_uri}" alt="change"/><figcaption>Change (2050 − today). Brown = worse, green = better.</figcaption></figure>"""
        layers_js = '{"Today p": now, "2050 p": fut}'
        fut_overlay = f'const fut = L.imageOverlay("{fut_uri}", bounds, {{opacity: 0.78}});'
    else:
        fut_overlay = "const fut = null;"
    cols = "1fr 1fr 1fr" if have_2050 else "1fr"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{species} — USA 1 km suitability</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  body {{ margin: 0; font-family: Georgia, serif; background: #f4f1ea; color: #12202b; }}
  header {{ padding: 16px 24px 8px; }}
  h1 {{ margin: 0 0 6px; font-size: 1.35rem; }}
  .sub {{ color: #5b6b76; font-size: 0.92rem; max-width: 78ch; line-height: 1.4; }}
  .panels {{ display: grid; grid-template-columns: {cols}; gap: 8px; padding: 0 16px 16px; }}
  .panels figure {{ margin: 0; background: #fff; border: 1px solid #d9d1c3; border-radius: 10px; overflow: hidden; }}
  .panels img {{ width: 100%; display: block; background: #d7e0d4; }}
  figcaption {{ padding: 8px 10px 10px; font-size: 0.85rem; }}
  #map {{ height: 420px; margin: 0 16px 20px; border: 1px solid #d9d1c3; border-radius: 10px; }}
  .hint {{ padding: 0 24px 24px; font-size: 0.8rem; color: #5b6b76; }}
</style>
</head>
<body>
<header>
  <h1>{species}: USA 1 km climate suitability</h1>
  <p class="sub">Saved XGBoost niche on the CONUS 30-arc-second grid (~1 km).
  Soil and terrain held fixed. 2050 swaps only the 19 BIO layers (CMIP6
  MPI-ESM1-2-HR ssp245 2041–2060, delta onto the 2015–2024 1 km climate —
  not a resized 18 km cube). Not trained on tree cover.</p>
</header>
<div class="panels">
  <figure><img src="{now_uri}" alt="today"/><figcaption>Today (2015–2024 1 km BIO). Cream → green = higher p.</figcaption></figure>
  {fut_block}
</div>
<div id="map"></div>
<p class="hint">Toggle layers. This is a 1 km cell, still not a backyard.</p>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const bounds = [[{south}, {west}], [{north}, {east}]];
const map = L.map("map").fitBounds(bounds);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
  attribution: "Tiles &copy; Esri",
  maxZoom: 10
}}).addTo(map);
const now = L.imageOverlay("{now_uri}", bounds, {{opacity: 0.78}});
{fut_overlay}
now.addTo(map);
L.control.layers({layers_js}, {{}}, {{collapsed: false}}).addTo(map);
</script>
</body>
</html>
"""
    path = os.path.join(out_dir, f"{stem}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="USA 1 km today vs 2050 suitability maps.")
    p.add_argument("--species", default="Quercus virginiana")
    p.add_argument("--bbox", default=DEFAULT_BBOX)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--model-dir", default=MODEL_DIR)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    species = args.species.strip()
    slug = species_slug(species)
    model_path = os.path.join(args.model_dir, f"{slug}.json")
    if not os.path.isfile(model_path):
        raise SystemExit(f"No saved USA 1 km model at {model_path}")

    bbox = parse_bbox(args.bbox)
    print(f"Loading USA 30s predictors in bbox {bbox} …")
    layers = collect_predictor_paths()
    block = read_stack_window(layers, bbox)
    stack = block["stack"]
    valid = np.all(np.isfinite(stack), axis=0)
    print(f"Window {stack.shape[2]}x{stack.shape[1]}  valid cells {int(valid.sum()):,}")

    stack_fut, fut_err = apply_future_bio(stack, block["names"], bbox)
    have_2050 = stack_fut is not None
    if not have_2050:
        print(f"2050 skipped: {fut_err}")
        valid_fut = valid
    else:
        valid_fut = valid & np.all(np.isfinite(stack_fut), axis=0)

    model = load_booster(model_path)
    rows, cols = np.where(valid_fut)
    if len(rows) == 0:
        raise SystemExit("No valid cells in bbox")
    print(f"Scoring {species} on {len(rows):,} cells …")
    p_now_vec = score_grid(model, stack[:, rows, cols].T)
    p_now = np.full(valid.shape, np.nan, np.float32)
    p_now[rows, cols] = p_now_vec
    p_fut = None
    delta = None
    if have_2050:
        p_fut_vec = score_grid(model, stack_fut[:, rows, cols].T)
        p_fut = np.full_like(p_now, np.nan)
        p_fut[rows, cols] = p_fut_vec
        delta = p_fut - p_now

    os.makedirs(args.out, exist_ok=True)
    stem = f"{slug}_usa_30s"
    profile = block["profile"].copy()
    profile.update(
        count=1, dtype="float32", nodata=np.nan, compress="deflate",
        tiled=True, blockxsize=256, blockysize=256,
    )

    def dump(path, arr):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr.astype("float32"), 1)

    dump(os.path.join(args.out, f"{stem}_p_now.tif"), p_now)
    if have_2050:
        dump(os.path.join(args.out, f"{stem}_p_2050.tif"), p_fut)
        dump(os.path.join(args.out, f"{stem}_p_delta.tif"), delta)

    vis = valid_fut
    write_png(os.path.join(args.out, f"{stem}_now.png"), colorize_p(p_now, vis))
    if have_2050:
        write_png(os.path.join(args.out, f"{stem}_2050.png"), colorize_p(p_fut, vis))
        write_png(os.path.join(args.out, f"{stem}_delta.png"), colorize_delta(delta, vis))
    html = write_map_html(species, stem, block["bounds"], args.out, have_2050)
    n = int(vis.sum())
    print(f"Visible cells: {n:,}")
    print(f"Mean p today {float(np.nanmean(p_now[vis])):.3f}", end="")
    if have_2050:
        print(f"  2050 {float(np.nanmean(p_fut[vis])):.3f}  "
              f"delta {float(np.nanmean(delta[vis])):+.3f}")
    else:
        print()
    print(f"Wrote {html}")


if __name__ == "__main__":
    main()
