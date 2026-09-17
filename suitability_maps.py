"""Flagship species suitability maps: today vs 2050.

Scores every land cell with the saved XGBoost model. Soil and elevation stay
put; only the 19 BIO layers are swapped from the CMIP6 ssp245 2041–2060 file.
Does not retrain. Vegetation satellite layers are not features — they only
mask ocean/ice in the PNG.

Example
-------
python suitability_maps.py --species "Quercus robur"
python suitability_maps.py --species "Quercus robur" --bbox -15,35,40,72
"""

from __future__ import annotations

import argparse
import base64
import os
import struct
import zlib

import numpy as np
import rasterio

from recommend import (
    FUTURE_CLIMATE_PATH,
    TEMPLATE_RASTER,
    bio_number,
    load_booster,
)
from xgboost_training import PredictorRasters, collect_predictor_paths

OUT_DIR = "maps"
EXCLUSION = os.path.join("satellite", "planting_exclusion_mask_10m.tif")


def write_png(path, rgba):
    """RGBA uint8 PNG via stdlib zlib (no matplotlib / PIL required)."""
    h, w, c = rgba.shape
    if c != 4 or rgba.dtype != np.uint8:
        raise ValueError("need HxWx4 uint8")
    raw = b"".join(b"\x00" + rgba[i].tobytes() for i in range(h))

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def colorize_p(p, visible):
    """Cream (low p) → forest green (high p). Transparent off-land."""
    t = np.clip(np.nan_to_num(p, nan=0.0), 0.0, 1.0)
    rgba = np.zeros(p.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = (245 * (1.0 - t) + 22 * t).astype(np.uint8)
    rgba[..., 1] = (232 * (1.0 - t) + 128 * t).astype(np.uint8)
    rgba[..., 2] = (210 * (1.0 - t) + 55 * t).astype(np.uint8)
    rgba[..., 3] = np.where(visible, 220, 0).astype(np.uint8)
    rgba[~visible] = 0
    return rgba


def colorize_delta(delta, visible):
    """Brown (decline) → cream → green (expansion). Transparent off-land."""
    d = np.clip(np.nan_to_num(delta, nan=0.0), -0.4, 0.4) / 0.4
    rgba = np.zeros(delta.shape + (4,), dtype=np.uint8)
    pos = d >= 0
    neg = ~pos
    rgba[pos, 0] = (245 * (1 - d[pos]) + 22 * d[pos]).astype(np.uint8)
    rgba[pos, 1] = (232 * (1 - d[pos]) + 140 * d[pos]).astype(np.uint8)
    rgba[pos, 2] = (210 * (1 - d[pos]) + 55 * d[pos]).astype(np.uint8)
    rgba[neg, 0] = (245 * (1 + d[neg]) + 150 * (-d[neg])).astype(np.uint8)
    rgba[neg, 1] = (232 * (1 + d[neg]) + 72 * (-d[neg])).astype(np.uint8)
    rgba[neg, 2] = (210 * (1 + d[neg]) + 28 * (-d[neg])).astype(np.uint8)
    rgba[..., 3] = np.where(visible, 220, 0).astype(np.uint8)
    rgba[~visible] = 0
    return rgba


def crop_window(transform, height, width, bbox):
    west, south, east, north = bbox
    row0, col0 = rasterio.transform.rowcol(transform, west, north)
    row1, col1 = rasterio.transform.rowcol(transform, east, south)
    r0, r1 = sorted((int(row0), int(row1)))
    c0, c1 = sorted((int(col0), int(col1)))
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(height, r1 + 1)
    c1 = min(width, c1 + 1)
    return slice(r0, r1), slice(c0, c1)


def write_geotiff(path, array, template_src):
    with rasterio.open(
        path, "w", driver="GTiff",
        height=template_src.height, width=template_src.width, count=1,
        dtype="float32", crs=template_src.crs, transform=template_src.transform,
        nodata=np.nan, compress="deflate", tiled=True,
        blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(array.astype("float32"), 1)


def parse_bbox(text):
    if not text:
        return None
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--bbox must be min_lon,min_lat,max_lon,max_lat")
    return tuple(parts)


def load_future_bands(shape):
    if not os.path.isfile(FUTURE_CLIMATE_PATH):
        raise SystemExit(
            f"Missing {FUTURE_CLIMATE_PATH}\nRun: python future_climate_rasters.py"
        )
    height, width = shape
    with rasterio.open(FUTURE_CLIMATE_PATH) as src:
        if src.count < 19:
            raise SystemExit(f"{FUTURE_CLIMATE_PATH} has {src.count} bands, need 19")
        if (src.height, src.width) != (height, width):
            raise SystemExit(
                f"2050 grid {src.width}x{src.height} != today {width}x{height}"
            )
        bands = {}
        for k in range(1, 20):
            band = src.read(k).astype("float32")
            if src.nodata is not None:
                band = np.where(band == src.nodata, np.nan, band)
            band = np.where(np.abs(band) > 1e30, np.nan, band)
            bands[k] = band
    return bands


def score_grid(model, X):
    try:
        model.set_params(device="cpu")
    except Exception:
        pass
    # Chunk to keep host RAM predictable on 800k+ rows.
    out = np.empty(len(X), dtype=np.float32)
    step = 200_000
    for i in range(0, len(X), step):
        sl = slice(i, i + step)
        out[sl] = model.predict_proba(X[sl])[:, 1]
    return out


def png_data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def write_map_html(species, stem, bbox, out_dir):
    title = species
    west, south, east, north = bbox
    now_uri = png_data_uri(os.path.join(out_dir, f"{stem}_now.png"))
    fut_uri = png_data_uri(os.path.join(out_dir, f"{stem}_2050.png"))
    delta_uri = png_data_uri(os.path.join(out_dir, f"{stem}_delta.png"))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title} — today vs 2050</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  body {{ margin: 0; font-family: Georgia, serif; background: #f4f1ea; color: #12202b; }}
  header {{ padding: 16px 24px 8px; }}
  h1 {{ margin: 0 0 6px; font-size: 1.35rem; }}
  .sub {{ color: #5b6b76; font-size: 0.92rem; max-width: 70ch; line-height: 1.4; }}
  .panels {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; padding: 0 16px 16px; }}
  .panels figure {{ margin: 0; background: #fff; border: 1px solid #d9d1c3; border-radius: 10px; overflow: hidden; }}
  .panels img {{ width: 100%; display: block; background: #d7e0d4; }}
  figcaption {{ padding: 8px 10px 10px; font-size: 0.85rem; }}
  #map {{ height: 420px; margin: 0 16px 20px; border: 1px solid #d9d1c3; border-radius: 10px; }}
  .hint {{ padding: 0 24px 24px; font-size: 0.8rem; color: #5b6b76; }}
</style>
</head>
<body>
<header>
  <h1>{title}: climate suitability today and 2050</h1>
  <p class="sub">Same saved XGBoost niche, soil and elevation held fixed.
  2050 = CMIP6 MPI-ESM1-2-HR ssp245 (2041–2060) BIO layers only.
  ~18 km cells. Ocean and ice masked. Not a backyard map, and not trained on tree cover.</p>
</header>
<div class="panels">
  <figure><img src="{now_uri}" alt="today"/><figcaption>Today (WorldClim 1970–2000). Cream → green = higher p.</figcaption></figure>
  <figure><img src="{fut_uri}" alt="2050"/><figcaption>2050 projection. Same color scale.</figcaption></figure>
  <figure><img src="{delta_uri}" alt="change"/><figcaption>Change (2050 − today). Brown = worse match, green = better.</figcaption></figure>
</div>
<div id="map"></div>
<p class="hint">Toggle Today / 2050. Cell size is 1/6 degree. One HTML file — the paint is embedded, no extra PNGs needed.</p>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const bounds = [[{south}, {west}], [{north}, {east}]];
const map = L.map("map").fitBounds(bounds);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
  attribution: "Tiles &copy; Esri",
  maxZoom: 8
}}).addTo(map);
const now = L.imageOverlay("{now_uri}", bounds, {{opacity: 0.78}});
const fut = L.imageOverlay("{fut_uri}", bounds, {{opacity: 0.78}});
now.addTo(map);
L.control.layers({{"Today p": now, "2050 p": fut}}, {{}}, {{collapsed: false}}).addTo(map);
</script>
</body>
</html>
"""
    path = os.path.join(out_dir, f"{stem}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Today vs 2050 suitability GeoTIFFs + PNG/HTML.")
    p.add_argument("--species", default="Quercus robur",
                   help='Binomial matching models/Name_name.json')
    p.add_argument("--bbox", default="-15,35,40,72",
                   help="PNG/HTML crop: min_lon,min_lat,max_lon,max_lat (default western+central Europe)")
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--global-png", action="store_true",
                   help="Colorize the full globe instead of --bbox")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    species = args.species.strip()
    slug = species.replace(" ", "_")
    model_path = os.path.join("models", f"{slug}.json")
    if not os.path.isfile(model_path):
        raise SystemExit(f"No saved model at {model_path}")

    print("Loading 42 predictor layers…")
    rasters = PredictorRasters(collect_predictor_paths())
    print(f"Grid {rasters.width}x{rasters.height}, land cells {int(rasters.valid_mask.sum()):,}")

    print("Loading 2050 BIO (19 bands)…")
    future_bio = load_future_bands((rasters.height, rasters.width))
    stack_fut = rasters.stack.copy()
    for i, name in enumerate(rasters.names):
        k = bio_number(name)
        if k is not None:
            stack_fut[i] = future_bio[k]
    valid_fut = rasters.valid_mask & np.all(np.isfinite(stack_fut), axis=0)

    exclusion = None
    if os.path.isfile(EXCLUSION):
        with rasterio.open(EXCLUSION) as src:
            exclusion = src.read(1).astype("float32")
            if src.nodata is not None:
                exclusion = np.where(exclusion == src.nodata, np.nan, exclusion)

    print(f"Scoring {species} on CPU…")
    model = load_booster(model_path)
    rows, cols = np.where(valid_fut)
    X_now = rasters.stack[:, rows, cols].T
    X_fut = stack_fut[:, rows, cols].T
    p_now_vec = score_grid(model, X_now)
    p_fut_vec = score_grid(model, X_fut)

    p_now = np.full((rasters.height, rasters.width), np.nan, dtype="float32")
    p_fut = np.full_like(p_now, np.nan)
    p_now[rows, cols] = p_now_vec
    p_fut[rows, cols] = p_fut_vec
    delta = p_fut - p_now

    visible = valid_fut
    if exclusion is not None:
        visible = visible & ~(np.nan_to_num(exclusion, nan=0.0) > 0.5)

    os.makedirs(args.out, exist_ok=True)
    with rasterio.open(TEMPLATE_RASTER) as tmpl:
        write_geotiff(os.path.join(args.out, f"{slug}_p_now.tif"), p_now, tmpl)
        write_geotiff(os.path.join(args.out, f"{slug}_p_2050.tif"), p_fut, tmpl)
        write_geotiff(os.path.join(args.out, f"{slug}_p_delta.tif"), delta, tmpl)
        transform = tmpl.transform
        height, width = tmpl.height, tmpl.width

    bbox = None if args.global_png else parse_bbox(args.bbox)
    if bbox:
        rs, cs = crop_window(transform, height, width, bbox)
        p_now_v = p_now[rs, cs]
        p_fut_v = p_fut[rs, cs]
        delta_v = delta[rs, cs]
        vis_v = visible[rs, cs]
        west, south, east, north = bbox
    else:
        p_now_v, p_fut_v, delta_v, vis_v = p_now, p_fut, delta, visible
        west, north = transform * (0, 0)
        east, south = transform * (width, height)

    stem = slug
    write_png(os.path.join(args.out, f"{stem}_now.png"), colorize_p(p_now_v, vis_v))
    write_png(os.path.join(args.out, f"{stem}_2050.png"), colorize_p(p_fut_v, vis_v))
    write_png(os.path.join(args.out, f"{stem}_delta.png"), colorize_delta(delta_v, vis_v))
    html = write_map_html(species, stem, (west, south, east, north), args.out)

    n = int(vis_v.sum())
    mean_now = float(np.nanmean(p_now_v[vis_v])) if n else float("nan")
    mean_fut = float(np.nanmean(p_fut_v[vis_v])) if n else float("nan")
    print(f"Visible cells in frame: {n:,}")
    print(f"Mean p today {mean_now:.3f}  2050 {mean_fut:.3f}  delta {mean_fut-mean_now:+.3f}")
    print(f"Wrote {html}")
    print("Open that file plus suggest.html from recommend.py --html for the pin demo.")


if __name__ == "__main__":
    main()
