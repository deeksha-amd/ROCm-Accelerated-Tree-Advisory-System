"""USA 1 km (30 arcsec) 2050 BIO via change-factor delta.

Does not upsample the 18 km future file and call it 1 km.

The USA training BIO in data/country_data/USA/climate_30s/ is 2015–2024:
fine pattern from WorldClim 30s 1970–2000, epoch shift from a 10-arcmin
anomaly. WorldClim CMIP6 uses the same split of labour.

This script:

  1. Reads the existing 10-arcmin ssp245 2041–2060 cube
     (data/climate_future_2050/…tif, 19 bands).
  2. Reads the matching 10-arcmin 2015–2024 BIO
     (data/country_data/USA/climate_recent/).
  3. Warps both onto the USA 30s grid (bilinear on the fields, then
     the difference — not on the climatology itself).
  4. Applies the anomaly to the 30s training BIO:
       temperature-like BIO  →  additive
       precipitation BIO     →  multiplicative (change factor)
  5. Writes data/country_data/USA/climate_future_2050_30s/wc2.1_30s_bio_{k}.tif

Soil, terrain, PET, wind, etc. are not projected. recommend_usa_30s.py
swaps only these 19 BIO layers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from repo_paths import data

TEMPLATE = data("country_data", "USA", "climate_30s", "wc2.1_30s_bio_1.tif")
RECENT_30S_DIR = data("country_data", "USA", "climate_30s")
RECENT_10M_DIR = data("country_data", "USA", "climate_recent")
FUTURE_10M = data(
    "climate_future_2050",
    "wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif",
)
OUT_DIR = data("country_data", "USA", "climate_future_2050_30s")

# BIO 12–14 and 16–19 are millimetres of rain. BIO 15 is a CV (percent-like).
PRECIP_MULTIPLICATIVE = {12, 13, 14, 16, 17, 18, 19}
PRECIP_FLOOR_MM = 1.0
WARP_FILL = -9999.0


def _finite(arr, nodata):
    out = arr.astype(np.float32, copy=False)
    if nodata is not None and np.isfinite(nodata):
        out = np.where(out == nodata, np.nan, out)
    out = np.where(np.abs(out) > 1e30, np.nan, out)
    return out


def load_template(path=TEMPLATE):
    if not os.path.isfile(path):
        raise SystemExit(f"Missing USA 30s template {path}")
    with rasterio.open(path) as src:
        return {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "profile": src.profile.copy(),
        }


def warp_band(src_path, template, band=1):
    """Bilinear resample one band onto the USA 30s grid."""
    dst = np.full((template["height"], template["width"]), WARP_FILL, np.float32)
    with rasterio.open(src_path) as src:
        arr = _finite(src.read(band), src.nodata)
        filled = np.where(np.isfinite(arr), arr, WARP_FILL).astype(np.float32)
        reproject(
            source=filled,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs or "EPSG:4326",
            dst_transform=template["transform"],
            dst_crs=template["crs"] or "EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=WARP_FILL,
            dst_nodata=WARP_FILL,
        )
    out = dst.astype(np.float32)
    out[out == WARP_FILL] = np.nan
    return out


def apply_anomaly(recent_30s, recent_10m_on_30s, future_10m_on_30s, bio_k):
    if bio_k in PRECIP_MULTIPLICATIVE:
        base = np.maximum(recent_10m_on_30s, PRECIP_FLOOR_MM)
        ratio = future_10m_on_30s / base
        ratio = np.clip(ratio, 0.05, 20.0)
        out = recent_30s * ratio
        out = np.maximum(out, 0.0)
    else:
        out = recent_30s + (future_10m_on_30s - recent_10m_on_30s)
    out = np.where(
        np.isfinite(recent_30s)
        & np.isfinite(recent_10m_on_30s)
        & np.isfinite(future_10m_on_30s),
        out,
        np.nan,
    )
    return out.astype(np.float32)


def write_tif(path, array, template):
    profile = template["profile"].copy()
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Delta-downscale CMIP6 2050 BIO onto the USA 1 km grid."
    )
    parser.add_argument("--future", default=FUTURE_10M)
    parser.add_argument("--recent-10m", default=RECENT_10M_DIR)
    parser.add_argument("--recent-30s", default=RECENT_30S_DIR)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    if not os.path.isfile(args.future):
        raise SystemExit(
            f"Missing {args.future}\nRun: python data/scripts/future_climate_rasters.py"
        )
    with rasterio.open(args.future) as src:
        if src.count < 19:
            raise SystemExit(f"{args.future} has {src.count} bands, need 19")

    template = load_template()
    print("=" * 62)
    print("USA 1 km 2050 BIO  (delta onto climate_30s, not a 10m upsample)")
    print("=" * 62)
    print(f"Future 10m: {args.future}")
    print(f"Recent 10m: {args.recent_10m}")
    print(f"Recent 30s: {args.recent_30s}")
    print(f"Output:     {args.out}")
    print(
        f"Grid:       {template['width']}x{template['height']}  "
        "bilinear anomaly, additive T / multiplicative P"
    )

    report = {
        "method": "change-factor delta: 30s_recent + warp(future_10m - recent_10m)",
        "gcm": "MPI-ESM1-2-HR",
        "ssp": "ssp245",
        "period": "2041-2060",
        "recent_epoch": "2015-2024",
        "precip_multiplicative": sorted(PRECIP_MULTIPLICATIVE),
        "layers": {},
    }
    os.makedirs(args.out, exist_ok=True)

    for k in range(1, 20):
        recent_10m_path = os.path.join(args.recent_10m, f"wc2.1_10m_bio_{k}.tif")
        recent_30s_path = os.path.join(args.recent_30s, f"wc2.1_30s_bio_{k}.tif")
        if not os.path.isfile(recent_10m_path):
            raise SystemExit(f"Missing {recent_10m_path}")
        if not os.path.isfile(recent_30s_path):
            raise SystemExit(f"Missing {recent_30s_path}")

        print(f"BIO {k:2d}  warping 10m fields …", flush=True)
        rec10 = warp_band(recent_10m_path, template)
        fut10 = warp_band(args.future, template, band=k)
        with rasterio.open(recent_30s_path) as src:
            rec30 = _finite(src.read(1), src.nodata)

        out = apply_anomaly(rec30, rec10, fut10, k)
        out_path = os.path.join(args.out, f"wc2.1_30s_bio_{k}.tif")
        write_tif(out_path, out, template)

        ok = np.isfinite(out)
        n_ok = int(ok.sum())
        mean_now = float(np.nanmean(rec30))
        mean_fut = float(np.nanmean(out))
        kind = "precip_ratio" if k in PRECIP_MULTIPLICATIVE else "temp_add"
        report["layers"][f"bio_{k}"] = {
            "kind": kind,
            "valid_cells": n_ok,
            "mean_recent_30s": mean_now,
            "mean_2050_30s": mean_fut,
            "mean_delta": mean_fut - mean_now,
        }
        print(
            f"         {kind:12s}  valid {n_ok:,}  "
            f"mean {mean_now:.3f} → {mean_fut:.3f}  "
            f"Δ {mean_fut - mean_now:+.3f}"
        )

    manifest = os.path.join(args.out, "MANIFEST.json")
    with open(manifest, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote 19 layers + {manifest}")
    print("Next: python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --html maps/austin.html")


if __name__ == "__main__":
    main()
