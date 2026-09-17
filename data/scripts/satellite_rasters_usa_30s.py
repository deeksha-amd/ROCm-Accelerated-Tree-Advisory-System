"""Align ESA WorldCover to the USA 1 km (30 arcsec) grid as recommend filters.

Not an XGBoost predictor. Vegetation measures trees that already exist.

Leaves data/scripts/satellite_rasters.py (global ~18 km) untouched. Reuses
its tile fetch + average, but the destination is the CONUS 30s template
(7020×3060), so a pin and the satellite mix share the same ~1 km cell.

    python data/scripts/satellite_rasters_usa_30s.py --smoke   # Austin + Portland
    python data/scripts/satellite_rasters_usa_30s.py           # CONUS

Writes data/country_data/USA/satellite_30s/*_30s.tif and
DO_NOT_TRAIN_ON_THIS.md. recommend_usa_30s.py reads those files; it does
not fall back to the 18 km global stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import satellite_rasters as sat
from clean_species_usa_30s import TEMPLATE, read_grid
from repo_paths import data

OUTPUT_DIR = data("country_data", "USA", "satellite_30s")
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
# Austin downtown, Portland — enough to demo recommend_usa_30s without CONUS.
USA_SMOKE_TILES = ("N30W099", "N45W123")

DO_NOT_TRAIN = """# Do not train on anything in this directory

These layers are **site filters for recommend_usa_30s.py**, not predictors.

They are ESA WorldCover 10 m class fractions averaged onto the USA 30-arc-second
grid (~1 km). Tree cover and NDVI answer “are there trees here already?”, which
is circular for a species-distribution model.

`xgboost_training_usa_30s.py` already refuses any path containing `satellite`.
"""


def to_30s_name(name_10m):
    return name_10m.replace("_10m.tif", "_30s.tif")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Align ESA WorldCover to the USA 1 km grid as recommend filters."
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel tile fetches. 1 = no multiprocessing.",
    )
    p.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="min_lon,min_lat,max_lon,max_lat — default is the CONUS 30s envelope",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Austin + Portland WorldCover tiles only. Fast check.",
    )
    p.add_argument("--max-tiles", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("=" * 50)
    print("USA 1 km SATELLITE FILTERS  (not for training)")
    print("=" * 50)
    print("Source: ESA WorldCover 10 m 2021 v200 (fallback: 2020 v100)")
    print("Role:   post-score filters for recommend_usa_30s.py")
    print("Do not add these files to collect_predictor_paths().")

    if not os.path.isfile(TEMPLATE):
        raise SystemExit(
            f"Missing USA 30s template {TEMPLATE} — need data/country_data/USA/climate_30s/"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    marker = os.path.join(OUTPUT_DIR, "DO_NOT_TRAIN_ON_THIS.md")
    with open(marker, "w", encoding="utf-8") as handle:
        handle.write(DO_NOT_TRAIN)

    grid_meta = read_grid()
    grid = sat.read_template(TEMPLATE)
    print(
        f"\nTarget grid: {grid['width']}x{grid['height']} {grid['crs']} "
        f"pixel {abs(grid['transform'].a):.10f} deg"
    )

    geo = sat.download_grid(os.path.join(RAW_DIR, "esa_worldcover_2020_grid.geojson"))
    if args.smoke:
        tiles = list(USA_SMOKE_TILES)
    else:
        bbox = sat.parse_bbox(args.bbox) if args.bbox else tuple(grid_meta["bbox"])
        tiles = sat.list_tiles(geo, bbox=bbox, smoke=False)
    if args.max_tiles:
        tiles = tiles[: args.max_tiles]
    print(f"Tiles to stream: {len(tiles)}  workers={args.workers}")
    if not tiles:
        raise SystemExit("No WorldCover tiles in this bbox.")

    sums, counts, failed = sat.accumulate(tiles, grid, max(1, args.workers))
    # Smoke / default CONUS bbox: leave unmapped cells as NaN, never fake ocean.
    layers = sat.fractions_from_accum(sums, counts, fill_unmapped_as_water=False)
    print("Unmapped cells stay NaN (recommend skips them).")

    n_land = int((layers["land_fraction_10m.tif"] > 0).sum())
    n_excl = int((layers["planting_exclusion_mask_10m.tif"] > 0.5).sum())
    print(f"\nMapped cells with some land: {n_land:,}")
    print(f"Cells excluded (>50% water/ice/wetland): {n_excl:,}")

    tags_base = {
        "satellite_source": "ESA WorldCover 10m 2021 v200 (2020 v100 fallback)",
        "satellite_doi": "10.5281/zenodo.5571936",
        "satellite_licence": "CC-BY 4.0",
        "satellite_role": "recommend_usa_30s.py site filter — do not train",
        "satellite_grid": "USA 30 arc-second 7020x3060",
    }
    written = []
    for name_10m, array in layers.items():
        name = to_30s_name(name_10m)
        path = os.path.join(OUTPUT_DIR, name)
        sat.write_aligned(array, path, grid, {**tags_base, "satellite_layer": name})
        written.append(name)
        print(f"  wrote {path}")

    manifest = {
        "target_grid": {
            "width": grid["width"],
            "height": grid["height"],
            "crs": str(grid["crs"]),
            "transform": list(grid["transform"])[:6],
            "template": TEMPLATE,
        },
        "source": {
            "product": "ESA WorldCover 10m",
            "year": "2021 (v200), fallback 2020 (v100)",
            "native_resolution": "10 m (overviews streamed, never the full mosaic)",
            "licence": "CC-BY 4.0",
        },
        "n_tiles": len(tiles),
        "n_failed": len(failed),
        "failed_tiles": [t for t, _ in failed[:50]],
        "layers": written,
        "training": "NEVER — vegetation layers are circular with the SDM target",
        "recommend": "python recommend_usa_30s.py --lat 30.2672 --lon -97.7431",
        "not": "data/satellite/*_10m.tif (18 km global POC)",
    }
    man_path = os.path.join(OUTPUT_DIR, "MANIFEST.json")
    with open(man_path, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nManifest: {man_path}")
    if failed:
        print(f"WARNING: {len(failed)} tiles failed (see MANIFEST.json)")
        for tile, err in failed[:8]:
            print(f"  {tile}: {err.splitlines()[-1][:160]}")

    print("\nNext:")
    print("  python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --html maps/austin.html")
    print("=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
