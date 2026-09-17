"""
Clean GBIF tree records and thin them onto the USA 1 km (30 arcsec) grid.

Does not touch data/gbif_500_species.csv or poc/xgboost_training.py (the global
10-arc-minute path).

Input  : data/species_occurrences/US/gbif_trees_US_raw.csv
         (taxonKey download for the curated US list, already vascular plants)
Output : data/species_occurrences/US/gbif_trees_US_30s_clean.csv
         data/species_occurrences/US/gbif_trees_US_30s_thinned.csv
         data/species_occurrences/US/gbif_trees_US_30s_flags.csv
         data/species_occurrences/US/gbif_trees_US_30s_cleaning_report.json

Jobs, in order:

1. SPECIES LIST. Keep names on the seed CSV (default) or the full US checklist.
   Growing the model means rerunning this with --species-list
   data/usa_tree_species_list.csv, then retraining.
2. DUPLICATES. Exact species + coordinate repeats.
3. COORDINATE ERRORS. Null island, lat==lon, outside the lower-48 envelope,
   magnet (herbarium/centroid) coordinates, predictor-nodata / ocean cells,
   extreme range outliers.
4. SPATIAL THINNING. One record per species per 30-arcsec cell (~1 km),
   keeping the newest year so a 2023 sighting beats an 1889 sheet in the
   same kilometre.

Taxonomy resolution against the GBIF API is skipped by default: the US raw
file was requested by taxonKey. Pass --taxonomy if you feed an untrusted CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from repo_paths import data

SCHEMA = ["species", "latitude", "longitude", "year", "country", "basis"]

COUNTRY_DATA = data("country_data", "USA")
GRID_JSON = os.path.join(COUNTRY_DATA, "country_grid.json")
TEMPLATE = os.path.join(COUNTRY_DATA, "climate_30s", "wc2.1_30s_bio_1.tif")
OCC_DIR = data("species_occurrences", "US")
RAW_CSV = os.path.join(OCC_DIR, "gbif_trees_US_raw.csv")
SEED_LIST = data("usa_tree_species_seed.csv")
FULL_LIST = data("usa_tree_species_list.csv")

NULL_ISLAND_DEG = 0.01
COUNTRY_ENVELOPE_MARGIN_DEG = 0.5
MAGNET_MIN_SPECIES = 8
MAGNET_MIN_RECORDS = 25
MAGNET_ABSURD_SPECIES = 30
CENTROID_MAX_DECIMALS = 2
OUTLIER_MIN_RECORDS = 20
OUTLIER_MAD_MULT = 8.0
# CONUS is ~4,500 km across; planted east-coast trees in California must not
# be treated as geocoding errors. This only catches wild misplots.
OUTLIER_FLOOR_KM = 3000.0
EARTH_RADIUS_KM = 6371.0
USABLE_THINNED_MIN = 150


def load_species_list(path):
    frame = pd.read_csv(path, comment="#", skipinitialspace=True)
    if "species" not in frame.columns:
        raise SystemExit(f"{path} has no 'species' column")
    names = (
        frame["species"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    if not names:
        raise SystemExit(f"{path} has no species names")
    return names


def read_grid(template_path=TEMPLATE, grid_json=GRID_JSON):
    if not os.path.isfile(template_path):
        raise SystemExit(f"USA 30s template missing: {template_path}")
    with rasterio.open(template_path) as src:
        grid = {
            "width": src.width,
            "height": src.height,
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
        }
    if os.path.isfile(grid_json):
        with open(grid_json) as handle:
            meta = json.load(handle)
        grid["bbox"] = tuple(meta["bbox"])
        grid["res_deg"] = float(meta["res_deg"])
    else:
        west, south, east, north = rasterio.transform.array_bounds(
            grid["height"], grid["width"], grid["transform"]
        )
        grid["bbox"] = (west, south, east, north)
        grid["res_deg"] = abs(float(grid["transform"].a))
    return grid


def grid_index(frame, grid):
    transform = grid["transform"]
    col = np.floor(
        (frame["longitude"].to_numpy(dtype=np.float64) - transform.c) / transform.a
    )
    row = np.floor(
        (frame["latitude"].to_numpy(dtype=np.float64) - transform.f) / transform.e
    )
    return row.astype(np.int32), col.astype(np.int32)


def flag_null_island(frame):
    return (frame["latitude"].abs() < NULL_ISLAND_DEG) & (
        frame["longitude"].abs() < NULL_ISLAND_DEG
    )


def flag_lat_equals_lon(frame):
    lat = frame["latitude"].abs()
    lon = frame["longitude"].abs()
    return np.isclose(lat, lon, atol=1e-6) & (lat > NULL_ISLAND_DEG)


def flag_outside_country(frame, bbox):
    west, south, east, north = bbox
    m = COUNTRY_ENVELOPE_MARGIN_DEG
    return (
        ~frame["latitude"].between(south - m, north + m)
        | ~frame["longitude"].between(west - m, east + m)
    )


def _decimals(values):
    remainder = np.rint(np.abs(values) * 1e6).astype(np.int64)
    decimals = np.full(len(values), 6, dtype=np.int8)
    for places in range(6):
        if (remainder % (10 ** (6 - places)) == 0).any():
            hit = (remainder % (10 ** (6 - places)) == 0) & (decimals == 6)
            decimals[hit] = places
    return decimals


def flag_magnet_coordinates(frame):
    lat = np.round(frame["latitude"].to_numpy(), 5)
    lon = np.round(frame["longitude"].to_numpy(), 5)
    coordinate = pd.Series(list(zip(lat, lon)), index=frame.index)
    stats = (
        pd.DataFrame({"coordinate": coordinate, "species": frame["species"].to_numpy()})
        .groupby("coordinate")
        .agg(n_records=("species", "size"), n_species=("species", "nunique"))
    )
    hits = stats[
        (stats["n_species"] >= MAGNET_MIN_SPECIES)
        & (stats["n_records"] >= MAGNET_MIN_RECORDS)
    ]
    if hits.empty:
        return pd.Series(False, index=frame.index), hits

    lat_dec = _decimals(np.array([c[0] for c in hits.index]))
    lon_dec = _decimals(np.array([c[1] for c in hits.index]))
    hits = hits.assign(
        centroid_like=(lat_dec <= CENTROID_MAX_DECIMALS)
        & (lon_dec <= CENTROID_MAX_DECIMALS)
    )
    address_like = hits[
        hits["centroid_like"] | (hits["n_species"] >= MAGNET_ABSURD_SPECIES)
    ]
    return coordinate.isin(set(address_like.index)), hits


def flag_range_outliers(frame):
    flagged = np.zeros(len(frame), dtype=bool)
    lat = np.deg2rad(frame["latitude"].to_numpy(dtype=np.float64))
    lon = np.deg2rad(frame["longitude"].to_numpy(dtype=np.float64))
    for _, index in frame.groupby("species", sort=False).indices.items():
        if len(index) < OUTLIER_MIN_RECORDS:
            continue
        plat, plon = lat[index], lon[index]
        centre_lat, centre_lon = np.median(plat), np.median(plon)
        a = (
            np.sin((plat - centre_lat) / 2) ** 2
            + np.cos(plat)
            * np.cos(centre_lat)
            * np.sin((plon - centre_lon) / 2) ** 2
        )
        distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        median = np.median(distance)
        sigma = np.median(np.abs(distance - median)) * 1.4826
        flagged[index] = distance > max(
            median + OUTLIER_MAD_MULT * sigma, OUTLIER_FLOOR_KM
        )
    return pd.Series(flagged, index=frame.index)


def flag_invalid_cells(frame, grid, template_path=TEMPLATE):
    """Drop points off the grid or in nodata (ocean / missing climate)."""
    with rasterio.open(template_path) as src:
        band = src.read(1)
        nodata = src.nodata
    row, col = grid_index(frame, grid)
    inside = (
        (row >= 0)
        & (row < band.shape[0])
        & (col >= 0)
        & (col < band.shape[1])
    )
    bad = np.ones(len(frame), dtype=bool)
    vals = band[row[inside], col[inside]].astype(np.float64)
    ok = np.isfinite(vals) & (np.abs(vals) < 1e30)
    if nodata is not None:
        ok &= vals != nodata
    bad[np.flatnonzero(inside)[ok]] = False
    return pd.Series(bad, index=frame.index)


def thin_to_grid(frame, grid):
    """One record per species per 1 km cell; keep the newest year."""
    row, col = grid_index(frame, grid)
    out = frame.assign(grid_row=row, grid_col=col)
    counts = (
        out.groupby(["species", "grid_row", "grid_col"], sort=False)
        .size()
        .rename("n_records")
    )
    order = out["year"].fillna(-1).to_numpy()
    out = out.iloc[np.argsort(-order, kind="stable")]
    out = out.drop_duplicates(["species", "grid_row", "grid_col"], keep="first")
    out = out.merge(counts, on=["species", "grid_row", "grid_col"], how="left")
    return (
        out[SCHEMA + ["grid_row", "grid_col", "n_records"]]
        .sort_values(["species", "grid_row", "grid_col"])
        .reset_index(drop=True)
    )


def summarise(frame, label):
    if not len(frame):
        return {"label": label, "records": 0, "species": 0}
    year = frame["year"]
    return {
        "label": label,
        "records": int(len(frame)),
        "species": int(frame["species"].nunique()),
        "year_median": None if year.isna().all() else int(year.median()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean and thin US tree GBIF records onto the 1 km grid."
    )
    parser.add_argument("--input", default=RAW_CSV)
    parser.add_argument(
        "--species-list",
        default=SEED_LIST,
        help=f"default {SEED_LIST}; pass {FULL_LIST} to grow the set",
    )
    parser.add_argument("--output-dir", default=OCC_DIR)
    parser.add_argument(
        "--all-species",
        action="store_true",
        help=f"ignore --species-list and keep every name in the input CSV",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(
            f"No input at {args.input}. Copy gbif_trees_US_raw.csv into {OCC_DIR}/"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, "gbif_trees_US_30s")
    clean_file = f"{prefix}_clean.csv"
    thinned_file = f"{prefix}_thinned.csv"
    flags_file = f"{prefix}_flags.csv"
    report_file = f"{prefix}_cleaning_report.json"

    print("=" * 62)
    print("CLEANING AND SPATIAL THINNING — USA 30 arcsec (~1 km)")
    print("=" * 62)

    frame = pd.read_csv(args.input)
    missing = [c for c in SCHEMA if c not in frame.columns]
    if missing:
        raise SystemExit(f"{args.input} is missing columns: {missing}")
    frame = frame[SCHEMA].copy()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")

    if args.all_species:
        allowed = None
        print("Species list: ALL names in the input file")
    else:
        allowed = set(load_species_list(args.species_list))
        print(f"Species list: {args.species_list}  ({len(allowed)} names)")

    grid = read_grid()
    print(
        f"Input:       {args.input}\n"
        f"             {len(frame):,} records, {frame['species'].nunique():,} species"
    )
    print(
        f"Target grid: {grid['width']}x{grid['height']}  "
        f"pixel {grid['res_deg']:.6f} deg (~1 km)  bbox {grid['bbox']}\n"
    )

    before = summarise(frame, "input")
    reason = pd.Series(pd.NA, index=frame.index, dtype="object")
    dropped = {}

    def apply_test(name, mask, note=""):
        mask = mask.fillna(False)
        fresh = mask & reason.isna()
        reason[fresh] = name
        print(
            f"  {name:32s} {int(mask.sum()):>8,} hit  "
            f"{int(fresh.sum()):>8,} dropped  {note}"
        )
        return int(fresh.sum())

    print("Tests:")
    if allowed is not None:
        dropped["not_on_species_list"] = apply_test(
            "not_on_species_list", ~frame["species"].isin(allowed)
        )
    else:
        print("  not_on_species_list               skipped (--all-species)")

    dropped["bad_coordinate"] = apply_test(
        "bad_coordinate",
        frame["latitude"].isna() | frame["longitude"].isna(),
    )
    dropped["duplicate_species_coordinate"] = apply_test(
        "duplicate_species_coordinate",
        frame.duplicated(["species", "latitude", "longitude"], keep="first"),
    )
    dropped["null_island"] = apply_test("null_island", flag_null_island(frame))
    dropped["lat_equals_lon"] = apply_test("lat_equals_lon", flag_lat_equals_lon(frame))
    dropped["outside_conus"] = apply_test(
        "outside_conus", flag_outside_country(frame, grid["bbox"])
    )

    magnet_mask, magnet_hits = flag_magnet_coordinates(frame)
    dropped["magnet_coordinate"] = apply_test(
        "magnet_coordinate",
        magnet_mask,
        f"{len(magnet_hits)} multi-species coordinates"
        if len(magnet_hits)
        else "",
    )
    dropped["invalid_cell"] = apply_test(
        "invalid_cell", flag_invalid_cells(frame, grid)
    )
    dropped["range_outlier"] = apply_test(
        "range_outlier", flag_range_outliers(frame)
    )

    rejected = frame[reason.notna()].assign(drop_reason=reason[reason.notna()])
    rejected.to_csv(flags_file, index=False)
    clean = frame[reason.isna()].reset_index(drop=True)
    clean.to_csv(clean_file, index=False)

    print(
        f"\nDropped {len(rejected):,} of {len(frame):,} records "
        f"({100 * len(rejected) / max(len(frame), 1):.2f}%)"
    )
    print(
        f"Clean:   {clean_file}  ({len(clean):,} records, "
        f"{clean['species'].nunique()} species)"
    )
    print(f"Audit:   {flags_file}")

    report = {
        "country": "USA",
        "grid": "30s",
        "input": args.input,
        "species_list": None if allowed is None else args.species_list,
        "dropped": dropped,
        "before": before,
        "after": summarise(clean, "clean"),
        "magnet_coordinates": int(len(magnet_hits)),
        "config": {
            "magnet_min_species": MAGNET_MIN_SPECIES,
            "outlier_floor_km": OUTLIER_FLOOR_KM,
            "usable_thinned_min": USABLE_THINNED_MIN,
        },
    }

    if len(clean):
        thinned = thin_to_grid(clean, grid)
        thinned.to_csv(thinned_file, index=False)
        per_species = thinned["species"].value_counts()
        usable = int((per_species >= USABLE_THINNED_MIN).sum())
        cells = thinned.groupby(["grid_row", "grid_col"], sort=False).ngroups
        report["thinned"] = {
            **summarise(thinned, "thinned"),
            "distinct_cells": int(cells),
            "species_usable": usable,
            "species_below_threshold": int(len(per_species) - usable),
            "records_per_species_median": float(per_species.median()),
        }
        print("\n" + "=" * 62)
        print("THINNED TO THE USA 1 km GRID")
        print("=" * 62)
        print(
            f"Records:            {len(thinned):,} of {len(clean):,} kept "
            f"({100 * len(thinned) / len(clean):.1f}%)"
        )
        print(f"Species:            {thinned['species'].nunique()}")
        print(f"Distinct cells:     {cells:,}")
        print(
            f"Records/species:    median {per_species.median():.0f}, "
            f"max {per_species.max()}"
        )
        print(f"Usable (>= {USABLE_THINNED_MIN} cells): {usable} species")
        print(f"Training table:     {thinned_file}")
        if len(per_species) - usable:
            print(f"\nBelow {USABLE_THINNED_MIN} thinned cells (trainer will skip):")
            for name, n in per_species[per_species < USABLE_THINNED_MIN].items():
                print(f"    {name:34s} {n:>4}")
    else:
        print("Nothing left to thin.")

    with open(report_file, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nReport: {report_file}")
    print("Next: python xgboost_training_usa_30s.py")


if __name__ == "__main__":
    main()
