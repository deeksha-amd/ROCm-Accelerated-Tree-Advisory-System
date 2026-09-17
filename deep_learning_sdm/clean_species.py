"""
Clean GBIF occurrence records and thin them onto the shared 18.5 km grid

Input  : a CSV with columns species, latitude, longitude, year, country, basis
         (species_occurrences/<CC>/gbif_trees_<CC>_raw.csv from
         download_species.py, or the old gbif_500_species.csv, which is worth
         running through this just to see the damage quantified)
Output : gbif_trees_<CC>_clean.csv   — same six columns, bad records removed
         gbif_trees_<CC>_thinned.csv — the training table: one record per
                                       species per grid cell
         cleaning_flags_<CC>.csv     — every rejected row with its reason, so
                                       each drop is auditable
         cleaning_report_<CC>.json   — the counts behind the printed summary

Four jobs, in order:

1. TAXONOMY. Every distinct species name is resolved through /species/match and
   anything that is not Plantae/Tracheophyta is dropped. download_species.py
   queries by taxonKey so this should find nothing; on gbif_500_species.csv it
   finds the warblers, gall wasps, weevil, fungus and bacterium. Belt and
   braces, and the braces have already failed once.

2. DUPLICATES. Exact species+coordinate repeats — 23.1% of gbif_500_species.csv.
   Usually the same herbarium sheet republished by several aggregators; left
   in, they silently weight those locations during training.

3. COORDINATE ERRORS. The tests the R package CoordinateCleaner runs. R is not
   available here and neither is pip, so what is practical is reimplemented in
   numpy and what is not is listed explicitly below.

4. SPATIAL THINNING. One record per species per 1/6-degree cell, on exactly the
   grid every raster in this repo uses. Two records 3 km apart carry one cell's
   worth of environmental information but two rows' worth of influence on the
   loss function, and in a presence-only model that is how road-side sampling
   bias turns into a fitted climate preference.

Because the pipeline is now scoped to a single country, two of these get
sharper: the ocean test and a country-envelope test can both be checked against
one known boundary instead of against every coastline on earth.

What CoordinateCleaner does that this does NOT, and why:

  * cc_cen  — country and province centroids. Needs the GeoNames-derived
              `countryref` table of ~10,000 centroid coordinates that ships
              inside the R package. The magnet test below catches many of them
              indirectly, since a centroid is a rounded coordinate shared by
              many unrelated species, but it is an approximation.
  * cc_inst — biodiversity-institution coordinates. Same problem: a reference
              table of ~10,000 museum and herbarium positions. Also partly
              caught by the magnet test.
  * cc_urb  — points inside urban areas. Needs a cities polygon layer, and is
              skipped on purpose as well: for trees an urban point is often a
              real street tree, and dropping them would bias the sample
              against exactly the warm-edge city records a planting advisory
              needs.
  * cc_sea  — implemented, using this repo's own land-fraction raster rather
              than a coastline polygon.
  * cc_outl — implemented, but with a coarser rule. See `flag_range_outliers`.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from download_species import COUNTRY, COUNTRY_BOUNDS, DEFAULT_BOUNDS, paths
from gbif_api import GbifClient, GbifRequestError

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"       # defines the target grid
LAND_RASTER = "satellite/landcover_land_frac_10m.tif"  # optional; for cc_sea

SCHEMA = ["species", "latitude", "longitude", "year", "country", "basis"]

# ── null island and its neighbourhood ──
# Exactly (0,0) is the signature of a parser that saw an empty string. A small
# radius is used because 0.0 and 0.00001 are the same bug.
NULL_ISLAND_DEG = 0.01

# ── country envelope ──
# Cheap and decisive now that everything is one country: a record filed under
# FR whose coordinate is in Poland is wrong, whatever else is true about it.
# The margin allows for overseas-adjacent coastal cells and for the bounding
# boxes in COUNTRY_BOUNDS being approximate.
COUNTRY_ENVELOPE_MARGIN_DEG = 0.5
APPLY_COUNTRY_ENVELOPE = True

# ── "magnet" coordinates: institutions and country centroids ──
# One coordinate shared by many unrelated species is not a place where all of
# them grow, it is an address: a herbarium's own lat/long stamped onto every
# sheet it holds, or a country centroid standing in for "somewhere in France".
#
# Tuned after a first pass flagged 1,180 coordinates and 21,074 records in
# France, none of them rounded. Those are not institutions: a botanical relevé
# legitimately lists every species in one plot at a single GPS fix, and full
# coordinate precision is the signature of a real survey, not of an address.
# Dropping them would delete exactly the well-surveyed sites. So a coordinate
# is only treated as an address if it is EITHER rounded like a centroid, or
# carries an implausible number of species for one plot.
MAGNET_MIN_SPECIES = 8        # distinct species at one identical coordinate
MAGNET_MIN_RECORDS = 25       # and at least this many records
MAGNET_ABSURD_SPECIES = 30    # too many for any real plot, rounded or not
CENTROID_MAX_DECIMALS = 2     # a centroid is always a rounded coordinate

# ── gridded / rounded coordinates ──
# Whole-degree coordinates are almost always a range description rather than a
# sighting. Flagged, not dropped: at 1/6 degree a whole-degree coordinate still
# lands in a defensible cell.
DROP_WHOLE_DEGREE = False

# ── ocean test ──
OCEAN_MIN_LAND_PERCENT = 1.0  # landcover_land_frac is a percentage, 0-100

# ── per-species range outliers ──
OUTLIER_MIN_RECORDS = 20      # below this the robust statistics are noise
OUTLIER_MAD_MULT = 6.0        # distance > median + 6 robust sigma
OUTLIER_FLOOR_KM = 400.0      # within one country a tighter floor than the
                              # 2,500 km a global run would need
EARTH_RADIUS_KM = 6371.0

# Species with fewer thinned records than this cannot support spatial block
# cross-validation; xgboost_training.py already skips them at 30.
USABLE_THINNED_MIN = 30

# sdm_features.py looks for the training table at a fixed, country-free path
# (its OCCURRENCE_CANDIDATES list). The country-scoped file stays the master
# copy so several countries can coexist; this is the one the trainer reads.
CANONICAL_THINNED = "species_occurrences/gbif_trees_thinned.csv"
CANONICAL_CLEAN = "species_occurrences/gbif_trees_clean.csv"


# ─────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────
def read_template(path):
    with rasterio.open(path) as src:
        return {"width": src.width, "height": src.height,
                "transform": src.transform, "crs": src.crs}


def grid_index(frame, grid):
    """Row/col of every record on the target grid, as int32.

    Computed straight from the affine transform rather than calling
    rasterio.transform.rowcol in a Python loop, which is what
    xgboost_training.extract_climate does and which takes minutes.
    """
    transform = grid["transform"]
    col = np.floor((frame["longitude"].to_numpy() - transform.c) / transform.a)
    row = np.floor((frame["latitude"].to_numpy() - transform.f) / transform.e)
    return row.astype("int32"), col.astype("int32")


# ─────────────────────────────────────────────────────
# 1. TAXONOMY
# ─────────────────────────────────────────────────────
def resolve_taxonomy(names, client, cache_path):
    """name -> {kingdom, phylum, family, ...} via /species/match, cached.

    One request per distinct name, so ~150 for a France run and ~300 for the
    old file. The cache makes a re-run free.
    """
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as handle:
            cache = json.load(handle)

    missing = [n for n in names if n not in cache]
    resolved_any = False
    for name in tqdm(missing, desc="Resolving taxonomy", unit="name",
                     disable=not missing):
        try:
            payload = client.match(name)
        except GbifRequestError:
            payload = None
        # A failed or empty match is NOT cached. Caching it once cost Salix
        # alba its 4,000 records: a single transient miss was stored as
        # kingdom=None and every later run read that back as "not a plant".
        if not payload or not payload.get("kingdom"):
            continue
        resolved_any = True
        cache[name] = {k: payload.get(k) for k in
                       ("usageKey", "kingdom", "phylum", "class", "order",
                        "family", "rank", "matchType")}
    if resolved_any:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)

    return cache


def flag_non_plants(frame, taxonomy):
    """Drop records whose species is known NOT to be a vascular plant.

    A name that failed to resolve is kept, not dropped. Records reach here
    from a taxonKey query, so they are plants by construction; an unresolved
    name means the API call failed, and deleting data on the strength of a
    failed request is the wrong default. Unresolved names are counted and
    printed instead.
    """
    kingdom = frame["species"].map(lambda n: (taxonomy.get(n) or {}).get("kingdom"))
    phylum = frame["species"].map(lambda n: (taxonomy.get(n) or {}).get("phylum"))
    known = kingdom.notna()
    return known & ~((kingdom == "Plantae") & (phylum == "Tracheophyta"))


# ─────────────────────────────────────────────────────
# 3. COORDINATE ERROR TESTS
# ─────────────────────────────────────────────────────
def flag_null_island(frame):
    return ((frame["latitude"].abs() < NULL_ISLAND_DEG) &
            (frame["longitude"].abs() < NULL_ISLAND_DEG))


def flag_lat_equals_lon(frame):
    """|lat| == |lon|: the signature of a transposed or copy-pasted pair.

    Excludes the 0,0 case, which null island already owns. Inherently a little
    lossy, since real points do sit on the diagonal; at France's longitudes it
    can only fire between 41 and 51 degrees east, which is outside the country
    entirely, so here it costs nothing.
    """
    lat = frame["latitude"].abs()
    lon = frame["longitude"].abs()
    return np.isclose(lat, lon, atol=1e-6) & (lat > NULL_ISLAND_DEG)


def flag_outside_country(frame, country):
    """Coordinates outside the country's bounding box.

    A bounding box is not a border — it keeps Corsica and loses nothing, but it
    also keeps a strip of Spain, Germany and the Channel. That is the right
    trade for a defect test: it only ever fires on records that are
    unambiguously misplaced, and the ocean test cleans up the sea strip.
    """
    bounds = COUNTRY_BOUNDS.get(country.upper())
    if not bounds or not APPLY_COUNTRY_ENVELOPE:
        return pd.Series(False, index=frame.index), None

    lat_min, lat_max, lon_min, lon_max = bounds
    margin = COUNTRY_ENVELOPE_MARGIN_DEG
    outside = (~frame["latitude"].between(lat_min - margin, lat_max + margin) |
               ~frame["longitude"].between(lon_min - margin, lon_max + margin))
    return outside, bounds


def _decimals(values):
    """Number of decimal places in each coordinate, capped at 6."""
    remainder = np.rint(np.abs(values) * 1e6).astype("int64")
    decimals = np.full(len(values), 6, dtype="int8")
    for places in range(6):
        if (remainder % (10 ** (6 - places)) == 0).any():
            hit = (remainder % (10 ** (6 - places)) == 0) & (decimals == 6)
            decimals[hit] = places
    return decimals


def flag_whole_degree(frame):
    return ((_decimals(frame["latitude"].to_numpy()) == 0) &
            (_decimals(frame["longitude"].to_numpy()) == 0))


def flag_magnet_coordinates(frame):
    """Coordinates that host implausibly many unrelated species.

    Stands in for CoordinateCleaner's cc_inst and part of cc_cen without their
    reference tables. A coordinate carrying MAGNET_MIN_SPECIES distinct species
    and MAGNET_MIN_RECORDS records is treated as an address rather than a
    habitat; if it is also rounded to CENTROID_MAX_DECIMALS or fewer it is
    reported separately as centroid-like.
    """
    lat = np.round(frame["latitude"].to_numpy(), 5)
    lon = np.round(frame["longitude"].to_numpy(), 5)
    coordinate = pd.Series(list(zip(lat, lon)), index=frame.index)

    stats = pd.DataFrame({"coordinate": coordinate,
                          "species": frame["species"].to_numpy()}) \
        .groupby("coordinate").agg(n_records=("species", "size"),
                                   n_species=("species", "nunique"))
    hits = stats[(stats["n_species"] >= MAGNET_MIN_SPECIES) &
                 (stats["n_records"] >= MAGNET_MIN_RECORDS)]
    if hits.empty:
        return pd.Series(False, index=frame.index), hits

    lat_dec = _decimals(np.array([c[0] for c in hits.index]))
    lon_dec = _decimals(np.array([c[1] for c in hits.index]))
    hits = hits.assign(centroid_like=(lat_dec <= CENTROID_MAX_DECIMALS) &
                                     (lon_dec <= CENTROID_MAX_DECIMALS))

    address_like = hits[hits["centroid_like"] |
                        (hits["n_species"] >= MAGNET_ABSURD_SPECIES)]
    return coordinate.isin(set(address_like.index)), hits


def flag_range_outliers(frame):
    """Per-species geographic outliers, robustly.

    CoordinateCleaner's cc_outl compares every point to the distance
    distribution of all other points of that species, which is O(n^2) per
    species and wants a spatial index; scipy is not installed. Instead each
    point is compared to its species' geographic median and flagged when it
    lies beyond the median distance plus OUTLIER_MAD_MULT robust standard
    deviations, and beyond OUTLIER_FLOOR_KM regardless.

    Blunter than cc_outl — it cannot split a genuinely disjunct range into two
    clusters — but it reliably catches the single wildly misplaced record,
    which is the error that actually occurs. DATA_OVERVIEW.md records the
    example: the old file's only three records below 65 degrees South are all
    a Himalayan pine tagged in Antarctica.
    """
    flagged = np.zeros(len(frame), dtype=bool)
    lat = np.deg2rad(frame["latitude"].to_numpy())
    lon = np.deg2rad(frame["longitude"].to_numpy())

    for _, index in frame.groupby("species", sort=False).indices.items():
        if len(index) < OUTLIER_MIN_RECORDS:
            continue
        plat, plon = lat[index], lon[index]
        centre_lat, centre_lon = np.median(plat), np.median(plon)

        # Haversine great-circle distance to the species' median point.
        a = (np.sin((plat - centre_lat) / 2) ** 2 +
             np.cos(plat) * np.cos(centre_lat) * np.sin((plon - centre_lon) / 2) ** 2)
        distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        median = np.median(distance)
        sigma = np.median(np.abs(distance - median)) * 1.4826
        flagged[index] = distance > max(median + OUTLIER_MAD_MULT * sigma,
                                        OUTLIER_FLOOR_KM)

    return pd.Series(flagged, index=frame.index)


def flag_ocean(frame, grid, land_raster=LAND_RASTER):
    """Points whose grid cell is sea or has no land-cover data.

    Uses this repo's own landcover_land_frac_10m.tif, already on the target
    grid, instead of a coastline polygon. If the satellite layers have not been
    built the test is skipped rather than guessed.
    """
    if not os.path.exists(land_raster):
        return pd.Series(False, index=frame.index), False

    with rasterio.open(land_raster) as src:
        land = src.read(1)

    row, col = grid_index(frame, grid)
    inside = ((row >= 0) & (row < land.shape[0]) &
              (col >= 0) & (col < land.shape[1]))
    fraction = np.full(len(frame), np.nan, dtype="float32")
    fraction[inside] = land[row[inside], col[inside]]
    return pd.Series(~(fraction >= OCEAN_MIN_LAND_PERCENT),
                     index=frame.index), True


# ─────────────────────────────────────────────────────
# 4. SPATIAL THINNING
# ─────────────────────────────────────────────────────
def thin_to_grid(frame, grid):
    """One record per species per grid cell.

    Keeps the most recent record in each cell, because the climate rasters we
    train against are the 2015-2024 window (DATA_OVERVIEW.md), so a 2023
    sighting is better matched to its predictors than an 1889 herbarium sheet
    from the same valley. `n_records` keeps how many collapsed, which is the
    per-cell sampling effort for that species and is worth having downstream.
    """
    row, col = grid_index(frame, grid)
    out = frame.assign(grid_row=row, grid_col=col)

    counts = out.groupby(["species", "grid_row", "grid_col"], sort=False) \
                .size().rename("n_records")

    order = out["year"].fillna(-1).to_numpy()          # newest first
    out = out.iloc[np.argsort(-order, kind="stable")]
    out = out.drop_duplicates(["species", "grid_row", "grid_col"], keep="first")

    out = out.merge(counts, on=["species", "grid_row", "grid_col"], how="left")
    return out[SCHEMA + ["grid_row", "grid_col", "n_records"]] \
        .sort_values(["species", "grid_row", "grid_col"]) \
        .reset_index(drop=True)


# ─────────────────────────────────────────────────────
def summarise(frame, label):
    if not len(frame):
        return {"label": label, "records": 0}
    return {
        "label": label,
        "records": int(len(frame)),
        "species": int(frame["species"].nunique()),
        "year_median": (None if frame["year"].isna().all()
                        else int(frame["year"].median())),
        "top5_species_pct": round(float(
            frame["species"].value_counts(normalize=True).head(5).sum() * 100), 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clean and spatially thin GBIF occurrence records.")
    parser.add_argument("--country", default=COUNTRY,
                        help=f"ISO country code (default {COUNTRY})")
    parser.add_argument("--input", default=None,
                        help="input CSV; defaults to this country's raw file")
    parser.add_argument("--output-prefix", default=None,
                        help="prefix for outputs, for auditing a foreign file")
    parser.add_argument("--skip-taxonomy", action="store_true",
                        help="do not call GBIF; keeps every species name")
    parser.add_argument("--drop-whole-degree", action="store_true",
                        help="also drop whole-degree coordinates "
                             "(flagged but kept by default)")
    parser.add_argument("--no-thin", action="store_true",
                        help="write the clean file but skip thinning")
    args = parser.parse_args()

    country = args.country.upper()
    out = paths(country)
    source = args.input or out["raw_csv"]
    if not os.path.exists(source):
        raise SystemExit(f"No input at {source}. "
                         f"Run: python3 download_species.py --download")

    prefix = args.output_prefix or os.path.join(out["dir"], f"gbif_trees_{country}")
    clean_file = f"{prefix}_clean.csv"
    thinned_file = f"{prefix}_thinned.csv"
    flags_file = f"{prefix}_flags.csv"
    report_file = f"{prefix}_cleaning_report.json"
    cache_file = os.path.join(out["dir"], "taxonomy_cache.json")
    os.makedirs(out["dir"], exist_ok=True)

    print("=" * 62)
    print(f"CLEANING AND SPATIAL THINNING — {country}")
    print("=" * 62)
    frame = pd.read_csv(source)
    missing = [c for c in SCHEMA if c not in frame.columns]
    if missing:
        raise SystemExit(f"{source} is missing columns: {missing}")
    frame = frame[SCHEMA].copy()

    grid = read_template(TEMPLATE)
    print(f"Input:       {source}")
    print(f"             {len(frame):,} records, "
          f"{frame['species'].nunique():,} species")
    print(f"Target grid: {grid['width']}x{grid['height']} {grid['crs']} "
          f"pixel {abs(grid['transform'].a):.10f} deg (~18.5 km)\n")

    before = summarise(frame, "input")
    reason = pd.Series(pd.NA, index=frame.index, dtype="object")

    def apply_test(name, mask, note=""):
        """Attribute each row to the first test that rejects it."""
        fresh = mask.fillna(False) & reason.isna()
        reason[fresh] = name
        print(f"  {name:30s} {int(mask.sum()):>8,} hit  "
              f"{int(fresh.sum()):>8,} dropped  {note}")
        return int(fresh.sum())

    print("Tests:")
    dropped = {}

    if args.skip_taxonomy:
        print("  taxonomy check skipped (--skip-taxonomy)")
    else:
        client = GbifClient()
        taxonomy = resolve_taxonomy(sorted(frame["species"].dropna().unique()),
                                    client, cache_file)
        mask = flag_non_plants(frame, taxonomy)
        dropped["not_vascular_plant"] = apply_test("not_vascular_plant", mask)
        for name, n in frame.loc[mask, "species"].value_counts().head(6).items():
            info = taxonomy.get(name) or {}
            print(f"      {name:28s} {n:>7,}  {info.get('kingdom')}/"
                  f"{info.get('phylum')}/{info.get('family')}")

    dropped["duplicate_species_coordinate"] = apply_test(
        "duplicate_species_coordinate",
        frame.duplicated(["species", "latitude", "longitude"], keep="first"))

    dropped["null_island"] = apply_test("null_island", flag_null_island(frame))
    dropped["lat_equals_lon"] = apply_test("lat_equals_lon",
                                           flag_lat_equals_lon(frame))

    outside, bounds = flag_outside_country(frame, country)
    dropped["outside_country"] = apply_test(
        "outside_country", outside,
        f"bbox {bounds} +{COUNTRY_ENVELOPE_MARGIN_DEG} deg" if bounds
        else f"SKIPPED — no bounds for {country}")

    magnet_mask, magnet_hits = flag_magnet_coordinates(frame)
    dropped["magnet_coordinate"] = apply_test(
        "magnet_coordinate", magnet_mask,
        f"{len(magnet_hits)} multi-species coordinates, "
        f"{int(magnet_hits['centroid_like'].sum())} centroid-like; "
        f"relevé plots kept"
        if len(magnet_hits) else "")

    ocean_mask, ocean_ran = flag_ocean(frame, grid)
    dropped["ocean"] = apply_test(
        "ocean", ocean_mask,
        "" if ocean_ran else f"SKIPPED — {LAND_RASTER} not built")

    dropped["range_outlier"] = apply_test("range_outlier",
                                          flag_range_outliers(frame))

    whole_degree = flag_whole_degree(frame)
    if args.drop_whole_degree or DROP_WHOLE_DEGREE:
        dropped["whole_degree"] = apply_test("whole_degree", whole_degree)
    else:
        print(f"  {'whole_degree':30s} {int(whole_degree.sum()):>8,} hit  "
              f"{0:>8,} dropped  flagged only")

    rejected = frame[reason.notna()].assign(drop_reason=reason[reason.notna()])
    rejected.to_csv(flags_file, index=False)
    clean = frame[reason.isna()].reset_index(drop=True)
    clean.to_csv(clean_file, index=False)
    if not args.output_prefix:
        clean.to_csv(CANONICAL_CLEAN, index=False)

    print(f"\nDropped {len(rejected):,} of {len(frame):,} records "
          f"({100 * len(rejected) / max(len(frame), 1):.2f}%)")
    print(f"Clean:   {clean_file}  ({len(clean):,} records, "
          f"{clean['species'].nunique()} species)")
    print(f"Audit:   {flags_file}")

    report = {"country": country, "input": source, "dropped": dropped,
              "before": before, "after": summarise(clean, "clean"),
              "magnet_coordinates": int(len(magnet_hits)),
              "whole_degree_flagged": int(whole_degree.sum()),
              "ocean_test_ran": bool(ocean_ran),
              "config": {"magnet_min_species": MAGNET_MIN_SPECIES,
                         "magnet_min_records": MAGNET_MIN_RECORDS,
                         "outlier_mad_mult": OUTLIER_MAD_MULT,
                         "outlier_floor_km": OUTLIER_FLOOR_KM,
                         "country_envelope_margin_deg":
                             COUNTRY_ENVELOPE_MARGIN_DEG}}

    if not args.no_thin and len(clean):
        thinned = thin_to_grid(clean, grid)
        thinned.to_csv(thinned_file, index=False)
        if not args.output_prefix:
            thinned.to_csv(CANONICAL_THINNED, index=False)

        cells = thinned.groupby(["grid_row", "grid_col"], sort=False).ngroups
        per_species = thinned["species"].value_counts()
        usable = int((per_species >= USABLE_THINNED_MIN).sum())
        report["thinned"] = {
            **summarise(thinned, "thinned"),
            "distinct_cells": int(cells),
            "species_usable": usable,
            "species_below_threshold": int(len(per_species) - usable),
            "records_per_species_median": int(per_species.median()),
            "records_per_species_p10": int(per_species.quantile(0.10)),
            "records_per_species_p90": int(per_species.quantile(0.90)),
            "records_per_species_max": int(per_species.max()),
        }

        print("\n" + "=" * 62)
        print("THINNED TO THE 18.5 km GRID")
        print("=" * 62)
        print(f"Records:            {len(thinned):,} of {len(clean):,} kept "
              f"({100 * len(thinned) / len(clean):.1f}%)")
        print(f"Species:            {thinned['species'].nunique()}")
        print(f"Distinct cells:     {cells:,}")
        print(f"Records/species:    median {per_species.median():.0f}, "
              f"p10 {per_species.quantile(0.10):.0f}, "
              f"p90 {per_species.quantile(0.90):.0f}, "
              f"max {per_species.max()}")
        print(f"Usable (>= {USABLE_THINNED_MIN}):      {usable} species")
        print(f"Below threshold:    {len(per_species) - usable} species")
        print(f"Training table:     {thinned_file}")
        if not args.output_prefix:
            print(f"                    {CANONICAL_THINNED} "
                  f"(what sdm_features.py reads)")

        if len(per_species) - usable:
            print(f"\nSpecies below {USABLE_THINNED_MIN} thinned records:")
            for name, n in per_species[per_species < USABLE_THINNED_MIN].items():
                print(f"    {name:34s} {n:>4}")

    with open(report_file, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nReport: {report_file}")
    print(f"Next: python3 sampling_effort.py --surface --background")


if __name__ == "__main__":
    main()
