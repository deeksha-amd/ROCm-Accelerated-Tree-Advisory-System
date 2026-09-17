"""
Download occurrence records for a curated list of plantable trees, one country

Replaces the original script that produced gbif_500_species.csv. That file is
unusable and is kept only because it cannot currently be regenerated:

  * It searched with `name_lookup(q=genus)` — Elasticsearch FULL-TEXT search,
    not taxonomy — so 9.5% of its 392,725 records are not trees. Searching
    "Pinus" returned the warbler `Dendroica pinus` (3,000 records); "Betula" a
    weevil (`Betulapion`, 2,992) and a fungus (`Boletellus betula`);
    "Quercus" gall wasps (`Andricus`, `Cynips`) and the bacterium
    `Paenibacillus quercus`; "Acer" the herb `Erigeron acer` (2,999). The very
    first row of the file is `Quercusia quercus`, a butterfly.
  * Only 28 of its 39 configured genera appear at all, and five of them are
    90.5% of the records.

This version fixes the taxonomy structurally: every target is resolved through
`/species/match` to a verified vascular-plant backbone key, and occurrences are
requested by `taxonKey`, which GBIF evaluates against each record's full
classification chain. Only descendants of that plant taxon can come back, so a
warbler is excluded by the shape of the query rather than by a filter someone
has to remember to apply. See `gbif_api.match_genus` and `match_species`.

WHAT THIS IS FOR, AND WHY IT IS NOT A BIODIVERSITY ATLAS
--------------------------------------------------------
This feeds a user-facing planting advisory: someone asks "what should I plant
here?" and expects oak, lime, walnut, Douglas fir — trees they recognise and
can buy as saplings. So the target is a curated list of roughly 100-300
well-known plantable species held in `tree_species_list.csv`, an editable CSV
rather than a list buried in code, because which species to recommend is a
product decision that will change.

INCLUSION RULE (the list is auditable, so state it plainly)
-----------------------------------------------------------
A species enters the download when ALL of these hold:

  1. It appears in tree_species_list.csv. That file was assembled by hand from
     the common native and naturalised trees of temperate Europe plus the
     widely planted timber, fruit, nut, agroforestry, street and park trees of
     each major region, so the same file still works when COUNTRY changes.
  2. /species/match resolves it to an accepted Plantae/Tracheophyta species.
  3. It has at least MIN_RECORDS_PER_SPECIES usable records in COUNTRY.

Ranking the country's tree species by record count and taking the top N was
the other candidate rule. It is not used as the primary rule because in France
it promotes hedgerow shrubs — Crataegus monogyna, Prunus spinosa, Cornus
sanguinea, Sambucus nigra are among the most-recorded woody plants in the
country and nobody asks an advisory tool whether to plant blackthorn. It is
still computed and printed by `--suggest`, so anything genuinely missing from
the curated list is visible and can be added to the CSV.

Modes:

    python3 download_species.py --countries   # which country to use, measured
    python3 download_species.py --plan        # resolve + size, no records
    python3 download_species.py --suggest     # top species NOT in the CSV
    python3 download_species.py --download    # the real job, resumable
"""

import argparse
import csv
import json
import math
import os
import time

import pandas as pd
from tqdm import tqdm

import gbif_api
from gbif_api import (BASIS_OF_RECORD, OCCURRENCE_FILTERS, PAGE_SIZE,
                      GbifClient, GbifRequestError, match_genus, match_species)

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
# The whole pipeline is country-scoped. Change this one value and rerun to
# move to another country; nothing else needs editing.
COUNTRY = "FR"

# The curated lists are code-adjacent: they ship inside deep_learning_sdm/ and
# are anchored to this module's directory. OUTPUT_DIR is data and stays relative
# to the working directory, which is the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))

SPECIES_LIST_FILE = os.path.join(HERE, "tree_species_list.csv")
OUTPUT_DIR = "species_occurrences"

MIN_RECORDS_PER_SPECIES = 60      # below this, spatial block CV cannot work
MAX_RECORDS_PER_SPECIES = 4000    # the main size dial

# Hard ceiling on requests for one species, and the shortfall worth chasing.
#
# Without these, a species whose strata run dry just short of its budget sends
# the top-up pass paging ever deeper looking for a handful of records it will
# never find, at which point one species can consume thousands of requests and
# stall the run. Observed: species taking 48 s each while the API itself was
# answering in 0.2 s. The last 5% of a budget is not worth any of that.
MAX_REQUESTS_PER_SPECIES = 60
TOPUP_MIN_SHORTFALL = 0.05        # skip the top-up below this fraction short

# Records coarser than a grid cell cannot say which cell the tree was in. Most
# herbarium sheets record no uncertainty at all and dropping those would delete
# the historical record, so unknown is accepted.
MAX_COORDINATE_UNCERTAINTY_M = 10_000
ACCEPT_UNKNOWN_UNCERTAINTY = True

# ── SAMPLING STRATA ──
# Measured on GBIF, and the reason this is not a simple paged loop:
#
#   * The default page order front-loads the newest records hard. Quercus robur
#     in France has 322,231 usable records of which only 1,794 are 2023-2026,
#     yet the first 300 returned are all 2025-2026 from four datasets. A naive
#     "take the first 4,000" gets a pure recent-citizen-science sample, which
#     is the most spatially biased slice there is — app users walk paths and
#     parks. A 3-species smoke test of the earlier unstratified version came
#     back 100% HUMAN_OBSERVATION, years 2025-2026, zero herbarium specimens.
#   * Deep offsets are unusable. offset=0 answers in 0.9 s; offset=30,000 did
#     not answer within 60 s. Filters, by contrast, are all ~0.15 s. So the fix
#     is to cut the query into strata small enough to page shallowly, never to
#     page deep into one big query.
#
# Year bands break the recency front-loading and, because old records are
# herbarium specimens and new ones are observations, they also spread the
# basis-of-record mix. Latitude bands add geographic spread.
YEAR_BANDS = [(1700, 1979), (1980, 1999), (2000, 2009),
              (2010, 2016), (2017, 2021), (2022, 2026)]
LAT_BANDS = 2

# Approximate bounding boxes, only used to place the latitude band edges, so
# they do not need to be exact. (lat_min, lat_max, lon_min, lon_max)
COUNTRY_BOUNDS = {
    # Metropolitan France only, Corsica included. GBIF files the overseas
    # departments under their own ISO codes (GF, RE, MQ, GP, YT, NC, PM, WF,
    # PF, TF, BL, MF), so country=FR mostly excludes them already; this box is
    # what clean_species.py enforces, so a French Guiana record mislabelled FR
    # cannot put Amazonian rainforest in a temperate-Europe dataset.
    "FR": (41.3, 51.2, -5.2, 9.6),
    "GB": (49.9, 60.9, -8.2, 1.8),
    "DE": (47.2, 55.1, 5.8, 15.1),
    "ES": (35.9, 43.8, -9.4, 4.4),
    "SE": (55.3, 69.1, 11.1, 24.2),
    "US": (24.5, 49.4, -125.0, -66.9),
    "AU": (-43.7, -10.0, 112.9, 153.7),
    "NL": (50.7, 53.6, 3.3, 7.3),
    "IT": (36.6, 47.1, 6.6, 18.5),
    "CA": (41.7, 70.0, -141.0, -52.6),
}
DEFAULT_BOUNDS = (-56.0, 72.0, -180.0, 180.0)

# Countries compared by --countries. Chosen as the plausible demo candidates:
# the big vascular-plant recorders plus a few high-diversity controls.
CANDIDATE_COUNTRIES = ["FR", "GB", "US", "ES", "AU", "DE", "SE", "NL",
                       "IT", "CA", "BE", "DK", "CH", "PL", "BR", "MX",
                       "ZA", "IN", "CN", "JP"]

# GBIF answers 400 if the query string carries too many repeated taxonKey
# values; 60 per request is comfortably under the limit.
TAXON_KEY_BATCH = 60

# Genera used only by --suggest, to find well-known trees missing from the CSV.
SUGGEST_GENERA = [
    "Quercus", "Acer", "Fagus", "Betula", "Populus", "Salix", "Fraxinus",
    "Ulmus", "Tilia", "Carpinus", "Alnus", "Castanea", "Juglans", "Platanus",
    "Aesculus", "Prunus", "Sorbus", "Malus", "Pyrus", "Corylus", "Celtis",
    "Pinus", "Picea", "Abies", "Larix", "Pseudotsuga", "Juniperus", "Cedrus",
    "Cupressus", "Taxus", "Robinia", "Olea", "Ficus", "Eucalyptus", "Acacia",
    "Magnolia", "Liquidambar", "Liriodendron", "Catalpa", "Morus", "Ilex",
]

# Derived paths, all carrying the country so two runs never collide.
def paths(country=None):
    country = (country or COUNTRY).upper()
    base = os.path.join(OUTPUT_DIR, country)
    return {
        "dir": base,
        "raw_dir": os.path.join(base, "raw"),
        "plan": os.path.join(base, "species_plan.json"),
        "raw_csv": os.path.join(base, f"gbif_trees_{country}_raw.csv"),
        "log": os.path.join(base, f"download_log_{country}.csv"),
        "countries": os.path.join(OUTPUT_DIR, "country_comparison.csv"),
        "suggest": os.path.join(OUTPUT_DIR, f"suggested_species_{country}.csv"),
    }


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────
def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def human_time(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def atomic_write_csv(frame, path):
    """Write then rename, so an interrupted run never leaves a half CSV that
    the resume logic would mistake for a finished species."""
    tmp = path + ".part"
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def species_list_for(country):
    """The species list a country should use.

    Which trees to recommend is a product decision, not an ecological one, and
    it does not travel: the France list carries hornbeam and sessile oak, and a
    US advisory instead has to answer for red maple, flowering dogwood and
    loblolly pine. So a country may carry its own plain CSV, named
    tree_species_list_<ISO2>.csv, and falls back to the shared list when it
    does not. France has no per-country file, so its behaviour is unchanged.
    """
    candidate = os.path.join(HERE, f"tree_species_list_{country.upper()}.csv")
    return candidate if os.path.exists(candidate) else SPECIES_LIST_FILE


# Set by main() from --species-list or species_list_for(). Module-level so the
# plan, the merge and the manifest all report the list that was actually read.
ACTIVE_SPECIES_LIST = SPECIES_LIST_FILE


def load_species_list(path=None):
    """Read the curated list. Blank lines and '#' comments are ignored."""
    path = path or ACTIVE_SPECIES_LIST
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}")
    entries = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("species") or "").strip()
            if not name or name.startswith("#"):
                continue
            entries.append({"species": name,
                            "common_name": (row.get("common_name") or "").strip(),
                            "use": (row.get("use") or "").strip(),
                            "region": (row.get("region") or "").strip(),
                            "notes": (row.get("notes") or "").strip()})
    return entries


def usable_record(record):
    lat = record.get("decimalLatitude")
    lon = record.get("decimalLongitude")
    if lat is None or lon is None:
        return False
    uncertainty = record.get("coordinateUncertaintyInMeters")
    if uncertainty is None:
        return ACCEPT_UNKNOWN_UNCERTAINTY
    return uncertainty <= MAX_COORDINATE_UNCERTAINTY_M


def build_strata(country):
    """The (year band, latitude band) grid this country is sampled over."""
    lat_min, lat_max, _, _ = COUNTRY_BOUNDS.get(country.upper(), DEFAULT_BOUNDS)
    edges = [lat_min + (lat_max - lat_min) * i / LAT_BANDS
             for i in range(LAT_BANDS + 1)]
    strata = []
    for year_lo, year_hi in YEAR_BANDS:
        for i in range(LAT_BANDS):
            strata.append({"year": f"{year_lo},{year_hi}",
                           "decimalLatitude": f"{edges[i]:.4f},{edges[i + 1]:.4f}"})
    return strata


# ─────────────────────────────────────────────────────
# STAGE 0: WHICH COUNTRY?
# ─────────────────────────────────────────────────────
def compare_countries(client, entries, countries=None):
    """Rank candidate countries by data available for the CURATED species.

    Not by total tree records and not by species richness. The product is a
    planting advisory for well-known trees, so the question is how many of
    those species the country can actually support a model for, and how well
    sampled they are. A country with five million records of forty species is
    a worse demo than one with one million records of two hundred — but a
    hyper-diverse tropical country whose records are thin per species is worse
    still, because every model there trains on scraps.
    """
    countries = countries or CANDIDATE_COUNTRIES
    resolved, failed = resolve_species(client, entries, country=None,
                                       with_counts=False)
    keys = [item["key"] for item in resolved]
    print(f"\nCurated list: {len(entries)} names, {len(resolved)} resolved to "
          f"plant species keys, {len(failed)} unresolved")

    rows = []
    for country in tqdm(countries, desc="Countries", unit="cc"):
        per_species = {}
        # taxonKey can be repeated for OR semantics, but 263 of them overflow
        # the URL and GBIF answers 400, so they go in batches. Each species
        # sits in exactly one batch, so merging the facets cannot double-count.
        for batch in range(0, len(keys), TAXON_KEY_BATCH):
            params = [("country", country), ("limit", 0),
                      ("facet", "speciesKey"), ("facetLimit", 1200)]
            params += [("taxonKey", k)
                       for k in keys[batch:batch + TAXON_KEY_BATCH]]
            params += [(k, v) for k, v in OCCURRENCE_FILTERS.items()]
            params += [("basisOfRecord", b) for b in BASIS_OF_RECORD]
            payload = client.get(f"{gbif_api.V1}/occurrence/search", params)
            for facet in (payload.get("facets") or []):
                for entry in facet["counts"]:
                    per_species[entry["name"]] = int(entry["count"])

        counts = sorted(per_species.values(), reverse=True)
        usable = [n for n in counts if n >= MIN_RECORDS_PER_SPECIES]
        rows.append({
            "country": country,
            "curated_records": sum(counts),
            "curated_species_present": len(counts),
            "curated_species_usable": len(usable),
            "median_records_per_usable": int(pd.Series(usable).median()) if usable else 0,
            "records_if_capped": sum(min(n, MAX_RECORDS_PER_SPECIES) for n in usable),
            "top5_share_pct": round(100 * sum(counts[:5]) / max(sum(counts), 1), 1),
        })

    frame = pd.DataFrame(rows).sort_values("curated_species_usable",
                                           ascending=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frame.to_csv(paths()["countries"], index=False)

    print("\n" + "=" * 96)
    print("COUNTRY COMPARISON — curated plantable species only")
    print("=" * 96)
    print(f"{'cc':4s}{'curated records':>17s}{'species':>9s}{'usable':>8s}"
          f"{'median recs':>13s}{'to download':>13s}{'top5%':>8s}")
    for _, row in frame.iterrows():
        print(f"{row['country']:4s}{row['curated_records']:>17,}"
              f"{row['curated_species_present']:>9}"
              f"{row['curated_species_usable']:>8}"
              f"{row['median_records_per_usable']:>13,}"
              f"{row['records_if_capped']:>13,}"
              f"{row['top5_share_pct']:>8.1f}")
    print(f"\n'usable' = curated species with >= {MIN_RECORDS_PER_SPECIES} "
          f"records in that country")
    print(f"Written to {paths()['countries']}")
    return frame


def suggest_species(client, entries, country=None):
    """Top-recorded species in the country that are NOT in the curated list.

    This is the data-driven rule, run as a check rather than as the rule. Read
    it and add anything genuinely well known to tree_species_list.csv — but
    expect most of it to be hedgerow shrubs and thicket species.
    """
    country = (country or COUNTRY).upper()
    have = {e["species"].lower() for e in entries}
    found = []

    for genus in tqdm(SUGGEST_GENERA, desc="Scanning genera", unit="genus"):
        info, reason = match_genus(client, genus)
        if info is None:
            continue
        facets = client.facet_counts("speciesKey", taxonKey=info["key"],
                                     country=country,
                                     basis_of_record=BASIS_OF_RECORD,
                                     **OCCURRENCE_FILTERS)
        for key, n in facets[:25]:
            if n < MIN_RECORDS_PER_SPECIES * 4:
                continue
            record = client.species(key)
            name = (record or {}).get("canonicalName") or ""
            if not name or name.lower() in have or len(name.split()) != 2:
                continue
            found.append({"species": name, "genus": genus, "records": n,
                          "family": (record or {}).get("family")})

    found.sort(key=lambda r: -r["records"])
    frame = pd.DataFrame(found)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frame.to_csv(paths(country)["suggest"], index=False)

    print(f"\nTop 30 species in {country} absent from {ACTIVE_SPECIES_LIST}:")
    for row in found[:30]:
        print(f"  {row['species']:32s} {row['records']:>9,}  {row['family']}")
    print(f"\nFull list: {paths(country)['suggest']}")
    print("Most of these are shrubs or thicket species; add only what a user "
          "would plausibly plant.")
    return frame


# ─────────────────────────────────────────────────────
# STAGE 1: PLAN
# ─────────────────────────────────────────────────────
def resolve_species(client, entries, country=None, with_counts=True):
    """Curated name -> verified plant species key, with its record count."""
    resolved, failed = [], []
    desc = "Resolving species" + (f" in {country}" if country else "")

    for entry in tqdm(entries, desc=desc, unit="sp"):
        try:
            info, reason = match_species(client, entry["species"])
        except GbifRequestError as exc:
            info, reason = None, str(exc)
        if info is None:
            failed.append({**entry, "reason": reason})
            continue

        item = {**entry, **info}
        if with_counts:
            item["records"] = client.count(taxonKey=info["key"],
                                           country=country,
                                           basis_of_record=BASIS_OF_RECORD,
                                           **OCCURRENCE_FILTERS)
        resolved.append(item)

    return resolved, failed


def build_plan(client, country=None):
    country = (country or COUNTRY).upper()
    out = paths(country)
    entries = load_species_list()

    print("=" * 62)
    print(f"PLAN — curated plantable trees in {country}")
    print("=" * 62)
    print(f"Species list:            {ACTIVE_SPECIES_LIST} ({len(entries)} names)")
    print(f"Country:                 {country}")
    print(f"Min records per species: {MIN_RECORDS_PER_SPECIES}")
    print(f"Max records per species: {MAX_RECORDS_PER_SPECIES}")
    print(f"Sampling strata:         {len(YEAR_BANDS)} year bands x "
          f"{LAT_BANDS} latitude bands = "
          f"{len(YEAR_BANDS) * LAT_BANDS} per species")
    print(f"basisOfRecord kept:      {len(BASIS_OF_RECORD)} of 11 "
          f"(no FOSSIL_SPECIMEN, no LIVING_SPECIMEN)\n")

    resolved, unresolved = resolve_species(client, entries, country=country)

    selected = [s for s in resolved if s["records"] >= MIN_RECORDS_PER_SPECIES]
    too_few = [s for s in resolved if s["records"] < MIN_RECORDS_PER_SPECIES]
    for item in selected:
        item["take"] = min(item["records"], MAX_RECORDS_PER_SPECIES)
        item["capped"] = item["records"] > MAX_RECORDS_PER_SPECIES

    throughput = measure_throughput(client, selected, country)

    total_take = sum(s["take"] for s in selected)
    n_strata = len(YEAR_BANDS) * LAT_BANDS
    # A species costs one request per page of its budget, plus one per stratum
    # that returns a short page, plus the top-up pass.
    requests_est = sum(math.ceil(s["take"] / PAGE_SIZE) +
                       min(n_strata, max(1, s["take"] // PAGE_SIZE)) + 2
                       for s in selected)
    est_seconds = requests_est * max(throughput["seconds_per_page"],
                                     gbif_api.MIN_INTERVAL)

    plan = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "country": country,
        "config": {
            "species_list_file": ACTIVE_SPECIES_LIST,
            "curated_names": len(entries),
            "min_records_per_species": MIN_RECORDS_PER_SPECIES,
            "max_records_per_species": MAX_RECORDS_PER_SPECIES,
            "max_coordinate_uncertainty_m": MAX_COORDINATE_UNCERTAINTY_M,
            "accept_unknown_uncertainty": ACCEPT_UNKNOWN_UNCERTAINTY,
            "year_bands": YEAR_BANDS,
            "lat_bands": LAT_BANDS,
            "occurrence_filters": OCCURRENCE_FILTERS,
            "basis_of_record": BASIS_OF_RECORD,
            "inclusion_rule": (
                "in tree_species_list.csv AND resolves via /species/match to "
                "an accepted Plantae/Tracheophyta species AND has >= "
                f"{MIN_RECORDS_PER_SPECIES} records in {country}"),
        },
        "species": sorted(selected, key=lambda s: (-s["records"], s["species"])),
        "species_too_few_records": sorted(too_few, key=lambda s: -s["records"]),
        "species_unresolved": unresolved,
        "throughput": throughput,
        "estimate": {
            "species": len(selected),
            "records_available": sum(s["records"] for s in selected),
            "records_to_download": total_take,
            "species_at_cap": sum(1 for s in selected if s["capped"]),
            "api_requests": requests_est,
            "wall_clock_seconds": est_seconds,
            "csv_bytes": total_take * 58,
        },
    }

    os.makedirs(out["dir"], exist_ok=True)
    with open(out["plan"], "w") as handle:
        json.dump(plan, handle, indent=2)

    report_plan(plan, client)
    return plan


def measure_throughput(client, selected, country, n=3):
    """Time a few real pages so the estimate is measured, not guessed."""
    elapsed, records = 0.0, 0
    for item in selected[:n]:
        start = time.monotonic()
        page = client.occurrence_search(limit=PAGE_SIZE, taxonKey=item["key"],
                                        country=country,
                                        basis_of_record=BASIS_OF_RECORD,
                                        **OCCURRENCE_FILTERS)
        elapsed += time.monotonic() - start
        records += len(page.get("results", []))
    runs = max(min(n, len(selected)), 1)
    return {"seconds_per_page": elapsed / runs,
            "records_per_page": records / runs}


def report_plan(plan, client=None):
    estimate = plan["estimate"]
    print("\n" + "=" * 62)
    print(f"PLAN — {plan['country']}")
    print("=" * 62)
    print(f"Curated names:           {plan['config']['curated_names']}")
    print(f"Species selected:        {estimate['species']}")
    print(f"Below the {plan['config']['min_records_per_species']}-record floor:"
          f"      {len(plan['species_too_few_records'])}")
    print(f"Name did not resolve:    {len(plan['species_unresolved'])}")
    print(f"Records available:       {estimate['records_available']:,}")
    print(f"Records to download:     {estimate['records_to_download']:,}")
    print(f"Species at the cap:      {estimate['species_at_cap']}")
    print(f"API requests:            {estimate['api_requests']:,}")
    print(f"Estimated wall clock:    {human_time(estimate['wall_clock_seconds'])}")
    print(f"Estimated raw CSV:       {human_bytes(estimate['csv_bytes'])}")

    if plan["species_unresolved"]:
        print(f"\nUnresolved names:")
        for item in plan["species_unresolved"]:
            print(f"    {item['species']:30s} {item['reason']}")

    if plan["species_too_few_records"]:
        print(f"\nWell-known trees dropped for lack of data in "
              f"{plan['country']} (top 15 by what they do have):")
        for item in plan["species_too_few_records"][:15]:
            print(f"    {item['species']:30s} {item['records']:>6,} records  "
                  f"{item.get('common_name', '')}")

    print(f"\nBest-sampled 10:")
    for item in plan["species"][:10]:
        flag = "  CAPPED" if item["capped"] else ""
        print(f"    {item['species']:30s} {item['records']:>9,} available, "
              f"take {item['take']:>5,}{flag}  {item.get('common_name', '')}")

    if client is not None:
        print(f"\nPlanning cost: {client.n_requests} requests, "
              f"{human_bytes(client.bytes_received)}, {client.n_retries} retries")
    print(f"\nNext: python3 download_species.py --download")


# ─────────────────────────────────────────────────────
# STAGE 2: DOWNLOAD  (stratified, shallow offsets, resumable)
# ─────────────────────────────────────────────────────
def page_stratum(client, taxon_key, country, budget, seen, start_offset=0,
                 max_requests=MAX_REQUESTS_PER_SPECIES, **stratum):
    """Collect up to `budget` records from one stratum.

    Offsets stay within a few pages because the stratum is small — this is the
    whole reason for stratifying rather than paging one big query. Returns the
    offset it stopped at so a later top-up pass resumes instead of re-reading
    from zero, which is what made the first version four times slower than it
    needed to be.
    """
    rows, offset, available = [], start_offset, 0
    exhausted = False
    spent = 0

    while len(rows) < budget and spent < max_requests:
        spent += 1
        limit = min(PAGE_SIZE, budget - len(rows))
        page = client.occurrence_search(limit=limit, offset=offset,
                                        taxonKey=taxon_key, country=country,
                                        basis_of_record=BASIS_OF_RECORD,
                                        **OCCURRENCE_FILTERS, **stratum)
        available = int(page.get("count", 0))
        results = page.get("results", [])
        if not results:
            exhausted = True
            break

        for record in results:
            key = record.get("key")
            if key in seen or not usable_record(record):
                continue
            seen.add(key)
            rows.append({
                "species": record.get("species") or record.get("scientificName"),
                "latitude": record["decimalLatitude"],
                "longitude": record["decimalLongitude"],
                "year": record.get("year"),
                "country": record.get("countryCode"),
                "basis": record.get("basisOfRecord"),
            })

        offset += limit
        if page.get("endOfRecords") or offset >= gbif_api.MAX_OFFSET:
            exhausted = True
            break

    return rows, offset, exhausted, spent


def choose_strata(strata, budget):
    """Use no more strata than the budget can fill a page from.

    A stratum costs at least one request whatever it returns, so asking twelve
    strata for 5 records each is twelve requests for 60 records. The count is
    scaled to the budget and the chosen strata are spread evenly through the
    list, so two strata means one old year band and one recent one rather than
    two adjacent ones.
    """
    wanted = max(1, min(len(strata), budget // PAGE_SIZE))
    if wanted >= len(strata):
        return strata
    step = len(strata) / wanted
    return [strata[int(i * step)] for i in range(wanted)]


def download_species(client, item, country, all_strata):
    """Fetch one species across its strata, then top up any shortfall."""
    budget = item["take"]
    strata = choose_strata(all_strata, budget)
    seen, rows = set(), []
    offsets = [0] * len(strata)
    done = [False] * len(strata)
    share = max(1, budget // len(strata))
    left = MAX_REQUESTS_PER_SPECIES

    # Pass 1: an equal share from every stratum. Strata holding fewer records
    # than their share just return what they have.
    for i, stratum in enumerate(strata):
        if len(rows) >= budget or left <= 0:
            break
        got, offsets[i], done[i], spent = page_stratum(
            client, item["key"], country, min(share, budget - len(rows)),
            seen, max_requests=left, **stratum)
        rows += got
        left -= spent
    used_strata = sum(1 for o in offsets if o)

    # Pass 2: strata that still have records absorb what pass 1 could not
    # fill, so a species concentrated in one decade is not under-sampled.
    # Each resumes at its own offset rather than re-reading from zero.
    for i, stratum in enumerate(strata):
        if len(rows) >= budget or left <= 0:
            break
        if done[i]:
            continue
        got, offsets[i], done[i], spent = page_stratum(
            client, item["key"], country, budget - len(rows), seen,
            start_offset=offsets[i], max_requests=left, **stratum)
        rows += got
        left -= spent

    # Pass 3: unstratified. Picks up records carrying no year at all, which no
    # year band can reach. Only worth running for a real shortfall — chasing
    # the last few percent is what made this pathologically slow.
    if len(rows) < budget * (1 - TOPUP_MIN_SHORTFALL) and left > 0:
        got, _, _, spent = page_stratum(client, item["key"], country,
                                        budget - len(rows), seen,
                                        max_requests=left)
        rows += got
        left -= spent

    years = [r["year"] for r in rows if r["year"]]
    return rows, {
        "species_key": item["key"],
        "species": item["species"],
        "common_name": item.get("common_name", ""),
        "use": item.get("use", ""),
        "records_available": item["records"],
        "records_requested": budget,
        "records_written": len(rows),
        "capped": bool(item["capped"]),
        "strata_with_data": used_strata,
        "requests_used": MAX_REQUESTS_PER_SPECIES - left,
        "hit_request_cap": left <= 0,
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
    }


def run_download(client, plan, limit_species=None):
    country = plan["country"]
    out = paths(country)
    os.makedirs(out["raw_dir"], exist_ok=True)
    strata = build_strata(country)

    species = plan["species"]
    if limit_species:
        species = species[:limit_species]

    print("=" * 62)
    print(f"DOWNLOAD — {country}")
    print("=" * 62)
    print(f"Species:          {len(species)}")
    print(f"Records to fetch: {sum(s['take'] for s in species):,}")
    print(f"Strata/species:   {len(strata)}")
    print(f"Checkpoints:      {out['raw_dir']}/<speciesKey>.csv "
          f"(delete one to refetch it)\n")

    meta_rows, skipped = [], 0
    progress = tqdm(species, desc="Species", unit="sp")
    for item in progress:
        csv_path = os.path.join(out["raw_dir"], f"{item['key']}.csv")
        meta_path = os.path.join(out["raw_dir"], f"{item['key']}.json")

        if os.path.exists(csv_path) and os.path.exists(meta_path):
            with open(meta_path) as handle:
                meta_rows.append(json.load(handle))
            skipped += 1
            continue

        progress.set_postfix_str(item["species"][:26])
        try:
            rows, meta = download_species(client, item, country, strata)
        except GbifRequestError as exc:
            tqdm.write(f"  FAILED {item['species']}: {exc}")
            continue

        atomic_write_csv(pd.DataFrame(rows, columns=[
            "species", "latitude", "longitude", "year", "country", "basis"]),
            csv_path)
        with open(meta_path, "w") as handle:
            json.dump(meta, handle, indent=2)
        meta_rows.append(meta)

    if skipped:
        print(f"\nResumed from checkpoints: {skipped} species")
    merge_raw(country, meta_rows)


def merge_raw(country, meta_rows):
    """Concatenate the per-species checkpoints into one raw CSV.

    Only species whose checkpoint is provably complete are admitted. A species
    is complete when its CSV and its metadata JSON both exist and the CSV holds
    exactly the number of rows the metadata claims. That matters because a
    species interrupted partway through its strata would carry records from
    only some year or latitude bands, which biases that species towards
    whichever periods happened to finish — invisible in the merged file.

    The writes are ordered to make this checkable: records go to a .part file
    that is then renamed, and the metadata is written only afterwards. So an
    interrupted species leaves either nothing or an unpaired file, never a
    plausible-looking short one.
    """
    out = paths(country)
    frames, admitted, rejected = [], [], []

    for name in sorted(os.listdir(out["raw_dir"])):
        if not name.endswith(".csv"):
            continue
        key = name[:-4]
        csv_path = os.path.join(out["raw_dir"], name)
        meta_path = os.path.join(out["raw_dir"], f"{key}.json")

        if not os.path.exists(meta_path):
            rejected.append((key, "no metadata — interrupted mid-species"))
            continue
        with open(meta_path) as handle:
            meta = json.load(handle)

        frame = pd.read_csv(csv_path)
        if len(frame) != meta.get("records_written"):
            rejected.append((key, f"{len(frame)} rows but metadata says "
                                  f"{meta.get('records_written')}"))
            continue
        if not len(frame):
            rejected.append((key, "no records"))
            continue

        frames.append(frame)
        admitted.append(meta)

    if rejected:
        print(f"\nExcluded {len(rejected)} incomplete species:")
        for key, why in rejected:
            print(f"    {key}: {why}")
    else:
        print(f"\nIntegrity check: all {len(admitted)} species checkpoints "
              f"complete, 0 partial")

    # Prefer the metadata read back from disk, so --merge-only reports the
    # same numbers as a full run.
    meta_rows = admitted or meta_rows
    if not frames:
        print("Nothing to merge.")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(out["raw_csv"], index=False)
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(out["log"], index=False)

    print("\n" + "=" * 62)
    print("DOWNLOAD DONE")
    print("=" * 62)
    print(f"Records:   {len(merged):,}")
    print(f"Species:   {merged['species'].nunique()}")
    print(f"Years:     {merged['year'].min():.0f}-{merged['year'].max():.0f}, "
          f"median {merged['year'].median():.0f}")
    print(f"Basis mix: {merged['basis'].value_counts(normalize=True).head(4).round(3).to_dict()}")
    print(f"Raw CSV:   {out['raw_csv']} "
          f"({human_bytes(os.path.getsize(out['raw_csv']))})")
    print(f"Log:       {out['log']}")
    print(f"\nNext: python3 clean_species.py")


# ─────────────────────────────────────────────────────
def load_plan(country=None):
    path = paths(country)["plan"]
    if not os.path.exists(path):
        raise SystemExit(f"No plan at {path}. "
                         f"Run: python3 download_species.py --plan")
    with open(path) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(
        description="Download curated plantable-tree occurrences for one "
                    "country, by GBIF taxonKey.")
    parser.add_argument("--country", default=COUNTRY,
                        help=f"ISO country code (default {COUNTRY})")
    parser.add_argument("--countries", action="store_true",
                        help="compare candidate countries and exit")
    parser.add_argument("--suggest", action="store_true",
                        help="list well-recorded species missing from the CSV")
    parser.add_argument("--plan", action="store_true",
                        help="resolve the curated list and size the job")
    parser.add_argument("--download", action="store_true", help="fetch records")
    parser.add_argument("--show-plan", action="store_true",
                        help="reprint the saved plan without calling GBIF")
    parser.add_argument("--selftest", action="store_true",
                        help="probe every endpoint with a few requests")
    parser.add_argument("--merge-only", action="store_true",
                        help="rebuild the merged CSV from checkpoints")
    parser.add_argument("--limit-species", type=int, default=None,
                        help="stop after N species; for a smoke test")
    parser.add_argument("--species-list", default=None,
                        help="curated CSV to read (default "
                             "tree_species_list_<COUNTRY>.csv where it exists, "
                             f"else {SPECIES_LIST_FILE})")
    args = parser.parse_args()

    country = args.country.upper()

    global ACTIVE_SPECIES_LIST
    ACTIVE_SPECIES_LIST = args.species_list or species_list_for(country)

    if args.selftest:
        raise SystemExit(0 if gbif_api.selftest() else 1)
    if args.show_plan:
        report_plan(load_plan(country))
        return
    if args.merge_only:
        merge_raw(country, [])
        return

    client = GbifClient()
    if args.countries:
        compare_countries(client, load_species_list())
        return
    if args.suggest:
        suggest_species(client, load_species_list(), country)
        return
    if args.plan:
        build_plan(client, country)
        return
    if args.download:
        run_download(client, load_plan(country), args.limit_species)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
