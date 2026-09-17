"""
Pre-flight compatibility check for the deep SDM inputs

Answers one question: can `deep_sdm_training.py` be trusted to train on what is
currently on disk? Occurrences, background points and predictor rasters are
produced by three different pipelines on their own schedules, so "they were
compatible last time" is not evidence. This re-checks the joins that training
silently depends on, and exits non-zero if any of them is broken.

The checks, in the order a bad input would do damage:

  1. GRIDS      every predictor raster shares one exact affine per profile, and
                the coarse and fine profiles nest on an integer factor
  2. INDICES    the grid_row / grid_col columns shipped in the occurrence file
                reproduce from the climate template's transform
  3. EXTENTS    occurrence and background coordinates fall inside the rasters
  4. OVERLAP    background covers the presence cells it is meant to stand in for
  5. INTEGRITY  no duplicate species-cell rows, and the species sets agree
                between the cleaned and thinned files
  6. THINNING   re-thinning the cleaned file at each profile grid still works,
                and how many species-cell records it yields
  7. COVERAGE   predictors are actually present under the occurrence points

Run from the repo directory:

    python3 verify_sdm_inputs.py
    python3 verify_sdm_inputs.py --profile fra_30s
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio

import sdm_features as feat

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
PROFILES = ["fra_10m", "fra_30s"]
NESTING_PAIRS = [("fra_10m", "fra_30s")]

THINNED = "species_occurrences/gbif_trees_thinned.csv"
CLEAN = "species_occurrences/gbif_trees_clean.csv"
# The France background, named explicitly. This used to read
# background_points.csv, a country-agnostic name that sampling_effort.py rewrites
# for whichever country it last ran: after Spain was prepared that file became a
# byte-for-byte copy of background_ES_30s.csv, and this check failed with 72% of
# the points outside the French rasters. Both profiles here cover France and
# share an origin and bounds, so the 1 km file is valid against either grid.
BACKGROUND = "sampling_effort/background_FR_30s.csv"

# The grid the shipped grid_row / grid_col columns are indexed against. The
# occurrence pipeline works on the global 10 arcmin lattice, not the clipped
# French window, so this is the template the indices must reproduce from.
INDEX_TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"

TOLERANCE_DEG = 1e-9
FAILURES = []


def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def check(label, ok, detail="", fatal=True):
    """Print one PASS/FAIL line and remember failures."""
    print(f"   [{'PASS' if ok else 'FAIL'}] {label}"
          + (f"  —  {detail}" if detail else ""))
    if not ok and fatal:
        FAILURES.append(label)
    return ok


# ─────────────────────────────────────────────────────
# 1. GRIDS
# ─────────────────────────────────────────────────────
def check_grids(profiles):
    divider("1. GRIDS  -  do all predictor rasters share one lattice?")
    grids = {}
    for profile in profiles:
        spec = feat.GRID_PROFILES[profile]
        if not os.path.exists(spec["template"]):
            check(f"{profile} template present", False, spec["template"])
            continue
        grid = feat.load_grid(profile)
        grids[profile] = grid
        catalog = feat.build_catalog(profile)

        reference = (grid["width"], grid["height"], grid["crs"],
                     tuple(round(v, 12) for v in tuple(grid["transform"])[:6]))
        bad = []
        for layer in catalog:
            with rasterio.open(layer["path"]) as src:
                here = (src.width, src.height, str(src.crs),
                        tuple(round(v, 12)
                              for v in tuple(src.transform)[:6]))
            if here != reference:
                bad.append(layer["name"])

        counts = {}
        for layer in catalog:
            counts[layer["group"]] = counts.get(layer["group"], 0) + 1
        print(f"\n   {profile}: {grid['width']}x{grid['height']} @ "
              f"{grid['cell']:.10f} deg, origin "
              f"({grid['transform'].c}, {grid['transform'].f}), {grid['crs']}")
        print(f"   {len(catalog)} predictor rasters: "
              + ", ".join(f"{g} {n}" for g, n in sorted(counts.items())))
        check(f"{profile}: all {len(catalog)} predictors on one exact affine",
              not bad, f"mismatched: {bad}" if bad else "")
        check(f"{profile}: no vegetation or provenance layer in the set",
              feat.assert_no_vegetation_layers(catalog))

    for coarse, fine in NESTING_PAIRS:
        if coarse not in grids or fine not in grids:
            continue
        a, b = grids[coarse], grids[fine]
        factor = a["cell"] / b["cell"]
        integer = abs(factor - round(factor)) < 1e-9
        same_origin = (abs(a["transform"].c - b["transform"].c) < TOLERANCE_DEG
                       and abs(a["transform"].f - b["transform"].f)
                       < TOLERANCE_DEG)
        check(f"{coarse} and {fine} nest on an integer factor",
              integer and same_origin,
              f"factor {factor:.6f}, shared origin {same_origin}")
    return grids


# ─────────────────────────────────────────────────────
# 2. INDICES
# ─────────────────────────────────────────────────────
def check_indices(thinned):
    divider("2. INDICES  -  do the shipped grid_row/grid_col reproduce?")
    if not {"grid_row", "grid_col"} <= set(thinned.columns):
        check("thinned file carries grid_row / grid_col", False)
        return
    if not os.path.exists(INDEX_TEMPLATE):
        check(f"index template present ({INDEX_TEMPLATE})", False)
        return

    with rasterio.open(INDEX_TEMPLATE) as src:
        transform = src.transform
        shape = (src.height, src.width)
    print(f"   recomputing against {INDEX_TEMPLATE} "
          f"({shape[1]}x{shape[0]} @ {abs(transform.a):.10f} deg)")

    row = np.floor((thinned["latitude"].to_numpy(dtype="float64")
                    - transform.f) / transform.e).astype("int64")
    col = np.floor((thinned["longitude"].to_numpy(dtype="float64")
                    - transform.c) / transform.a).astype("int64")
    row_ok = int((row == thinned["grid_row"].to_numpy()).sum())
    col_ok = int((col == thinned["grid_col"].to_numpy()).sum())
    n = len(thinned)
    check("grid_row reproduces from the template transform", row_ok == n,
          f"{row_ok:,}/{n:,} match")
    check("grid_col reproduces from the template transform", col_ok == n,
          f"{col_ok:,}/{n:,} match")


# ─────────────────────────────────────────────────────
# 3. EXTENTS
# ─────────────────────────────────────────────────────
def check_extents(grids, frames):
    divider("3. EXTENTS  -  do the points fall inside the rasters?")
    for name, frame in frames.items():
        lat = frame["latitude"].to_numpy(dtype="float64")
        lon = frame["longitude"].to_numpy(dtype="float64")
        print(f"\n   {name}: {len(frame):,} rows, "
              f"lat {lat.min():.2f}..{lat.max():.2f}, "
              f"lon {lon.min():.2f}..{lon.max():.2f}")
        for profile, grid in grids.items():
            _, _, inside = feat.lonlat_to_rowcol(lat, lon, grid)
            outside = int((~inside).sum())
            check(f"{name} inside {profile} raster extent", outside == 0,
                  f"{outside:,} of {len(frame):,} outside "
                  f"({100 * outside / len(frame):.3f}%)")


# ─────────────────────────────────────────────────────
# 4. OVERLAP
# ─────────────────────────────────────────────────────
def check_overlap(grids, thinned, background):
    divider("4. OVERLAP  -  does background stand where presences stand?")
    for profile, grid in grids.items():
        prow, pcol, pin = feat.lonlat_to_rowcol(
            thinned["latitude"].to_numpy(), thinned["longitude"].to_numpy(),
            grid)
        brow, bcol, bin_ = feat.lonlat_to_rowcol(
            background["latitude"].to_numpy(),
            background["longitude"].to_numpy(), grid)
        pcells = set(zip(prow[pin].tolist(), pcol[pin].tolist()))
        bcells = set(zip(brow[bin_].tolist(), bcol[bin_].tolist()))
        shared = len(pcells & bcells)
        print(f"\n   {profile}: {len(pcells):,} presence cells, "
              f"{len(bcells):,} distinct background cells, "
              f"{shared:,} shared "
              f"({100 * shared / max(len(pcells), 1):.1f}% of presence cells)")
        # Not fatal: heavy overlap is what target-group background is *for*,
        # and near-zero overlap at 1 km is the known resolution mismatch that
        # sdm_features handles by resampling the effort surface.
        check(f"{profile}: background has some presence-cell overlap",
              shared > 0, fatal=False)


# ─────────────────────────────────────────────────────
# 5. INTEGRITY
# ─────────────────────────────────────────────────────
def check_integrity(thinned, clean):
    divider("5. INTEGRITY  -  duplicates and species-set agreement")
    if {"grid_row", "grid_col"} <= set(thinned.columns):
        dupes = int(thinned.duplicated(
            subset=["species", "grid_row", "grid_col"]).sum())
        check("no duplicate species-plus-cell rows in the thinned file",
              dupes == 0, f"{dupes:,} duplicates")

    thin_species = set(thinned["species"].unique())
    clean_species = set(clean["species"].unique())
    only_thin = sorted(thin_species - clean_species)
    only_clean = sorted(clean_species - thin_species)
    print(f"   thinned {len(thin_species)} species, "
          f"clean {len(clean_species)} species")
    check("species sets agree between clean and thinned",
          not only_thin and not only_clean,
          f"thinned-only {only_thin[:5]}, clean-only {only_clean[:5]}")

    counts = thinned["species"].value_counts()
    print(f"   per-species thinned records: min {counts.min():,}, "
          f"p10 {int(np.percentile(counts, 10)):,}, "
          f"median {int(counts.median()):,}, "
          f"p90 {int(np.percentile(counts, 90)):,}, max {counts.max():,}")
    check("every species clears the 30-record floor", counts.min() >= 30,
          f"thinnest: {counts.idxmin()} at {counts.min()}")
    return counts


# ─────────────────────────────────────────────────────
# 6. THINNING
# ─────────────────────────────────────────────────────
def check_thinning(grids, clean):
    divider("6. THINNING  -  re-thinning the cleaned file per profile")
    results = {}
    for profile, grid in grids.items():
        thinned = feat.thin_to_grid(clean.copy(), grid)
        cells = thinned.groupby(["grid_row", "grid_col"], sort=False).ngroups
        results[profile] = (len(thinned), cells)
        print(f"   {profile}: {len(thinned):,} species-cell records over "
              f"{cells:,} distinct cells, {thinned['species'].nunique()} "
              f"species")
        check(f"{profile}: re-thinning yields records", len(thinned) > 0)
    return results


# ─────────────────────────────────────────────────────
# 7. COVERAGE
# ─────────────────────────────────────────────────────
def check_coverage(grids, thinned):
    divider("7. COVERAGE  -  are predictors present under the points?")
    sample = thinned.sample(n=min(20000, len(thinned)), random_state=42)
    for profile, grid in grids.items():
        catalog = feat.build_catalog(profile)
        features, inside = feat.sample_points(
            catalog, sample["latitude"].to_numpy(),
            sample["longitude"].to_numpy(), grid, progress=False)
        groups = np.asarray(feat.feature_groups(catalog))
        print(f"\n   {profile} ({int(inside.sum()):,} sampled points)")
        for group in sorted(set(groups.tolist())):
            block = features[:, groups == group]
            complete = 100 * np.isfinite(block).all(axis=1).mean()
            print(f"      {group:<10s} complete at {complete:6.2f}% of points")
        usable = feat.drop_empty_rows(features) & inside
        share = 100 * usable.mean()
        check(f"{profile}: usable rows under occurrence points", share > 90,
              f"{share:.2f}% usable")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--profile", action="append", default=None,
                        choices=sorted(feat.GRID_PROFILES))
    args = parser.parse_args()
    profiles = args.profile or PROFILES

    divider("DEEP SDM INPUT COMPATIBILITY CHECK")
    for path in (THINNED, CLEAN, BACKGROUND):
        exists = os.path.exists(path)
        size = os.path.getsize(path) / 1e6 if exists else 0
        check(f"present: {path}", exists, f"{size:.2f} MB" if exists else "")
    if FAILURES:
        print("\nMissing inputs, cannot continue.")
        return 1

    thinned = pd.read_csv(THINNED)
    clean = pd.read_csv(CLEAN)
    background = pd.read_csv(BACKGROUND)

    grids = check_grids(profiles)
    check_indices(thinned)
    check_extents(grids, {"presences (thinned)": thinned,
                          "background": background})
    check_overlap(grids, thinned, background)
    check_integrity(thinned, clean)
    check_thinning(grids, clean)
    check_coverage(grids, thinned)

    divider("VERDICT")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for label in FAILURES:
            print(f"   {label}")
        print("\nDo not train on these inputs.")
        return 1
    print("All checks passed. Inputs are mutually compatible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
