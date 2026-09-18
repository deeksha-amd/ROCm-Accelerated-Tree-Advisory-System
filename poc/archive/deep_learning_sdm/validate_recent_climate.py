"""
Validate the present-day bioclim rasters produced by recent_climate_rasters.py.

Four checks, in order of how much they prove:

  1. BIOCLIM IMPLEMENTATION CORRECTNESS  <-- the strongest check
     Feed our biovars() the WorldClim 2.1 1970-2000 MONTHLY NORMALS, which are
     the exact same inputs WorldClim itself used, and confirm we reproduce the
     published bio_1 ... bio_19 rasters in climate_current/. If our formulas are
     right this agrees to float32 rounding. Several bioclim variables (BIO8,
     BIO9, BIO18, BIO19) are defined on rolling three-month quarters that wrap
     from December round to January, and are easy to get subtly wrong; this is
     the check that catches that.

  2. GRID IDENTITY
     Every output raster must sit on exactly the same 2160 x 1080 grid as
     climate_current/wc2.1_10m_bio_1.tif, cell for cell.

  3. COVERAGE AT THE GBIF POINTS
     The CRU-TS-derived monthly grids use a coarser land mask than WorldClim
     2.1, so some cells are lost. Count how many occurrence records that costs,
     against the 99.93% the existing climate_current/ achieves.

  4. THREE-WINDOW CLIMATE COMPARISON
     The number that decides whether any of this was worth doing: annual mean
     temperature (BIO1) and annual precipitation (BIO12) at all 392,725 GBIF
     occurrences, for 1970-2000, 2000-2015 and 2015-2024, plus the shifts.

USAGE
    python3 validate_recent_climate.py
"""

import os
import sys

import numpy as np
import pandas as pd
import rasterio

import recent_climate_rasters as rcr

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"
GBIF_CSV = "gbif_500_species.csv"

# The three blocks being compared: label -> (directory, filename prefix)
BLOCKS = [
    ("1970-2000", "climate_current", "wc2.1_10m_bio"),
    ("2000-2015", "climate_2000_2015", "wc2.1_10m_bio"),
    ("2015-2024", "climate_recent", "wc2.1_10m_bio"),
]
BASELINE = "1970-2000"
NEW_BLOCKS = ["2000-2015", "2015-2024"]

# float32 carries ~7 significant digits, so variables with large magnitudes
# (BIO4 up to ~2400, BIO12 up to ~11200 mm) cannot be compared tighter than this.
TOLERANCES = {3: 5e-3, 4: 5e-2, 12: 5e-2, 13: 5e-2,
              15: 5e-3, 16: 5e-2, 17: 5e-2, 18: 5e-2, 19: 5e-2}
DEFAULT_TOL = 5e-3


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def read_valid(path, band=1):
    with rasterio.open(path) as s:
        a = s.read(band).astype(np.float64)
    a[a < -1e30] = np.nan
    return a


def block_path(label, i):
    for lbl, d, prefix in BLOCKS:
        if lbl == label:
            return os.path.join(d, f"{prefix}_{i}.tif")
    raise KeyError(label)


# ─────────────────────────────────────────────────────
# 1. IMPLEMENTATION CORRECTNESS
# ─────────────────────────────────────────────────────
def check_implementation():
    hr("1. DOES OUR BIOCLIM REPRODUCE WORLDCLIM'S PUBLISHED 1970-2000 VALUES?")
    print("Inputs : WorldClim 2.1 published monthly normals (tmin, tmax, prec),")
    print("         i.e. the exact same inputs WorldClim used for its own bioclim.")
    print("Target : climate_current/wc2.1_10m_bio_1.tif ... _19.tif")
    print("A match here proves the 19 formulas, including the four rolling-quarter")
    print("variables and the standard-deviation conventions, are right.\n")

    for var in rcr.HIST_VARS:
        if rcr.ensure_normals_zip(var) is None:
            print("FAIL: could not obtain the monthly normals.")
            return False

    tmin = rcr.read_normals("tmin")
    tmax = rcr.read_normals("tmax")
    prec = rcr.read_normals("prec")
    bio, valid = rcr.biovars_grid(tmin, tmax, prec)
    n = int(valid.sum())
    print(f"land cells compared: {n:,}\n")

    print(f"{'BIO':>4} {'bit-identical cells':>22} {'max abs err':>12} "
          f"{'mean err':>11} {'WorldClim range':>20} result")
    print("-" * 78)
    ok_all = True
    n_exact = 0
    for i in range(1, 20):
        pub = np.asarray(rasterio.open(block_path(BASELINE, i)).read(1))
        ours = bio[i]
        identical = int((ours[valid] == pub[valid]).sum())
        d = ours[valid].astype(np.float64) - pub[valid].astype(np.float64)
        maxabs = float(np.max(np.abs(d)))
        tol = TOLERANCES.get(i, DEFAULT_TOL)
        ok = maxabs <= tol
        ok_all &= ok
        n_exact += identical == n
        lo, hi = float(np.min(pub[valid])), float(np.max(pub[valid]))
        print(f"{i:>4} {identical:>10,}/{n:,} {maxabs:12.6f} {d.mean():11.6f} "
              f"{f'[{lo:.1f}, {hi:.1f}]':>20} {'PASS' if ok else 'FAIL'}")
    print("-" * 78)
    print(f"{n_exact}/19 variables are BIT-FOR-BIT IDENTICAL to WorldClim's "
          f"published rasters")
    print("PASS: bioclim implementation verified." if ok_all
          else "FAIL: at least one variable does not reproduce.")
    return ok_all


# ─────────────────────────────────────────────────────
# 2. GRID IDENTITY
# ─────────────────────────────────────────────────────
def check_grid():
    hr("2. GRID IDENTITY vs " + TEMPLATE)
    with rasterio.open(TEMPLATE) as t:
        ref = dict(width=t.width, height=t.height, crs=t.crs,
                   transform=t.transform, dtype=t.dtypes[0], nodata=t.nodata)
    print(f"template : {ref['width']} x {ref['height']}  {ref['crs']}  "
          f"px={ref['transform'].a:.17f}")
    print(f"           upper-left ({ref['transform'].c}, {ref['transform'].f})  "
          f"{ref['dtype']}  nodata={ref['nodata']:.4g}\n")

    ok_all = True
    for label in NEW_BLOCKS:
        paths = [block_path(label, i) for i in range(1, 20)]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"{label}: FAIL — {len(missing)} rasters missing "
                  f"(e.g. {missing[0]})")
            ok_all = False
            continue
        bad = []
        for p in paths:
            with rasterio.open(p) as s:
                same = (s.width == ref['width'] and s.height == ref['height']
                        and s.crs == ref['crs']
                        and s.transform.almost_equals(ref['transform'],
                                                      precision=1e-12)
                        and s.dtypes[0] == ref['dtype']
                        and s.nodata is not None and s.nodata < -1e30)
            if not same:
                bad.append(p)
        if bad:
            print(f"{label}: FAIL — {len(bad)} rasters differ: {bad[:3]}")
            ok_all = False
        else:
            d = os.path.dirname(paths[0])
            print(f"{label}: PASS — all 19 rasters in {d}/ match the template "
                  f"grid, CRS, dtype and nodata")
    return ok_all


# ─────────────────────────────────────────────────────
# GBIF SAMPLING
# ─────────────────────────────────────────────────────
def gbif_rowcol(df):
    """Nearest-cell lookup: which grid row/col each occurrence falls in."""
    with rasterio.open(TEMPLATE) as t:
        tr = t.transform
    cols = np.floor((df["longitude"].to_numpy(dtype="float64") - tr.c) / tr.a)
    rows = np.floor((df["latitude"].to_numpy(dtype="float64") - tr.f) / tr.e)
    return rows.astype(np.int64), cols.astype(np.int64)


def sample_at_points(path, rows, cols, band=1):
    a = read_valid(path, band)
    v = np.full(rows.shape, np.nan)
    inside = ((rows >= 0) & (rows < a.shape[0])
              & (cols >= 0) & (cols < a.shape[1]))
    v[inside] = a[rows[inside], cols[inside]]
    return v


# ─────────────────────────────────────────────────────
# 3. COVERAGE
# ─────────────────────────────────────────────────────
def check_coverage(df, rows, cols, samples):
    hr("3. COVERAGE AT THE GBIF OCCURRENCE POINTS")
    n = len(df)
    print(f"records: {n:,}   species: {df['species'].nunique()}   "
          f"countries: {df['country'].nunique()}\n")

    print("Raster land-mask sizes:")
    with rasterio.open(TEMPLATE) as t:
        ref_cells = int((t.read(1) > -1e30).sum())
    print(f"    WorldClim 2.1 (climate_current/) : {ref_cells:>9,} cells")
    for label in NEW_BLOCKS:
        a = read_valid(block_path(label, 1))
        print(f"    CRU TS 4.09  ({label})        : "
              f"{int(np.isfinite(a).sum()):>9,} cells")

    print("\nGBIF records landing on a valid cell:")
    ok = {}
    for label, _, _ in BLOCKS:
        ok[label] = np.isfinite(samples[(label, 1)])
        print(f"    {label}  {ok[label].sum():>7,} / {n:,}  "
              f"({100 * ok[label].mean():.2f}%)")

    lost = ok[BASELINE] & ~ok[NEW_BLOCKS[-1]]
    print(f"\nlost by switching from {BASELINE} to {NEW_BLOCKS[-1]}: "
          f"{lost.sum():,} records ({100 * lost.mean():.2f}%)")
    if lost.sum():
        print("\nwhere those losses are (CRU TS drops narrow coasts "
              "and small islands):")
        for k, v in df.loc[lost, "country"].value_counts().head(10).items():
            print(f"    {str(k):<6} {v:>6,}")

    # The Antarctic land mask difference dominates the cell count but costs
    # almost no records, so separate the two effects.
    with rasterio.open(TEMPLATE) as t:
        old_mask = t.read(1) > -1e30
        tr = t.transform
    new_mask = np.isfinite(read_valid(block_path(NEW_BLOCKS[-1], 1)))
    gap = old_mask & ~new_mask
    gap_lat = tr.f + (np.where(gap)[0] + 0.5) * tr.e
    print(f"\ncell-level gap: {int(gap.sum()):,} cells "
          f"({100 * gap.sum() / old_mask.sum():.1f}% of WorldClim land)")
    print(f"    of which below 60S (Antarctica) : "
          f"{int((gap_lat < -60).sum()):>9,} "
          f"({100 * (gap_lat < -60).mean():.1f}%)")
    print(f"    of which everywhere else        : "
          f"{int((gap_lat >= -60).sum()):>9,} "
          f"({100 * (gap_lat >= -60).mean():.1f}%)")
    print("    Antarctica already has no soil and no terrain in this project,")
    print("    so it was never trainable ground.")
    return ok


# ─────────────────────────────────────────────────────
# 4. THREE-WINDOW COMPARISON
# ─────────────────────────────────────────────────────
def compare_windows(df, samples, ok):
    hr("4. THREE-WINDOW CLIMATE COMPARISON AT THE GBIF POINTS")
    common = ok[BLOCKS[0][0]].copy()
    for label, _, _ in BLOCKS:
        common &= ok[label]
    print(f"points with all three windows valid: {common.sum():,} "
          f"of {len(df):,}\n")

    for bio, name, unit in ((1, "BIO1  annual mean temperature", "C"),
                            (12, "BIO12 annual precipitation", "mm")):
        print(f"{name} ({unit})")
        print(f"    {'window':<12} {'mean':>9} {'sd':>9} {'median':>9} "
              f"{'5th pct':>9} {'95th pct':>9}")
        vals = {}
        for label, _, _ in BLOCKS:
            v = samples[(label, bio)][common]
            vals[label] = v
            print(f"    {label:<12} {v.mean():>9.3f} {v.std():>9.3f} "
                  f"{np.median(v):>9.3f} {np.percentile(v, 5):>9.3f} "
                  f"{np.percentile(v, 95):>9.3f}")

        print(f"\n    shifts ({unit}):")
        pairs = [(BASELINE, "2000-2015"), (BASELINE, "2015-2024"),
                 ("2000-2015", "2015-2024")]
        for a, b in pairs:
            d = vals[b] - vals[a]
            pct = 100 * d.mean() / abs(vals[a].mean())
            print(f"      {a} -> {b:<10} mean {d.mean():+8.3f}  "
                  f"median {np.median(d):+8.3f}  "
                  f"p5 {np.percentile(d, 5):+8.3f}  "
                  f"p95 {np.percentile(d, 95):+8.3f}  ({pct:+.2f}%)")
        if bio == 1:
            d = vals["2015-2024"] - vals[BASELINE]
            print(f"\n    warmer than {BASELINE} at "
                  f"{100 * (d > 0).mean():.2f}% of points; "
                  f"more than +1.0 C at {100 * (d > 1.0).mean():.2f}%")
        print()

    # Does the temperature shift depend on when the record was made?
    yr = pd.to_numeric(df["year"], errors="coerce").to_numpy()
    print("BIO1 shift from 1970-2000, broken down by the year of the record:")
    print(f"    {'record period':<16} {'records':>9} {'-> 2000-2015':>14} "
          f"{'-> 2015-2024':>14}")
    b0 = samples[(BASELINE, 1)]
    b1 = samples[("2000-2015", 1)]
    b2 = samples[("2015-2024", 1)]
    for lbl, lo, hi in [("before 1970", -1e9, 1969), ("1970-2000", 1970, 2000),
                        ("2001-2010", 2001, 2010), ("2011-2020", 2011, 2020),
                        ("2021-2026", 2021, 2026)]:
        sel = common & np.isfinite(yr) & (yr >= lo) & (yr <= hi)
        if sel.sum():
            print(f"    {lbl:<16} {sel.sum():>9,} "
                  f"{(b1[sel] - b0[sel]).mean():>+13.3f}C "
                  f"{(b2[sel] - b0[sel]).mean():>+13.3f}C")
    sel = common & ~np.isfinite(yr)
    if sel.sum():
        print(f"    {'no year given':<16} {sel.sum():>9,} "
              f"{(b1[sel] - b0[sel]).mean():>+13.3f}C "
              f"{(b2[sel] - b0[sel]).mean():>+13.3f}C")

    # Also report all 19 variables, briefly, so nothing is hidden.
    print("\nAll 19 variables, mean value at the occurrence points:")
    print(f"    {'BIO':>4} {'1970-2000':>12} {'2000-2015':>12} "
          f"{'2015-2024':>12} {'shift to recent':>17}")
    for i in range(1, 20):
        m0 = samples[(BASELINE, i)][common].mean()
        m1 = samples[("2000-2015", i)][common].mean()
        m2 = samples[("2015-2024", i)][common].mean()
        print(f"    {i:>4} {m0:>12.3f} {m1:>12.3f} {m2:>12.3f} "
              f"{m2 - m0:>+17.3f}")
    return True


def main():
    print("=" * 78)
    print("VALIDATE PRESENT-DAY CLIMATE RASTERS")
    print("=" * 78)
    for label, d, prefix in BLOCKS:
        print(f"    {label:<12} {d}/{prefix}_[1-19].tif")

    results = {"implementation": check_implementation(),
               "grid": check_grid()}

    print("\nreading GBIF occurrences and sampling all three blocks...")
    df = pd.read_csv(GBIF_CSV)
    rows, cols = gbif_rowcol(df)
    samples = {}
    for label, _, _ in BLOCKS:
        for i in range(1, 20):
            samples[(label, i)] = sample_at_points(block_path(label, i),
                                                   rows, cols)

    ok = check_coverage(df, rows, cols, samples)
    results["coverage"] = bool(ok[NEW_BLOCKS[-1]].mean() > 0.99)
    results["comparison"] = compare_windows(df, samples, ok)

    hr("SUMMARY")
    for k, v in results.items():
        print(f"    {k:<18} {'PASS' if v else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
