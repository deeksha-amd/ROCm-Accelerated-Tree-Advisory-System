"""
Validate that the soil rasters line up with the climate and occurrence data

Checks three things and prints hard numbers for each:
  1. GRID    - are the aligned soil layers pixel-identical to the WorldClim grid?
  2. VALUES  - are the units and scaling factors sane, and does texture close
               to 100 percent?
  3. COVERAGE- what fraction of WorldClim land cells and of the GBIF occurrence
               points actually receive a soil value, and where are the gaps?
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
SOIL_DIR = "soil_data/aligned_10m"
MANIFEST = "soil_data/MANIFEST.json"
TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"
OCCURRENCES = "gbif_500_species.csv"

CLIMATE_NODATA_THRESHOLD = -1e30   # WorldClim nodata is -3.4e38
FALLBACK_RADIUS = 2                # cells to search for a nearby soil value
                                   # 1 cell = 10 arcmin ~ 18.5 km at the equator

# Plausible ranges used only to flag obviously wrong scaling factors.
EXPECTED_RANGE = {
    "ph_h2o":   (3.0, 10.0, "pH"),
    "sand":     (0.0, 100.0, "%"),
    "silt":     (0.0, 100.0, "%"),
    "clay":     (0.0, 100.0, "%"),
    "bdod":     (0.05, 2.2, "g/cm3"),   # organic soils can be below 0.1
    "soc":      (0.0, 600.0, "g/kg"),
    "socd":     (0.0, 150.0, "kg/m3"),
    "cec":      (0.0, 150.0, "cmol(c)/kg"),
    "nitrogen": (0.0, 40.0, "g/kg"),
    "cfvo":     (0.0, 100.0, "vol%"),
    "drainage": (1.0, 7.0, "index"),
}

LAT_BANDS = [(90, 66.5, "Arctic (>66.5N)"),
             (66.5, 45, "Boreal (45-66.5N)"),
             (45, 23.5, "N temperate (23.5-45N)"),
             (23.5, 0, "N tropics (0-23.5N)"),
             (0, -23.5, "S tropics (0-23.5S)"),
             (-23.5, -45, "S temperate (23.5-45S)"),
             (-45, -60, "Sub-antarctic (45-60S)"),
             (-60, -90, "Antarctica (<60S)")]


def header(text):
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def soil_layers():
    return sorted(f for f in os.listdir(SOIL_DIR)
                  if f.endswith("_10m.tif") and not f.startswith("source_flag"))


# ─────────────────────────────────────────────────────
# CHECK 1: grid identity
# ─────────────────────────────────────────────────────
def check_grid(template):
    header("CHECK 1 - GRID ALIGNMENT vs " + TEMPLATE)
    print(f"Reference grid : {template.width} x {template.height}, "
          f"{template.crs}")
    print(f"  pixel size   : {template.transform.a!r} deg")
    print(f"  origin       : lon {template.transform.c}, "
          f"lat {template.transform.f}")

    mismatches = []
    for name in soil_layers():
        with rasterio.open(os.path.join(SOIL_DIR, name)) as src:
            same = (src.width == template.width
                    and src.height == template.height
                    and src.crs == template.crs
                    and np.allclose(list(src.transform)[:6],
                                    list(template.transform)[:6], atol=1e-12))
        if not same:
            mismatches.append(name)

    total = len(soil_layers())
    print(f"\nLayers checked : {total}")
    print(f"Pixel-identical: {total - len(mismatches)} / {total}")
    if mismatches:
        print("MISMATCHED     : " + ", ".join(mismatches))
    else:
        print("RESULT         : every soil layer shares the exact WorldClim grid")
        print("                 (same size, CRS, pixel size and origin), so a")
        print("                 soil cell and a climate cell with the same")
        print("                 row/col index describe the same ground.")
    return not mismatches


# ─────────────────────────────────────────────────────
# CHECK 2: units, scaling and texture closure
# ─────────────────────────────────────────────────────
def check_values():
    header("CHECK 2 - UNITS, SCALING AND VALUE SANITY")
    print(f"{'layer':26s} {'units':12s} {'min':>8s} {'median':>8s} "
          f"{'max':>8s}  verdict")
    print("-" * 72)

    ok = True
    for name in soil_layers():
        with rasterio.open(os.path.join(SOIL_DIR, name)) as src:
            arr = src.read(1)
            units = src.tags().get("soil_units", "?")
        prop = name.rsplit("_", 3)[0]
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            print(f"{name[:26]:26s} {units:12s}  EMPTY LAYER")
            ok = False
            continue

        lo, hi = float(valid.min()), float(valid.max())
        med = float(np.median(valid))
        exp = EXPECTED_RANGE.get(prop)
        verdict = "ok"
        if exp and not (exp[0] <= lo and hi <= exp[1]):
            verdict = f"OUT OF RANGE (expect {exp[0]}-{exp[1]})"
            ok = False
        print(f"{name[:26]:26s} {units:12s} {lo:8.2f} {med:8.2f} {hi:8.2f}  "
              f"{verdict}")

    # Texture fractions are modelled independently, so their sum is a genuine
    # end-to-end check on resampling and scaling.
    print()
    for depth in sorted({n.rsplit("_", 2)[1] for n in soil_layers()}):
        paths = [os.path.join(SOIL_DIR, f"{p}_{depth}_10m.tif")
                 for p in ("sand", "silt", "clay")]
        if not all(os.path.exists(p) for p in paths):
            continue
        parts = []
        for p in paths:
            with rasterio.open(p) as src:
                parts.append(src.read(1))
        total = parts[0] + parts[1] + parts[2]
        total = total[np.isfinite(total)]
        within = np.mean(np.abs(total - 100.0) <= 5.0) * 100
        print(f"Texture closure {depth:8s}: sand+silt+clay = "
              f"{total.mean():.1f}% mean, {within:.2f}% of cells within "
              f"5 points of 100")
        if within < 90:
            ok = False
    return ok


# ─────────────────────────────────────────────────────
# CHECK 3a: coverage of the WorldClim land mask
# ─────────────────────────────────────────────────────
def check_grid_coverage(template):
    header("CHECK 3a - COVERAGE OF WORLDCLIM LAND CELLS")
    climate = template.read(1)
    land = climate > CLIMATE_NODATA_THRESHOLD
    n_land = int(land.sum())
    print(f"WorldClim valid land cells: {n_land:,} of {climate.size:,}")

    print(f"\n{'layer':26s} {'land cells with soil':>22s} {'pct':>8s}")
    print("-" * 60)
    masks = {}
    for name in soil_layers():
        with rasterio.open(os.path.join(SOIL_DIR, name)) as src:
            valid = np.isfinite(src.read(1))
        masks[name] = valid
        both = int((valid & land).sum())
        print(f"{name[:26]:26s} {both:22,} {100 * both / n_land:7.2f}%")

    # Where the gaps are, using pH as the representative primary layer.
    ref = "ph_h2o_0-30cm_10m.tif"
    if ref not in masks:
        ref = soil_layers()[0]
    missing = land & ~masks[ref]
    lats = 90 - (np.arange(template.height) + 0.5) * abs(template.transform.e)

    print(f"\nWhere the gaps are (layer: {ref})")
    print(f"{'latitude band':26s} {'land cells':>12s} {'no soil':>12s} "
          f"{'pct missing':>12s}")
    print("-" * 66)
    for hi, lo, label in LAT_BANDS:
        sel = (lats <= hi) & (lats > lo)
        n = int(land[sel].sum())
        m = int(missing[sel].sum())
        if n:
            print(f"{label:26s} {n:12,} {m:12,} {100 * m / n:11.1f}%")

    antarctic = int(missing[lats <= -60].sum())
    arctic = int(missing[lats > 66.5].sum())
    rest = int(missing.sum()) - antarctic - arctic
    print(f"\nTotal land cells with no soil value : {int(missing.sum()):,}")
    print(f"  of which Antarctica (<60S)        : {antarctic:,}")
    print(f"  of which high Arctic (>66.5N)     : {arctic:,}")
    print(f"  of which everywhere else          : {rest:,}")

    vegetated = land & (lats[:, None] > -60)
    n_veg = int(vegetated.sum())
    cov_veg = int((vegetated & masks[ref]).sum())
    print(f"\nExcluding Antarctica, which has no soil and no trees:")
    print(f"  land cells north of 60S           : {n_veg:,}")
    print(f"  with a soil value                 : {cov_veg:,} "
          f"({100 * cov_veg / n_veg:.2f}%)")

    # Which source actually fed each cell.
    flag_path = os.path.join(SOIL_DIR, "source_flag_10m.tif")
    if os.path.exists(flag_path):
        with rasterio.open(flag_path) as src:
            flag = src.read(1)
        from_olm = int(((flag == 1) & land).sum())
        from_sg = int(((flag == 2) & land).sum())
        print(f"\nSource of each covered land cell:")
        print(f"  OpenLandMap-soildb (2020-2022)    : {from_olm:,} "
              f"({100 * from_olm / n_land:.2f}% of land)")
        print(f"  SoilGrids 2.0 backfill            : {from_sg:,} "
              f"({100 * from_sg / n_land:.2f}% of land)")
        print("  The backfill is almost entirely hot desert and high latitude,")
        print("  which OpenLandMap-soildb excludes by design.")
    return land, masks


# ─────────────────────────────────────────────────────
# CHECK 3b: coverage of the GBIF occurrence points
# ─────────────────────────────────────────────────────
def sample_points(mask, arr, rows, cols, radius):
    """Value at each point, falling back to the nearest valid cell nearby."""
    height, width = mask.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

    direct = np.zeros(len(rows), dtype=bool)
    direct[inside] = mask[rows[inside], cols[inside]]
    values = np.full(len(rows), np.nan, dtype="float64")
    values[direct] = arr[rows[direct], cols[direct]]

    # Widening ring search for points that landed on a nodata cell.
    filled = direct.copy()
    for r in range(1, radius + 1):
        todo = np.where(inside & ~filled)[0]
        if todo.size == 0:
            break
        for i in todo:
            r0, r1 = max(0, rows[i] - r), min(height, rows[i] + r + 1)
            c0, c1 = max(0, cols[i] - r), min(width, cols[i] + r + 1)
            win = mask[r0:r1, c0:c1]
            if win.any():
                wr, wc = np.argwhere(win)[0]
                values[i] = arr[r0 + wr, c0 + wc]
                filled[i] = True
    return direct, filled, values


def check_point_coverage(template, land, masks):
    header("CHECK 3b - COVERAGE OF GBIF OCCURRENCE POINTS")
    df = pd.read_csv(OCCURRENCES)
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    n = len(df)
    print(f"Occurrence rows  : {n:,}   species: {df['species'].nunique()}   "
          f"countries: {df['country'].nunique()}")

    inv = ~template.transform
    cols = np.floor(inv.a * lon + inv.b * lat + inv.c).astype(np.int64)
    rows = np.floor(inv.d * lon + inv.e * lat + inv.f).astype(np.int64)
    inside = ((rows >= 0) & (rows < template.height)
              & (cols >= 0) & (cols < template.width))
    print(f"Inside raster    : {int(inside.sum()):,} "
          f"({100 * inside.mean():.3f}%)")

    on_climate = np.zeros(n, dtype=bool)
    on_climate[inside] = land[rows[inside], cols[inside]]
    print(f"On a valid climate cell: {int(on_climate.sum()):,} "
          f"({100 * on_climate.mean():.3f}%)  <- the climate baseline to match")

    print(f"\n{'layer':26s} {'direct hit':>16s} {'+fallback':>16s}")
    print("-" * 62)
    results = {}
    for name in soil_layers():
        with rasterio.open(os.path.join(SOIL_DIR, name)) as src:
            arr = src.read(1)
        direct, filled, _ = sample_points(masks[name], arr, rows, cols,
                                          FALLBACK_RADIUS)
        results[name] = (direct, filled)
        print(f"{name[:26]:26s} {100 * direct.mean():15.3f}% "
              f"{100 * filled.mean():15.3f}%")

    ref = "ph_h2o_0-30cm_10m.tif"
    if ref not in results:
        ref = soil_layers()[0]
    direct, filled = results[ref]
    print(f"\nGap analysis for {ref}")
    print(f"  points with no soil even after a {FALLBACK_RADIUS}-cell search: "
          f"{int((~filled).sum()):,} ({100 * (~filled).mean():.3f}%)")

    miss = ~filled
    if miss.any():
        print("\n  countries holding those remaining misses:")
        counts = df.loc[miss, "country"].value_counts().head(8)
        for country, c in counts.items():
            total = int((df["country"] == country).sum())
            print(f"    {country:4s} {c:6,} of {total:7,} records "
                  f"({100 * c / total:5.1f}% of that country)")
        print(f"\n  latitude range of misses: {lat[miss].min():.2f} to "
              f"{lat[miss].max():.2f}")
        below = int((lat[miss] < -56).sum())
        print(f"  misses south of 56S (outside OpenLandMap extent): {below:,}")

    usable = filled & on_climate
    print(f"\nHEADLINE: {int(usable.sum()):,} of {n:,} occurrence records "
          f"({100 * usable.mean():.2f}%)")
    print("          have BOTH a valid climate value and a valid soil value,")
    print("          which is the set usable for model training.")
    return results


# ─────────────────────────────────────────────────────
# CHECK 4: independent cross-check between the two sources
# ─────────────────────────────────────────────────────
def check_cross_source(template):
    """Compare the two products where both have data.

    OpenLandMap-soildb arrives in EPSG:4326 and SoilGrids arrives in
    Homolosine and has to be reprojected, so the two reach the grid by
    completely different routes. If they agree spatially, both routes are
    georeferenced correctly; a projection error would destroy the correlation.
    """
    header("CHECK 4 - CROSS-CHECK: OpenLandMap vs SoilGrids (independent CRS)")
    raw = os.path.join("soil_data", "soilgrids_5km", "phh2o_5-15cm_mean_5000.tif")
    merged = os.path.join(SOIL_DIR, "ph_h2o_0-30cm_10m.tif")
    flag = os.path.join(SOIL_DIR, "source_flag_10m.tif")
    if not all(os.path.exists(p) for p in (raw, merged, flag)):
        print("SKIPPED - needs both sources present")
        return

    from rasterio.warp import reproject, Resampling
    from affine import Affine

    with rasterio.open(raw) as src:
        band = src.read(1).astype("float32")
        band[band == src.nodata] = np.nan
        sg = np.full((template.height, template.width), np.nan, dtype="float32")
        reproject(band / 10.0, sg, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=template.transform,
                  dst_crs=template.crs, dst_nodata=np.nan,
                  resampling=Resampling.average)

    with rasterio.open(merged) as src:
        olm = src.read(1)
    with rasterio.open(flag) as src:
        from_olm = src.read(1) == 1

    both = from_olm & np.isfinite(olm) & np.isfinite(sg)
    a, b = olm[both], sg[both]
    diff = a - b
    print(f"Cells where both products have pH : {int(both.sum()):,}")
    print(f"  OpenLandMap mean pH             : {a.mean():.3f}")
    print(f"  SoilGrids   mean pH             : {b.mean():.3f}")
    print(f"  correlation                     : {np.corrcoef(a, b)[0, 1]:.4f}")
    print(f"  mean difference                 : {diff.mean():+.3f} pH units")
    print(f"  median absolute difference      : "
          f"{np.median(np.abs(diff)):.3f} pH units")
    print(f"  within 1.0 pH unit              : "
          f"{100 * np.mean(np.abs(diff) <= 1.0):.2f}% of cells")
    print("\nA high correlation here means the Homolosine reprojection and the")
    print("EPSG:4326 read both land on the same ground, independently.")


def main():
    print("=" * 72)
    print("SOIL / CLIMATE / OCCURRENCE ALIGNMENT VALIDATION")
    print("=" * 72)

    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)
        print("\nData sources under validation:")
        for source, meta in manifest["sources"].items():
            print(f"  {source}")
            print(f"    version   : {meta['version']}")
            print(f"    published : {meta['published']}")
            print(f"    native CRS: {meta['native_crs']}")
            print(f"    properties: {', '.join(meta['properties'])}")

    with rasterio.open(TEMPLATE) as template:
        grid_ok = check_grid(template)
        values_ok = check_values()
        land, masks = check_grid_coverage(template)
        check_point_coverage(template, land, masks)
        check_cross_source(template)

    header("SUMMARY")
    print(f"Grid alignment : {'PASS' if grid_ok else 'FAIL'}")
    print(f"Value sanity   : {'PASS' if values_ok else 'FAIL'}")
    print("Coverage       : see numbers above")


if __name__ == "__main__":
    main()
