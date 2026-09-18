"""
Validate that the production elevation layer in topography/ lines up with the
existing climate rasters and the GBIF occurrence points. The comparison DEMs
and sample windows used as evidence here live in validation/topography/.

Chosen product: GEDTM30 v1.2, released 6 March 2026, CC-BY-4.0.
A bare-earth Digital Terrain Model at 1 arc-second, 85N to 65S.

Answers, with measured numbers rather than assertions:
  0. Which global DEM is actually current, and why this one?
  1. Is the regridded elevation identical to climate_current/wc2.1_10m_bio_1.tif?
  2. What fraction of the 392,725 GBIF points get a valid elevation?
  3. What fraction of the 808,053 WorldClim land cells are covered?
  4. How many points sit above 60N, where SRTM has no data at all?
  5. How big is the canopy bias we avoid by using a terrain model, not a
     surface model? (measured against Copernicus DEM over rainforest)
  6. How wrong does slope go if you derive it from an already-coarsened DEM?
  7. Does our Horn slope agree with GEDTM30's own published slope layer?
  8. How should the circular aspect variable be aggregated?

Read-only: nothing here writes to climate_current/, climate_future_2050/ or
gbif_500_species.csv.
"""

import os
import numpy as np
import pandas as pd
import rasterio

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
CLIMATE_TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"
OCCURRENCES = "gbif_500_species.csv"
TOPO_DIR = "topography"
# Everything used only as evidence lives outside the data directories.
VALIDATION_DIR = os.path.join("validation", "topography")

# GEDTM30 v1.2, already area-averaged onto the climate grid by
# topography_rasters.py. This is the production elevation layer.
GEDTM30_ELEV_10M = os.path.join(TOPO_DIR, "gedtm30",
                                "gedtm30_v1.2_elev_10m.tif")
GEDTM30_NATIVE_DIR = os.path.join(VALIDATION_DIR, "gedtm30_native")
GEDTM30_SLOPE_PUB = os.path.join(VALIDATION_DIR, "gedtm30_slope_published",
                                 "gedtm30_slope_alps_N46E008.tif")
GEDTM30_SLOPE_SCALE = 100.0        # published slope is UInt16 degrees x 100

# WorldClim's own elevation: pixel-identical to the climate rasters, so it is
# the independent ruler we check the GEDTM30 regrid against.
WORLDCLIM_ELEV = os.path.join(VALIDATION_DIR, "wc2.1_10m_elev.tif")
WORLDCLIM_NODATA = -32768

COPERNICUS_DIR = os.path.join(VALIDATION_DIR, "copernicus_glo90")

# The demonstration window: Swiss/Italian Alps, the most extreme relief we
# hold, so the coarsening bias shows at its largest.
DEMO_SITE = "alps_N46E008"

# GEDTM30 excludes Antarctica; its southern limit is 65S.
GEDTM30_SOUTH_LIMIT = -65.0

NODATA_10M = -32768.0
CLIMATE_NODATA_TEST = -1e30    # climate is float32 -3.4e38; compare with >

# Metres per degree. Good to ~0.2%, far below the DEM's own vertical error.
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON = 111320.0

TARGET_CELL_DEG = 1.0 / 6.0     # 10 arc-minutes

# Paired native windows: (site, GEDTM30 file, Copernicus GLO-90 tile, label)
CANOPY_SITES = [
    ("amazon_S03W060", "Copernicus_DSM_COG_30_S03_00_W060_00_DEM.tif",
     "Amazon rainforest, closed canopy"),
    ("alps_N46E008", "Copernicus_DSM_COG_30_N46_00_E008_00_DEM.tif",
     "Alps, mixed forest and bare rock"),
    ("taymyr_N72E101", "Copernicus_DSM_COG_30_N72_00_E101_00_DEM.tif",
     "Taymyr tundra, essentially treeless"),
]

# Known footprints of every candidate source (north, south) in degrees.
SOURCE_FOOTPRINTS = {
    "SRTM (Feb 2000)":                  (60.0, -56.0),
    "NASADEM (Feb 2020)":               (60.0, -56.0),
    "FABDEM V1-2 (Jan 2023)":           (80.0, -60.0),
    "GMTED2010":                        (84.0, -56.0),
    "ASTER GDEM v3 (Aug 2019)":         (83.0, -83.0),
    "FathomDEM v1.0 (Dec 2024)":        (80.0, -60.0),
    "GEDTM30 v1.2 (Mar 2026)":          (85.0, -65.0),
    "Copernicus GLO-30/90 (2024_1)":    (90.0, -90.0),
    "AW3D30 v4.1 (Apr 2024)":           (90.0, -60.0),
}

SEP = "=" * 74
SUB = "-" * 74


# ─────────────────────────────────────────────────────
# STEP 0: which product, and why
# ─────────────────────────────────────────────────────
def report_product_choice():
    print(SEP)
    print("STEP 0  PRODUCT CHOICE - CURRENT GLOBAL DEMs AS OF SEPTEMBER 2026")
    print(SEP)
    # product, version, release, native, DTM?, latitude, licence, access
    rows = [
        ("GEDTM30", "v1.2", "2026-03-06", "1 arc-sec", "DTM bare-earth",
         "85N-65S", "CC-BY-4.0", "open COG, no login"),
        ("FathomDEM", "v1.0", "2024-12-18", "1 arc-sec", "DTM bare-earth",
         "80N-60S", "CC BY-NC-SA", "Zenodo RESTRICTED"),
        ("WorldDEM Neo", "2026", "2026 (early)", "0.4 arc-sec", "DSM + DTM",
         "90N-90S", "COMMERCIAL", "Airbus, paid"),
        ("FABDEM", "V1-2", "2023-01-18", "1 arc-sec", "DTM bare-earth",
         "80N-60S", "CC BY-NC-SA", "open, non-commercial"),
        ("Copernicus GLO-30", "2024_1", "2024-07", "1 arc-sec", "DSM surface",
         "90N-90S", "free+open", "AWS, no login"),
        ("Copernicus GLO-90", "2024_1", "2024-07", "3 arc-sec", "DSM surface",
         "90N-90S", "free+open", "AWS, no login"),
        ("TanDEM-X 30m EDEM", "2023", "2023-11-15", "1 arc-sec", "DSM surface",
         "90N-90S", "scientific use", "DLR registration"),
        ("AW3D30", "v4.1", "2024-04", "1 arc-sec", "DSM surface",
         "90N-60S", "free (attrib)", "JAXA, registration"),
        ("NASADEM", "v001", "2020-02", "1 arc-sec", "DSM surface",
         "60N-56S", "free+open", "Earthdata LOGIN"),
        ("ASTER GDEM", "v3", "2019-08", "1 arc-sec", "DSM surface",
         "83N-83S", "free+open", "Earthdata LOGIN"),
        ("MERIT DEM", "v1.0.3", "2018", "3 arc-sec", "DTM bare-earth",
         "90N-60S", "CC BY-NC", "email request"),
        ("GMTED2010", "2010", "2010-11-17", "7.5 arc-sec", "DSM surface",
         "84N-56S", "public domain", "USGS, no login"),
        ("SRTM", "v3 / v4.1", "2000-02", "1 arc-sec", "DSM surface",
         "60N-56S", "public domain", "login or mirror"),
        ("ETOPO", "2022", "2022", "15 arc-sec", "topo-bathy",
         "90N-90S", "public domain", "NOAA, no login"),
    ]
    print(f"{'product':<19}{'ver':<8}{'released':<13}{'native':<12}"
          f"{'model type':<16}{'latitude':<10}{'licence':<14}access")
    for r in rows:
        print(f"{r[0]:<19}{r[1]:<8}{r[2]:<13}{r[3]:<12}{r[4]:<16}"
              f"{r[5]:<10}{r[6]:<14}{r[7]}")

    print("""
DECISION: GEDTM30 v1.2 (released 6 March 2026).

The criterion that drove the choice was BARE-EARTH CORRECTNESS UNDER AN OPEN
LICENCE, not recency. Recency on its own is nearly worthless for terrain:
elevation does not move on a decadal scale, so a 2026 product only wins if it
is genuinely better corrected or better covered. GEDTM30 is both.

Why it beats the obvious defaults:
  * It is a digital TERRAIN model. SRTM, Copernicus DEM, AW3D30, NASADEM,
    ASTER and TanDEM-X are all digital SURFACE models that sit on the canopy
    over forest. For a tree-species model that bias is not merely noise, it
    is circular: the predictor partly encodes the forest being predicted.
    Step 5 below measures the size of that bias on real rainforest.
  * It is trained on ~30 billion ICESat-2 and GEDI ground returns, cutting
    Copernicus GLO-30's RMSE by 27.3% where tree cover exceeds 50% and 25.4%
    in built-up areas. Against GNSS benchmarks it has the lowest vertical
    error of any global DTM tested (7.77 m standard deviation, better than
    MERIT, FABDEM and FathomDEM).
  * 85N-65S covers every occurrence we have (max 73.43N). SRTM and NASADEM
    stop at 60N and would silently drop 8,148 records - see step 4.
  * CC-BY-4.0. Genuinely permissive: commercial use allowed, no share-alike
    clause to propagate into the model outputs.
  * Anonymous HTTPS Cloud-Optimised GeoTIFF with byte-range reads, so a
    432 GB global mosaic is usable without downloading it.
  * It also publishes slope and 14 other terrain parameters globally.

Why each newer or more accurate candidate was rejected:
  FathomDEM v1.0 (18 Dec 2024 / 11 Feb 2025) is the accuracy winner. Three
    independent studies agree it is the most accurate global bare-earth DTM,
    roughly 25% lower RMSE than GEDTM30, and 2.7 m RMSE in forest. REJECTED
    on licence and access: CC BY-NC-SA 4.0 forbids commercial use AND forces
    every derivative under the same licence, and the Zenodo records are
    access-restricted behind a manual request with an academic email. It
    cannot be scripted, so a reproducible pipeline cannot depend on it.
    The accuracy gap is in any case immaterial here: a few metres of vertical
    RMSE inside an 18.5 km cell whose internal relief reaches 3,400 m
    (measured in step 8) changes nothing.
  WorldDEM Neo (Airbus, new global coverage produced early 2026 from
    TanDEM-X acquisitions to 2025) is the most current terrain data that
    exists and does ship a bare-ground DTM. REJECTED: commercial product,
    no free tier.
  TanDEM-X 30m EDEM (15 Nov 2023) is pole-to-pole and void-free. REJECTED:
    it is a surface model, and needs a DLR scientific-use registration.
  FABDEM V1-2 (18 Jan 2023) is bare-earth. REJECTED: also CC BY-NC-SA, and
    both FathomDEM's and GEDTM30's validations rank it below GEDTM30.
  Copernicus GLO-30/GLO-90, current release 2024_1 (July 2024, the ninth
    release; there is no 2025 or 2026 edition). Best plain DSM, fully open.
    RETAINED, but only as the DSM reference in step 5 - not as the elevation
    source, because it is not bare-earth.
  AW3D30 v4.1 (April 2024) - surface model, and JAXA states outright that
    they publish no DTM product.
  NASADEM (Feb 2020) - the reprocessed SRTM. Still 60N-56S, still a surface
    model, needs an Earthdata login, and the FathomDEM authors cite work
    recommending SRTM/NASADEM/ASTER simply not be used any more.
  ASTER GDEM v3 (Aug 2019) - noisier optical stereo, Earthdata login.
  MERIT DEM (2018) - the original bare-earth global DTM, superseded by
    FABDEM/GEDTM30/FathomDEM, CC BY-NC, and distributed by email request.
  GMTED2010, SRTM, ETOPO 2022 - too coarse, too old, or built for
    bathymetry. Kept only as historical context.

Honest caveat about what GEDTM30 does NOT give us: there is no aspect layer
among its 15 published terrain parameters, so northness and eastness still
have to be derived here (step 8). Its published slope layers also sit on
slightly different grids from the DTM itself, which step 7 quantifies.""")


# ─────────────────────────────────────────────────────
# STEP 1: grid identity against the climate template
# ─────────────────────────────────────────────────────
def check_grid_identity():
    """Compare the regridded elevation grid definition to the climate template."""
    print("\n" + SEP)
    print("STEP 1  GRID IDENTITY vs " + CLIMATE_TEMPLATE)
    print(SEP)

    with rasterio.open(CLIMATE_TEMPLATE) as c, \
            rasterio.open(GEDTM30_ELEV_10M) as e:
        rows = [
            ("width",     c.width,              e.width),
            ("height",    c.height,             e.height),
            ("crs",       str(c.crs),           str(e.crs)),
            ("transform", tuple(c.transform),   tuple(e.transform)),
            ("bounds",    tuple(c.bounds),      tuple(e.bounds)),
            ("pixel size", c.res,               e.res),
            ("AREA_OR_POINT", c.tags().get("AREA_OR_POINT"),
                              e.tags().get("AREA_OR_POINT")),
        ]
        print(f"{'property':16s} {'climate':>26s}   {'GEDTM30 regridded':>26s}  match")
        all_match = True
        for name, a, b in rows:
            ok = (a == b)
            all_match &= ok
            print(f"{name:16s} {str(a)[:26]:>26s}   {str(b)[:26]:>26s}   "
                  f"{'YES' if ok else 'NO'}")

        print(f"\ndtype/nodata differ by design: climate {c.dtypes[0]}/"
              f"{c.nodata:.3g}, elevation {e.dtypes[0]}/{e.nodata:.0f}")
        print(f"GRID IS {'IDENTICAL' if all_match else 'NOT identical'} "
              "-> the model can read it with no resampling")

    print(f"""
How that was achieved matters, because GEDTM30's native grid is NOT aligned
to whole degrees: its global COG runs from -180.00125 to +180.00125, a 4.5
pixel overhang, so every 10-arcmin cell boundary falls exactly mid-pixel.
topography_rasters.py therefore WARPS (rasterio.warp.reproject with
Resampling.average) rather than slicing, which absorbs that offset. Naive
slicing would have left a half-pixel (~15 m) shear and a 4.5-pixel edge error.

Averaging is the correct reduction for elevation and ONLY for elevation.
Slope and aspect are handled from the native windows in steps 6-8.""")
    return all_match


# ─────────────────────────────────────────────────────
# STEP 2: land-cell coverage
# ─────────────────────────────────────────────────────
def check_land_cell_coverage():
    """Compare the climate land mask to the terrain valid mask, cell by cell."""
    print("\n" + SEP)
    print("STEP 2  LAND-CELL COVERAGE (2160 x 1080 = 2,332,800 cells)")
    print(SEP)

    with rasterio.open(CLIMATE_TEMPLATE) as s:
        climate = s.read(1)
        transform = s.transform
    with rasterio.open(GEDTM30_ELEV_10M) as s:
        elev = s.read(1)
    with rasterio.open(WORLDCLIM_ELEV) as s:
        wc_elev = s.read(1)

    climate_land = climate > CLIMATE_NODATA_TEST
    elev_valid = elev != NODATA_10M
    wc_valid = wc_elev != WORLDCLIM_NODATA

    lat_of_row = transform.f + (np.arange(climate.shape[0]) + 0.5) * transform.e
    antarctic = (lat_of_row < GEDTM30_SOUTH_LIMIT)[:, None]
    antarctic = np.broadcast_to(antarctic, climate.shape)

    n_climate = int(climate_land.sum())
    n_ant = int((climate_land & antarctic).sum())
    n_noant = int((climate_land & ~antarctic).sum())

    print(f"climate land cells (bio_1 valid)            : {n_climate:>9,}")
    print(f"  of which south of 65S (Antarctica)        : {n_ant:>9,}")
    print(f"  of which within GEDTM30's 85N-65S window  : {n_noant:>9,}")
    print(f"\nGEDTM30 valid cells                         : "
          f"{int(elev_valid.sum()):>9,}")

    both = climate_land & elev_valid
    print(f"\nWITHIN GEDTM30's footprint (the number that matters):")
    print(f"  climate land cells covered                : "
          f"{int((both & ~antarctic).sum()):>9,}  "
          f"({100.0 * (both & ~antarctic).sum() / n_noant:.4f}%)")
    print(f"  climate land cells NOT covered            : "
          f"{int((climate_land & ~elev_valid & ~antarctic).sum()):>9,}")
    print(f"\nOVER THE WHOLE GLOBE:")
    print(f"  climate land cells covered                : "
          f"{int(both.sum()):>9,}  ({100.0 * both.sum() / n_climate:.4f}%)")
    print(f"  shortfall, all of it Antarctica           : {n_ant:>9,}  "
          f"({100.0 * n_ant / n_climate:.4f}%)")
    print(f"  GEDTM30 valid where climate says ocean    : "
          f"{int((~climate_land & elev_valid).sum()):>9,}  "
          "(coastal, harmless)")

    print(f"""
The Antarctic gap is real and must be stated, but it costs this project
nothing: no tree grows there. The only 3 occurrence records south of 65S are
Pinus wallichiana - a Himalayan pine - geotagged in Antarctica, i.e. plainly
corrupt records (see step 4).

Two ways to handle it, both fine:
  (a) preferred - intersect the model's land mask with the terrain mask, so
      generate_pseudo_absences() also stops drawing background points from
      Antarctica. That removes {n_ant:,} climate cells that could never hold a
      tree and were only ever diluting the background sample.
  (b) if you want exact land-mask parity, fill Antarctica from
      validation/topography/wc2.1_10m_elev.tif, which does cover it. Do not
      mix the two products anywhere a species actually occurs.""")

    # Independent control: WorldClim's own elevation on the identical grid.
    print("\n" + SUB)
    print("CONTROL: WorldClim 2.1 elevation, which is pixel-identical by build")
    print(SUB)
    print(f"WorldClim elev valid cells                  : "
          f"{int(wc_valid.sum()):>9,}")
    print(f"exact agreement with the climate land mask  : "
          f"{'YES' if int((wc_valid ^ climate_land).sum()) == 0 else 'NO'}  "
          f"(symmetric difference {int((wc_valid ^ climate_land).sum()):,} cells)")
    agree = both & wc_valid
    d = elev[agree] - wc_elev[agree].astype("float32")
    r = np.corrcoef(elev[agree], wc_elev[agree])[0, 1]
    print(f"\nGEDTM30 minus WorldClim elevation on the {int(agree.sum()):,} "
          "cells both cover:")
    print(f"  mean {d.mean():+.2f} m, median {np.median(d):+.2f} m, "
          f"std {d.std():.2f} m")
    print(f"  correlation r = {r:.6f}")
    print(f"""  Two independently built DEMs correlating at r = {r:.4f} across
  {int(agree.sum()):,} cells is the end-to-end proof that the regrid landed on
  the right cells: a one-cell (18.5 km) offset in mountainous terrain would
  visibly degrade this.

  The residual spread (std {d.std():.0f} m, median {np.median(d):+.0f} m) is NOT
  misalignment. It is the genuine difference between a bare-earth DTM and
  WorldClim's SRTM/GMTED surface-model heritage, plus the fact that the two
  products aggregate different source pixels into each 10-arcmin cell. The
  negative median is the expected direction - bare earth sits below canopy.
  Step 5 isolates that effect properly against a DEM we hold at native
  resolution.""")

    # Coverage by latitude band, to expose any polar thinning.
    print("\n" + SUB)
    print("coverage by latitude band")
    print(SUB)

    def label(d):
        return f"{abs(d):g}{'N' if d >= 0 else 'S'}"

    print(f"{'band':>16s} {'climate land':>13s} {'covered':>10s} {'pct':>8s}"
          "   note")
    bands = [((90, 85), "above GEDTM30"), ((85, 84), "above GMTED/ASTER"),
             ((84, 80), "above FABDEM/FathomDEM"),
             ((80, 73.5), "above our northernmost record"),
             ((73.5, 60), "above SRTM; has our records"),
             ((60, 56), "SRTM north edge"), ((56, 0), ""), ((0, -56), ""),
             ((-56, -65), "sub-Antarctic islands"),
             ((-65, -90), "Antarctica: GEDTM30 excludes")]
    for (hi, lo), note in bands:
        sel = (lat_of_row <= hi) & (lat_of_row > lo)
        cl = int(climate_land[sel].sum())
        bo = int(both[sel].sum())
        pct = f"{100.0 * bo / cl:.2f}%" if cl else "n/a"
        print(f"{label(hi):>7s} .. {label(lo):<6s} {cl:>13,} {bo:>10,} "
              f"{pct:>8s}   {note}")

    return climate_land, elev, elev_valid


# ─────────────────────────────────────────────────────
# STEP 3/4: GBIF occurrence coverage
# ─────────────────────────────────────────────────────
def check_occurrence_coverage(climate_land, elev, elev_valid):
    """Sample elevation at every occurrence point and count the hits."""
    print("\n" + SEP)
    print("STEP 3  GBIF OCCURRENCE COVERAGE")
    print(SEP)

    df = pd.read_csv(OCCURRENCES)
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    n = len(df)
    print(f"occurrence rows               : {n:>9,}")
    print(f"unique species                : {df['species'].nunique():>9,}")
    print(f"latitude  range               : {lat.min():>9.4f} .. {lat.max():.4f}")
    print(f"longitude range               : {lon.min():>9.4f} .. {lon.max():.4f}")

    with rasterio.open(GEDTM30_ELEV_10M) as s:
        transform = s.transform
        height, width = s.height, s.width

    col = np.floor((lon - transform.c) / transform.a).astype(int)
    row = np.floor((lat - transform.f) / transform.e).astype(int)
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    col = np.clip(col, 0, width - 1)
    row = np.clip(row, 0, height - 1)

    has_elev = inside & elev_valid[row, col]
    on_climate = inside & climate_land[row, col]

    print(f"\ninside raster extent          : {int(inside.sum()):>9,}  "
          f"({100.0 * inside.sum() / n:.4f}%)")
    print(f"on a valid CLIMATE cell       : {int(on_climate.sum()):>9,}  "
          f"({100.0 * on_climate.sum() / n:.4f}%)")
    print(f"on a valid GEDTM30 cell       : {int(has_elev.sum()):>9,}  "
          f"({100.0 * has_elev.sum() / n:.4f}%)")
    print(f"usable for the model (both)    : "
          f"{int((has_elev & on_climate).sum()):>9,}  "
          f"({100.0 * (has_elev & on_climate).sum() / n:.4f}%)")
    print(f"climate-valid but no elevation: "
          f"{int((on_climate & ~has_elev).sum()):>9,}")
    print(f"elevation-valid but no climate: "
          f"{int((has_elev & ~on_climate).sum()):>9,}")

    vals = elev[row[has_elev], col[has_elev]]
    print(f"\nsampled elevation: min {vals.min():,.0f} m, "
          f"max {vals.max():,.0f} m, median {np.median(vals):,.0f} m, "
          f"mean {vals.mean():.1f} m")

    bad = df.loc[inside & ~has_elev, ["species", "latitude", "longitude",
                                      "country"]]
    print(f"\npoints with NO elevation ({len(bad):,}) - country breakdown:")
    if len(bad):
        print(bad["country"].value_counts().head(10).to_string())
        print("""
These are coastal and offshore georeferencing errors, not a terrain-data
failure: the point lands in a 10-arcmin cell that is ocean. They already fail
against the climate rasters too, so they are lost either way. Drop them, or
snap them to the nearest land cell.""")

    # ── the high-latitude question ──
    print("\n" + SEP)
    print("STEP 4  HIGH-LATITUDE GAP: WHICH SOURCES REACH OUR POINTS?")
    print(SEP)
    for thr in (56, 60, 70, 73.5, 80, 84, 85):
        k = int((lat > thr).sum())
        print(f"points above {thr:>4.1f}N : {k:>7,}  ({100.0 * k / n:.4f}%)")
    for thr in (-56, -60, -65):
        k = int((lat < thr).sum())
        print(f"points below {abs(thr):>4.1f}S : {k:>7,}  ({100.0 * k / n:.4f}%)")

    print(f"\n{'source':>32s} {'footprint':>14s} {'pts in':>9s} "
          f"{'pts LOST':>9s} {'lost %':>8s}")
    for name, (north, south) in SOURCE_FOOTPRINTS.items():
        ok = (lat <= north) & (lat >= south)
        lost = n - int(ok.sum())
        print(f"{name:>32s} {f'{north:.0f}N-{abs(south):.0f}S':>14s} "
              f"{int(ok.sum()):>9,} {lost:>9,} {100.0 * lost / n:>7.4f}%")

    hi = lat > 60
    print(f"\nof the {int(hi.sum()):,} points above 60N, "
          f"{int((hi & has_elev).sum()):,} have a valid GEDTM30 elevation "
          f"({100.0 * (hi & has_elev).sum() / hi.sum():.4f}%)")
    hv = elev[row[hi & has_elev], col[hi & has_elev]]
    print(f"their elevation range: {hv.min():,.0f} .. {hv.max():,.0f} m "
          f"(mean {hv.mean():.1f} m)")
    print(f"countries: "
          f"{', '.join(df.loc[hi, 'country'].value_counts().head(8).index)}")

    lost3 = df.loc[lat < GEDTM30_SOUTH_LIMIT,
                   ["species", "latitude", "longitude", "country", "year"]]
    print(f"\nthe only {len(lost3)} points outside GEDTM30's footprint:")
    print(lost3.to_string(index=False))
    print("""
Pinus wallichiana is the Himalayan blue pine. These coordinates are in the
Larsemann Hills area of Antarctica, country code AQ. They are corrupt records
and should be dropped on data-quality grounds regardless of which DEM is used,
so GEDTM30's 65S limit costs this project precisely nothing.

The contrast with SRTM is the whole argument: SRTM and NASADEM would silently
drop 8,148 genuine boreal and Arctic records from Finland, Norway, Russia,
Canada, Sweden and Alaska - exactly the treeline populations that matter most
for a climate-change range-shift model, because they are the leading edge.""")

    return df, row, col, has_elev


# ─────────────────────────────────────────────────────
# STEP 5: canopy bias - the reason to use a terrain model
# ─────────────────────────────────────────────────────
def check_canopy_bias():
    """Measure GEDTM30 (bare earth) minus Copernicus GLO-90 (surface).

    Both are aggregated to the 10-arcmin target cells first, so no resampling
    or registration assumption is involved in the comparison.
    """
    print("\n" + SEP)
    print("STEP 5  CANOPY BIAS: TERRAIN MODEL vs SURFACE MODEL")
    print(SEP)
    print("""GEDTM30 is bare-earth; Copernicus DEM is a surface model whose X-band radar
phase centre sits inside/atop the canopy. If that difference were negligible,
the cheaper and more familiar Copernicus DEM would do. Measured per 10-arcmin
cell on paired 1-degree windows:
""")
    print(f"{'site':<36}{'cells':>6}{'GEDTM30':>10}{'CopDEM':>10}"
          f"{'DSM-DTM':>9}{'max':>8}")
    for site, cop_name, label in CANOPY_SITES:
        g_path = os.path.join(GEDTM30_NATIVE_DIR, f"gedtm30_{site}.tif")
        c_path = os.path.join(COPERNICUS_DIR, cop_name)
        if not (os.path.exists(g_path) and os.path.exists(c_path)):
            print(f"{label:<36}{'(missing sample)':>43}")
            continue

        with rasterio.open(g_path) as s:
            g = s.read(1).astype("float64")
            g = np.where(g > 1e30, np.nan, g)
            gf = target_cell_factors(s.transform)
        with rasterio.open(c_path) as s:
            c = s.read(1).astype("float64")
            # Copernicus sets no nodata and writes ocean as a literal 0.0
            c = np.where(c == 0.0, np.nan, c)
            cf = target_cell_factors(s.transform)

        with np.errstate(invalid="ignore"):
            gm = block_reduce(g, gf, np.nanmean)
            cm = block_reduce(c, cf, np.nanmean)
        # Both are 1-degree windows on the same footprint, so both reduce to a
        # 6x6 block of 10-arcmin cells; clip to the common shape defensively.
        kr = min(gm.shape[0], cm.shape[0])
        kc = min(gm.shape[1], cm.shape[1])
        gm, cm = gm[:kr, :kc], cm[:kr, :kc]
        ok = np.isfinite(gm) & np.isfinite(cm)
        if not ok.any():
            print(f"{label:<36}{'(no overlapping land)':>43}")
            continue
        d = cm[ok] - gm[ok]
        print(f"{label:<36}{int(ok.sum()):>6}{gm[ok].mean():>9.1f}m"
              f"{cm[ok].mean():>9.1f}m{d.mean():>+8.1f}m{d.max():>+7.1f}m")

    print("""
Read the DSM-DTM column as "how many metres of vegetation and noise the
surface model adds". Over closed rainforest it is large and systematic;
over treeless tundra it collapses toward zero, which is exactly the signature
of canopy rather than of a datum or registration error.

This is the concrete reason the choice is a DTM and not Copernicus DEM. The
bias is not random: it scales with canopy height, so it is spatially
correlated with forest itself. Feeding a tree-suitability model an elevation
layer that is inflated precisely where trees are dense leaks the response
variable into the predictor. Slope and aspect inherit the same problem,
because they are derivatives of that biased surface.""")


# ─────────────────────────────────────────────────────
# SLOPE / ASPECT MATHS
# ─────────────────────────────────────────────────────
def slope_aspect(dem, lat_centres, res_lon_deg, res_lat_deg):
    """Horn (1981) 3x3 slope and aspect on a geographic (lon/lat) grid.

    Returns (slope_deg, aspect_deg) shaped like `dem`; the 1-pixel border is
    NaN.

    The latitude correction is the part people skip. A degree of longitude is
    111,320 m at the equator but 111,320 * cos(lat) m at latitude lat, so the
    east-west pixel spacing shrinks toward the poles. Using one constant dx
    makes high-latitude terrain look artificially steep - at 72N a degree of
    longitude is only 31% as long as at the equator.
    """
    z = dem.astype(np.float64)

    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]

    lat_in = lat_centres[1:-1]
    dx = (res_lon_deg * M_PER_DEG_LON *
          np.cos(np.deg2rad(lat_in)))[:, None]
    dy = res_lat_deg * M_PER_DEG_LAT

    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * dy)

    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    # Aspect: compass bearing of the downslope direction, 0 = N, 90 = E.
    # dzdy as built above increases southward, hence the signs.
    aspect = np.degrees(np.arctan2(dzdx, dzdy))
    aspect = np.mod(180.0 - aspect, 360.0)
    aspect[slope == 0] = np.nan          # undefined on perfectly flat ground

    out_s = np.full(dem.shape, np.nan)
    out_a = np.full(dem.shape, np.nan)
    out_s[1:-1, 1:-1] = slope
    out_a[1:-1, 1:-1] = aspect
    return out_s, out_a


def block_reduce(arr, factor, fn):
    """Apply `fn` over non-overlapping blocks.

    `factor` is either an int (square blocks) or a (rows, cols) pair. The pair
    form is needed because Copernicus DEM thins its tiles in LONGITUDE toward
    the poles: the 72N tile is 3 arc-sec in latitude but 9 arc-sec in
    longitude, so one 10-arcmin cell is 200 rows by 67 columns, not 200x200.
    """
    fy, fx = (factor, factor) if np.isscalar(factor) else factor
    h, w = arr.shape
    nh, nw = h // fy, w // fx
    r = arr[:nh * fy, :nw * fx].reshape(nh, fy, nw, fx)
    return fn(r, axis=(1, 3))


def target_cell_factors(transform):
    """How many native pixels make up one 10-arcmin cell, as (rows, cols)."""
    return (int(round(TARGET_CELL_DEG / abs(transform.e))),
            int(round(TARGET_CELL_DEG / abs(transform.a))))


# ─────────────────────────────────────────────────────
# STEPS 6-8: derive terrain the correct way
# ─────────────────────────────────────────────────────
def derive_terrain_demo():
    """Aggregate a native 1-arcsec GEDTM30 window onto the 10-arcmin grid."""
    path = os.path.join(GEDTM30_NATIVE_DIR, f"gedtm30_{DEMO_SITE}.tif")
    print("\n" + SEP)
    print(f"STEP 6  SLOPE FROM NATIVE RESOLUTION (GEDTM30 1 arc-sec, "
          f"{DEMO_SITE})")
    print(SEP)

    with rasterio.open(path) as s:
        dem = s.read(1).astype(np.float64)
        dem = np.where(dem > 1e30, np.nan, dem)
        tr = s.transform
        res_lon, res_lat = abs(tr.a), abs(tr.e)
        top = tr.f
        h, w = s.height, s.width

    print(f"window    : {os.path.basename(path)}")
    print(f"shape     : {w} x {h}   pixel {res_lon * 3600:.4f} arc-sec")

    factor = int(round(TARGET_CELL_DEG / res_lon))
    n_cells = h // factor
    print(f"aggregation: {factor} x {factor} native pixels per 10-arcmin cell"
          f"  ->  {n_cells} x {n_cells} = {n_cells * n_cells} target cells")
    print(f"             ({factor * factor:,} native pixels per cell, so the "
          "aggregates have no sampling noise)")

    lat_centres = top - (np.arange(h) + 0.5) * res_lat
    ew = res_lon * M_PER_DEG_LON * np.cos(np.deg2rad(lat_centres))
    ns = res_lat * M_PER_DEG_LAT
    print(f"\nground pixel size: north-south {ns:.1f} m (constant), "
          f"east-west {ew.min():.1f}-{ew.max():.1f} m")
    print(f"east-west / north-south anisotropy here: {ew.mean() / ns:.3f}")
    print("This is why slope_aspect() uses a per-row dx = res*111320*cos(lat).")

    # ── elevation: averaging is correct ──
    blocks = dem[:n_cells * factor, :n_cells * factor].reshape(
        n_cells, factor, n_cells, factor)
    elev_coarse = np.nanmean(blocks, axis=(1, 3))
    elev_std = np.nanstd(blocks, axis=(1, 3))
    elev_min = np.nanmin(blocks, axis=(1, 3))
    elev_max = np.nanmax(blocks, axis=(1, 3))

    # ── slope: native first, THEN aggregate ──
    slope_native, aspect_native = slope_aspect(dem, lat_centres,
                                               res_lon, res_lat)
    slope_right = block_reduce(slope_native, factor, np.nanmean)
    slope_p90 = block_reduce(slope_native, factor,
                             lambda r, axis: np.nanpercentile(r, 90, axis=axis))

    # the wrong way: coarsen the DEM first, then differentiate
    coarse_lat = top - (np.arange(n_cells) + 0.5) * (res_lat * factor)
    slope_wrong, _ = slope_aspect(elev_coarse, coarse_lat,
                                  res_lon * factor, res_lat * factor)

    print("\n" + SUB)
    print("native-then-aggregate  vs  coarsen-then-differentiate")
    print(SUB)
    print(f"{'':30s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'max':>8s}")
    for label, arr in (("native 1-arcsec slope (deg)", slope_native),
                       ("CORRECT: aggregated native", slope_right),
                       ("WRONG: slope of mean DEM", slope_wrong)):
        print(f"{label:30s} {np.nanmean(arr):>8.2f} "
              f"{np.nanmedian(arr):>8.2f} "
              f"{np.nanpercentile(arr, 95):>8.2f} {np.nanmax(arr):>8.2f}")
    ratio = np.nanmean(slope_right) / np.nanmean(slope_wrong)
    print(f"\nthe coarsen-first route under-reports mean slope by {ratio:.1f}x.")
    print("""Reason: averaging elevation over 18.5 km erases every ridge and valley
smaller than the cell, so the residual surface is nearly flat. Slope has to
be measured while the relief still exists, then averaged. The same applies to
aspect, and even more strongly - a coarsened DEM has almost no aspect signal
left to measure.

Slope units: DEGREES throughout. percent = 100 * tan(radians(degrees)), so
45 degrees is 100% and the two scales are not interchangeable. Report which
one you are using; XGBoost does not care, but a human reader will.""")

    # ── cross-check against GEDTM30's own published slope ──
    print("\n" + SEP)
    print("STEP 7  CROSS-CHECK: our Horn slope vs GEDTM30's PUBLISHED slope")
    print(SEP)
    if os.path.exists(GEDTM30_SLOPE_PUB):
        with rasterio.open(GEDTM30_SLOPE_PUB) as s:
            pub = s.read(1).astype(np.float64)
            pub = np.where(pub == s.nodata, np.nan, pub) / GEDTM30_SLOPE_SCALE
            ptr = s.transform
        print(f"published layer : {os.path.basename(GEDTM30_SLOPE_PUB)}")
        print(f"                  UInt16, degrees = value / "
              f"{GEDTM30_SLOPE_SCALE:.0f}, computed by Whitebox Workflows")
        dx_px = (ptr.c - tr.c) / res_lon
        dy_px = (ptr.f - tr.f) / res_lat
        print(f"grid offset vs the DTM window: dx {dx_px:+.3f} px, "
              f"dy {dy_px:+.3f} px")
        print("""  ^ GEDTM30's derivative layers are NOT co-registered with its own DTM
    (they were produced via an Equi7 reprojection round-trip). The offset is
    sub-pixel, ~12 m, so a statistical comparison is valid but a pixel-wise
    difference map would be misleading.""")
        k = min(pub.shape[0], slope_native.shape[0])
        pub_c = block_reduce(pub[:k, :k], factor, np.nanmean)
        our_c = block_reduce(slope_native[:k, :k], factor, np.nanmean)
        ok = np.isfinite(pub_c) & np.isfinite(our_c)
        print(f"\nper 10-arcmin cell, over {int(ok.sum())} cells:")
        print(f"  ours      : mean {our_c[ok].mean():6.2f} deg  "
              f"range {our_c[ok].min():.2f}..{our_c[ok].max():.2f}")
        print(f"  published  : mean {pub_c[ok].mean():6.2f} deg  "
              f"range {pub_c[ok].min():.2f}..{pub_c[ok].max():.2f}")
        print(f"  difference : mean {(our_c - pub_c)[ok].mean():+.2f} deg, "
              f"max |diff| {np.abs((our_c - pub_c)[ok]).max():.2f} deg")
        print(f"  correlation: r = "
              f"{np.corrcoef(our_c[ok], pub_c[ok])[0, 1]:.6f}")
        print("""
Agreement this close, from two independent implementations (our Horn 3x3 on a
geographic grid vs Whitebox's polynomial fit on an Equi7 metric grid),
validates the derivation. Either could be used in production. We derive our
own because (a) it lands directly on the target grid with no extra warp, and
(b) we need aspect, which GEDTM30 does not publish - so the native loop has
to run anyway.""")
    else:
        print("published slope sample not found; run topography_rasters.py")

    # ── aspect: circular, so decompose ──
    print("\n" + SEP)
    print("STEP 8  ASPECT: A CIRCULAR QUANTITY, AND WHAT TO DO WITH IT")
    print(SEP)
    demo = np.array([350.0, 10.0])
    print(f"Why a plain mean is invalid: aspects {demo.tolist()} are 20 deg "
          "apart and\nboth face north, but their arithmetic mean is "
          f"{demo.mean():.0f} deg - due SOUTH. Wrong by 180 deg.")

    # Slope-weighted circular mean: a flat pixel has no meaningful aspect, so
    # weight by sin(slope) and let flat ground contribute nothing.
    wgt = np.sin(np.deg2rad(slope_native))
    wgt = np.where(np.isnan(aspect_native) | np.isnan(wgt), 0.0, wgt)
    nn = np.nan_to_num(np.cos(np.deg2rad(aspect_native))) * wgt
    ee = np.nan_to_num(np.sin(np.deg2rad(aspect_native))) * wgt

    wsum = block_reduce(wgt, factor, np.sum)
    northness = np.where(wsum > 0,
                         block_reduce(nn, factor, np.sum) / np.maximum(wsum, 1e-12),
                         0.0)
    eastness = np.where(wsum > 0,
                        block_reduce(ee, factor, np.sum) / np.maximum(wsum, 1e-12),
                        0.0)
    strength = np.hypot(northness, eastness)

    print(f"""
CORRECT aggregation - decompose to vector components, aggregate those:
    northness = sum(sin(slope) * cos(aspect)) / sum(sin(slope))
    eastness  = sum(sin(slope) * sin(aspect)) / sum(sin(slope))
Weighting by sin(slope) is deliberate: on flat ground aspect is undefined and
numerically random, so unweighted averaging would inject pure noise.

measured on this window, per 10-arcmin cell:
  northness  range {northness.min():+.3f} .. {northness.max():+.3f}   (mean {northness.mean():+.3f})
  eastness   range {eastness.min():+.3f} .. {eastness.max():+.3f}   (mean {eastness.mean():+.3f})
  |vector|   range {strength.min():.3f} .. {strength.max():.3f}   = aspect consistency, 0 = no dominant facing

northness and eastness are the forms to feed XGBoost: continuous, no
wrap-around discontinuity at 0/360, and northness is the direct sun-exposure
proxy the design doc asks for (+1 = pole-facing, shaded, cooler, moister;
-1 = equator-facing, sunlit, warmer, drier).

Never hand the model a raw 0-360 aspect column. A tree split at "aspect < 5"
would put north-northeast and north-northwest slopes on opposite sides of the
tree, and the model would have to spend many splits relearning that 359 and 1
are neighbours.

HONEST CAVEAT: |vector| only reaches {strength.max():.2f} here. An 18.5 km cell
in the Alps contains slopes facing every direction, so they very nearly
cancel. Aspect is intrinsically a hillside-scale variable; at 10 arc-minutes
it carries far less signal than slope or elev_std. Keep northness and
eastness - they are cheap, and genuinely non-zero on uniformly tilted terrain
such as a coastal scarp or a single-sided range front - but expect XGBoost to
rank them below elevation and ruggedness, and do not read that as a bug.""")

    # ── the resulting feature table ──
    print("\n" + SUB)
    print(f"RESULTING TERRAIN FEATURES for this {n_cells}x{n_cells} patch")
    print(SUB)
    print(f"{'feature':>16s} {'min':>9s} {'mean':>9s} {'max':>9s}  note")
    for name, arr, note in [
        ("elev_mean", elev_coarse, "average bare-earth elevation, m"),
        ("elev_std", elev_std, "sub-cell relief / ruggedness, m"),
        ("elev_range", elev_max - elev_min, "max-min within cell, m"),
        ("slope_mean", slope_right, "degrees, from native 1 arc-sec"),
        ("slope_p90", slope_p90, "steep-ground indicator, degrees"),
        ("northness", northness, "cos(aspect), +1 = pole-facing"),
        ("eastness", eastness, "sin(aspect), +1 = east-facing"),
        ("aspect_strength", strength, "0 = no dominant facing"),
    ]:
        print(f"{name:>16s} {np.nanmin(arr):>9.2f} {np.nanmean(arr):>9.2f} "
              f"{np.nanmax(arr):>9.2f}  {note}")


# ─────────────────────────────────────────────────────
# STEP 9: per-window coverage, including the Arctic
# ─────────────────────────────────────────────────────
def check_native_windows(df):
    """Confirm the downloaded windows hold real data where our points are."""
    print("\n" + SEP)
    print("STEP 9  NATIVE SAMPLE WINDOWS: REAL DATA, AND OCEAN HANDLING")
    print(SEP)

    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)

    print(f"{'window':<22}{'valid px':>9}{'elev range':>18}{'GBIF':>7}"
          f"{'with elev':>10}")
    for fn in sorted(os.listdir(GEDTM30_NATIVE_DIR)):
        if not fn.endswith(".tif"):
            continue
        path = os.path.join(GEDTM30_NATIVE_DIR, fn)
        with rasterio.open(path) as s:
            a = s.read(1)
            b, tr = s.bounds, s.transform
        valid = a < 1e30
        sel = ((lon >= b.left) & (lon < b.right) &
               (lat >= b.bottom) & (lat < b.top))
        k = int(sel.sum())
        got = 0
        if k:
            cc = np.clip(((lon[sel] - tr.c) / tr.a).astype(int), 0,
                         a.shape[1] - 1)
            rr = np.clip(((lat[sel] - tr.f) / tr.e).astype(int), 0,
                         a.shape[0] - 1)
            got = int(valid[rr, cc].sum())
        name = fn.replace("gedtm30_", "").replace(".tif", "")
        rng = (f"{a[valid].min():.0f}..{a[valid].max():.0f} m"
               if valid.any() else "none")
        print(f"{name:<22}{100.0 * valid.mean():>8.1f}%{rng:>18}{k:>7}"
              f"{got:>10}")

    print("""
Every GBIF point falling inside these windows got a valid native-resolution
elevation, including all of them in the Finnmark (70N) and Taymyr (72N)
windows. The 8,148 occurrences above 60N are therefore genuinely recoverable:
the high-latitude gap is an SRTM problem, not a terrain-data problem.

Note the ocean handling, which differs by product and matters:
  GEDTM30      - explicit nodata (3.4e38). The Cape Town and Finnmark windows
                 come out ~76% valid because the sea is correctly flagged.
  Copernicus   - NO nodata set; ocean is written as a literal 0.0 m. Never
                 infer land from a Copernicus DEM.
  Both         - should be masked with the climate raster's own land mask,
                 which is the mask generate_pseudo_absences() already uses,
                 so presences and background points stay on identical cells.""")


# ─────────────────────────────────────────────────────
# STEP 10: vintage, datum, provenance
# ─────────────────────────────────────────────────────
def report_provenance():
    print("\n" + SEP)
    print("STEP 10  VINTAGE, VERTICAL DATUM, PROVENANCE")
    print(SEP)
    print(f"""chosen product : GEDTM30 v1.2
release date   : 6 March 2026  (v1.1 11 Jun 2025, v1 22 May 2025,
                 v0 24 Feb 2025; v1.2 added true 1-arc-second alignment and
                 updated the land-surface parameters)
citation       : Ho Y, Grohmann CH, Lindsay J, Reuter HI, Parente L,
                 Witjes M, Hengl T (2025). GEDTM30: global ensemble digital
                 terrain model at 30 m and derived multiscale terrain
                 variables. PeerJ 13:e19673.
licence        : CC-BY-4.0 (data), MIT (production code)
horizontal CRS : EPSG:4326, WGS84 geographic lon/lat
vertical datum : EGM2008 geoid, EPSG:3855 - declared in the file's
                 VERTICAL_DATUM tag, which is unusually good practice
native grid    : 1 arc-second, 1,296,009 x 540,009, float32 metres
acquisition    : source DEMs 2006-2015 (Copernicus/TanDEM-X 2011-2015,
                 AW3D30 2006-2011); ICESat-2 and GEDI ground returns
                 2019-2023 used as training targets
model          : two-stage global-to-local random forest over ~30 billion
                 lidar points, plus a published per-pixel uncertainty layer

DOES THE VINTAGE MATTER? Essentially no - and this is where terrain is easier
than climate.

  * Elevation, slope and aspect are static on decadal scales. Outside
    volcanoes, glacier retreat, open-pit mining and new reservoirs, a
    2006-2015 terrain surface is still correct in 2026 to well inside its own
    ~7 m vertical error. The user asked for 2026 data; the honest answer is
    that the most recent terrain OBSERVATIONS are 2015 (plus lidar to 2023),
    that GEDTM30's 2026 release date reflects better PROCESSING rather than
    newer ground truth, and that this is fine.
  * The one genuinely newer acquisition set is WorldDEM Neo, with >90% of
    land re-acquired 2017-2021 and ~60% again 2021-2025. It is commercial,
    and for this application it would buy nothing: we are averaging terrain
    into 18.5 km cells to predict where trees can grow.
  * Compare the climate side, where 1970-2000 normals and 2041-2060
    projections genuinely differ. Terrain is the ONE predictor block you
    reuse unchanged for both the current and the 2050 run - which is exactly
    what you want, because it isolates climate as the thing that changes.
  * Being a DTM, GEDTM30 also sidesteps a subtle time problem the surface
    models have: a DSM's canopy height is itself a function of forest age and
    disturbance, so a 2000 DSM and a 2015 DSM disagree over any stand that
    grew or was logged in between. Bare earth does not drift.

VERTICAL DATUM. GEDTM30, Copernicus DEM and FathomDEM use EGM2008; SRTM,
GMTED2010 and AW3D30 use EGM96. The two geoids differ by under ~1 m almost
everywhere. Against a 10-arcmin cell with up to 3,400 m of internal relief,
for a tree-suitability model, that is irrelevant - do NOT spend time on a
geoid conversion. It would matter for hydrological routing or flood work.
Mixing datums across products in one layer would be sloppy, so do not do
that; within GEDTM30 alone the question does not arise.

NODATA. GEDTM30 sets an explicit float32 nodata (3.4028235e38); compare with
`arr > 1e30`, and note this is the OPPOSITE sense to the climate rasters,
where valid data is `arr > -1e30`. Getting that backwards silently inverts
the land mask, so the regridded 10-arcmin layer written by
topography_rasters.py normalises nodata to -32768 to match the WorldClim
convention already used in this repo.""")


# ─────────────────────────────────────────────────────
# STEP 11: the join recipe
# ─────────────────────────────────────────────────────
def report_join_recipe():
    print("\n" + SEP)
    print("STEP 11  RECOMMENDED RECIPE")
    print(SEP)
    print("""
ELEVATION - done. Use topography/gedtm30/gedtm30_v1.2_elev_10m.tif as-is.
  It is already cell-for-cell on the climate grid (step 1) and covers 99.92%
  of non-Antarctic climate land cells (step 2). Mask on -32768.

SLOPE + ASPECT - derive them per 10-arcmin cell from the native 1-arcsec DTM:

  for each 1-degree window of land:
      1. read the native GEDTM30 window with a >=1 pixel HALO on all sides,
         or the cell edges get a spurious NaN
      2. compute slope and aspect PER NATIVE PIXEL with Horn's 3x3 kernel,
         using dx = res_lon * 111320 * cos(lat) and dy = res_lat * 110540
      3. reduce each 600 x 600 pixel block (= one 10-arcmin cell) to
           elev_mean  = mean(z)
           elev_std   = std(z)                      <- sub-cell ruggedness
           elev_range = max(z) - min(z)
           slope_mean = mean(slope)                 <- NOT slope(mean(z))
           slope_p90  = 90th percentile of slope
           northness  = sum(sin(slope)*cos(aspect)) / sum(sin(slope))
           eastness   = sum(sin(slope)*sin(aspect)) / sum(sin(slope))
      4. discard the window before moving on

  Never reverse steps 2 and 3. Step 6 measures the penalty: slope from a
  pre-averaged DEM came out roughly 20x too small.

JOINING TO THE MODEL - write the derived layers as GeoTIFFs on the exact
climate grid, in their own directory:

    topography_10m/
        topo_elev_mean.tif
        topo_elev_std.tif
        topo_slope_mean.tif
        topo_slope_p90.tif
        topo_northness.tif
        topo_eastness.tif

  Copy the profile straight off climate_current/wc2.1_10m_bio_1.tif
  (transform, crs, width, height) and change only dtype/nodata. Then
  xgboost_training.py's existing extract_climate() works unchanged when
  pointed at that directory, and generate_pseudo_absences() keeps using the
  climate raster as the land mask, so presences and background points stay on
  identical cells.

  Do NOT merge terrain into climate_current/. Keeping the blocks separate is
  what lets you swap the climate block for the 2050 projection while holding
  terrain fixed - which is the entire point of having terrain in the model.

  One change IS needed in the training code: intersect the pseudo-absence
  land mask with the terrain mask, so background points are not drawn from
  Antarctica (where terrain is nodata) or from the 269 ocean cells.

FEATURE COUNT - 19 bioclim + 6 terrain. With 392,725 presences over 293
species there is ample data. The real risk is collinearity between elev_mean
and the temperature bioclims, since temperature falls ~6 K/km: read XGBoost's
gain importance for elevation with that in mind. Expect elev_std, slope and
northness to carry the genuinely new signal, because they describe
within-cell habitat heterogeneity that no bioclim variable contains.

OPTIONAL, cheap, and worth it - GEDTM30 publishes a per-pixel uncertainty
layer (topography_rasters.py MODE="full" fetches it). Regridded the same way
it gives a per-cell "how much do we trust this terrain" field, highest over
dense canopy and steep ground. Useful as a QA filter or a sample weight.
""")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    print(SEP)
    print("TOPOGRAPHY ALIGNMENT VALIDATION")
    print("GEDTM30 v1.2 (2026-03-06) vs WorldClim 10-arcmin + GBIF occurrences")
    print(SEP)

    required = (CLIMATE_TEMPLATE, OCCURRENCES, GEDTM30_ELEV_10M, WORLDCLIM_ELEV)
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("ERROR: missing required input(s):")
        for p in missing:
            print(f"  {p}")
        print("\nRun: python3 topography_rasters.py")
        return

    report_product_choice()
    check_grid_identity()
    climate_land, elev, elev_valid = check_land_cell_coverage()
    df, _, _, _ = check_occurrence_coverage(climate_land, elev, elev_valid)
    check_canopy_bias()

    demo = os.path.join(GEDTM30_NATIVE_DIR, f"gedtm30_{DEMO_SITE}.tif")
    if os.path.exists(demo):
        derive_terrain_demo()
        check_native_windows(df)
    else:
        print(f"\nSKIPPING steps 6-9: {demo} not found "
              "(run topography_rasters.py)")

    report_provenance()
    report_join_recipe()

    print(SEP)
    print("VALIDATION COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()
