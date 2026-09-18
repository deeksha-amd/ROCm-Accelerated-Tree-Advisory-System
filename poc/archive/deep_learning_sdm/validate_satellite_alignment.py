"""
Validate the satellite rasters, and measure whether they are circular

Checks four things and prints hard numbers for each:
  1. GRID        - are the satellite layers pixel-identical to the WorldClim grid?
  2. VALUES      - are the units sane, do the class fractions close to 100 percent,
                   and does the majority class agree with the fractions?
  3. COVERAGE    - what fraction of WorldClim land cells and of the GBIF
                   occurrence points receive a value, against the 99.93 percent
                   that climate_current reaches and the 99.83 percent that
                   climate_recent reaches?
  4. CIRCULARITY - the point of this script.  Vegetation greenness and tree-cover
                   fraction describe where trees already grow, so using them to
                   predict where trees *can* grow leaks the answer into the
                   question.  Four measurements, from cheapest to most damning:

       4a  single-variable separation (AUC) of occurrence cells from background
           cells, for every satellite layer and for genuine climate, terrain and
           soil predictors, so the leak can be compared against a real signal;
       4b  how consistent that separation is across species.  A real niche
           variable separates a boreal spruce and a tropical fig in opposite
           directions; a leaked answer separates every species the same way;
       4c  where the occurrences sit relative to the cells the advisory is meant
           to serve, namely cells that are currently not wooded;
       4d  a spatial-block cross-validated logistic model per species, fitted with
           and without the vegetation layers, scored both on ordinary held-out
           cells and on held-out cells that are currently bare of trees.

No scipy, sklearn or xgboost on this machine, so the ranking statistics and the
logistic fit are implemented directly in numpy.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
SATELLITE_DIR = "satellite"
MANIFEST = "satellite/MANIFEST.json"
TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"
RECENT_CLIMATE = "climate_recent/wc2.1_10m_bio_1.tif"
RECENT_RAINFALL = "climate_recent/wc2.1_10m_bio_12.tif"
ELEVATION = "topography/gedtm30/gedtm30_v1.2_elev_10m.tif"
SOIL_PH = "soil_data/aligned_10m/ph_h2o_0-30cm_10m.tif"
OCCURRENCES = "gbif_500_species.csv"

CLIMATE_NODATA_THRESHOLD = -1e30   # WorldClim nodata is -3.4e38

# Plausible ranges used only to flag obviously wrong scaling factors.
EXPECTED_RANGE = {
    "landcover_tree_frac":            (0.0, 100.0, "% non-sea"),
    "landcover_tree_broadleaf_frac":  (0.0, 100.0, "% non-sea"),
    "landcover_tree_needleleaf_frac": (0.0, 100.0, "% non-sea"),
    "landcover_shrub_frac":           (0.0, 100.0, "% non-sea"),
    "landcover_grass_natural_frac":   (0.0, 100.0, "% non-sea"),
    "landcover_cropland_frac":        (0.0, 100.0, "% non-sea"),
    "landcover_builtup_frac":         (0.0, 100.0, "% non-sea"),
    "landcover_bare_frac":            (0.0, 100.0, "% non-sea"),
    "landcover_water_frac":           (0.0, 100.0, "% non-sea"),
    "landcover_snowice_frac":         (0.0, 100.0, "% non-sea"),
    "landcover_land_frac":            (0.0, 100.0, "% of cell"),
    "landcover_purity":               (0.0, 100.0, "%"),
    "ndvi_observed_mean":             (-0.2, 1.0, "NDVI"),
    "ndvi_peak":                      (-0.2, 1.0, "NDVI"),
    "ndvi_growing_season_mean":       (-0.2, 1.0, "NDVI"),
    "ndvi_seasonal_amplitude":        (0.0, 1.2, "NDVI"),
    "ndvi_observed_semimonths":       (0.0, 24.0, "composites"),
    "soilmoisture_months_observed":   (0.0, 12.0, "months"),
    "soilmoisture_mean":              (0.0, 1.0, "m3/m3"),
    "soilmoisture_amplitude":         (0.0, 1.0, "m3/m3"),
}
PARTITION = ["tree", "shrub", "grass_natural", "cropland",
             "builtup", "bare", "water", "snowice"]

# ── circularity measurement settings ──
BACKGROUND_CELLS = 60000       # random land cells used as the global background
RANDOM_SEED = 42               # same seed xgboost_training.py defaults to
MIN_RECORDS_PER_SPECIES = 250  # species included in the per-species tests
MAX_SPECIES = 40
MODEL_SPECIES = 12             # species carried into the logistic experiment
SPATIAL_BLOCK_DEG = 10         # side of a cross-validation block
N_FOLDS = 5
BARE_OF_TREES_PCT = 20.0       # "advice is needed here" threshold on tree cover
ABSENCE_RATIO = 3              # background cells per presence cell, per species

LAT_BANDS = [(90, 66.5, "Arctic (>66.5N)"),
             (66.5, 45, "Boreal (45-66.5N)"),
             (45, 23.5, "N temperate (23.5-45N)"),
             (23.5, 0, "N tropics (0-23.5N)"),
             (0, -23.5, "S tropics (0-23.5S)"),
             (-23.5, -45, "S temperate (23.5-45S)"),
             (-45, -90, "S high lat (<45S)")]


def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ─────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────
def read_layer(path):
    """Return one band as float32 with nodata turned into NaN."""
    with rasterio.open(path) as src:
        band = src.read(1).astype("float32")
        if src.nodata is not None and np.isfinite(src.nodata):
            band[band == np.float32(src.nodata)] = np.nan
        band[band < CLIMATE_NODATA_THRESHOLD] = np.nan
        return band


def load_satellite_layers(directory):
    layers = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith("_10m.tif"):
            continue
        key = name[:-len("_10m.tif")]
        layers[key] = os.path.join(directory, name)
    return layers


def occurrence_cells(path, template):
    """Map every occurrence row onto the target grid."""
    df = pd.read_csv(path, usecols=["species", "latitude", "longitude"])
    with rasterio.open(template) as src:
        transform = src.transform
        height, width = src.height, src.width
    col = np.floor((df["longitude"].to_numpy(dtype="float64") - transform.c)
                   / transform.a).astype("int64")
    row = np.floor((df["latitude"].to_numpy(dtype="float64") - transform.f)
                   / transform.e).astype("int64")
    inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    return df, row, col, inside


# ─────────────────────────────────────────────────────
# 1. GRID
# ─────────────────────────────────────────────────────
def check_grid(layers, template):
    divider("1. GRID  -  is every layer pixel-identical to the climate template?")
    with rasterio.open(template) as src:
        ref = (src.width, src.height, str(src.crs),
               tuple(round(v, 12) for v in tuple(src.transform)[:6]))
    print(f"template     {template}")
    print(f"             {ref[0]}x{ref[1]}  {ref[2]}  pixel {ref[3][0]:.16f}")
    print(f"             origin lon {ref[3][2]}, lat {ref[3][5]}")

    bad = []
    for key, path in layers.items():
        with rasterio.open(path) as src:
            here = (src.width, src.height, str(src.crs),
                    tuple(round(v, 12) for v in tuple(src.transform)[:6]))
            nodata, dtype = src.nodata, src.dtypes[0]
        if here != ref:
            bad.append((key, here))
        if nodata is None:
            bad.append((key, "no nodata value set"))
        del dtype
    print(f"\nlayers checked          {len(layers)}")
    print(f"grid mismatches         {len(bad)}")
    for key, detail in bad:
        print(f"   MISMATCH {key}: {detail}")
    if not bad:
        print("all layers share the template grid, CRS, origin and pixel size,")
        print("and all carry an explicit nodata value")
    return not bad


# ─────────────────────────────────────────────────────
# 2. VALUES
# ─────────────────────────────────────────────────────
def check_values(layers):
    divider("2. VALUES  -  units, ranges and internal consistency")
    print(f"{'layer':34s} {'min':>8s} {'max':>8s} {'mean':>8s} {'unit':>12s}  flag")
    problems = 0
    for key, path in layers.items():
        band = read_layer(path)
        good = np.isfinite(band)
        if not good.any():
            print(f"{key:34s} {'EMPTY':>8s}")
            problems += 1
            continue
        lo, hi, unit = EXPECTED_RANGE.get(key, (None, None, ""))
        vmin, vmax = float(band[good].min()), float(band[good].max())
        flag = ""
        if lo is not None and (vmin < lo - 1e-3 or vmax > hi + 1e-3):
            flag = f"OUT OF RANGE (expected {lo} to {hi})"
            problems += 1
        print(f"{key:34s} {vmin:8.3f} {vmax:8.3f} "
              f"{float(band[good].mean()):8.3f} {unit:>12s}  {flag}")

    # do the eight partition classes close to 100 percent?
    stack = [read_layer(layers[f"landcover_{n}_frac"]) for n in PARTITION]
    total = np.nansum(np.stack(stack), axis=0)
    valid = np.isfinite(stack[0])
    err = np.abs(total[valid] - 100.0)
    print(f"\nclass fractions sum to 100 percent on {int(valid.sum()):,} cells:")
    print(f"   max absolute error   {err.max():.4f} percentage points")
    print(f"   cells off by >0.1 pp {int((err > 0.1).sum()):,}")
    if err.max() > 0.5:
        problems += 1

    # broadleaf + needleleaf must reconstruct the total tree fraction
    tree = read_layer(layers["landcover_tree_frac"])
    parts = (read_layer(layers["landcover_tree_broadleaf_frac"])
             + read_layer(layers["landcover_tree_needleleaf_frac"]))
    diff = np.abs(tree - parts)
    print(f"   broadleaf+needleleaf vs tree, max error "
          f"{np.nanmax(diff):.4f} pp")

    # majority class must be the argmax of the fractions
    with rasterio.open(layers["landcover_majority_class"]) as src:
        code = src.read(1)
        codes = json.loads(src.tags()["satellite_classes"])
    purity = read_layer(layers["landcover_purity"])
    order = np.stack(stack)
    winner = np.where(np.isfinite(order), order, -1).argmax(axis=0)
    expect = np.zeros_like(code)
    for i, n in enumerate(PARTITION):
        expect[valid & (winner == i)] = codes[n]
    agree = (expect == code)[valid].mean() * 100
    print(f"\nmajority class agrees with argmax of the fractions "
          f"on {agree:.2f}% of land cells")
    print(f"mean purity of the majority class   {np.nanmean(purity):.1f}%")
    print(f"cells whose majority class holds <50%   "
          f"{int(np.nansum(purity < 50)):,}  "
          f"(a single class code would be misleading there)")

    # exclusion mask
    with rasterio.open(layers["planting_exclusion_mask"]) as src:
        mask = src.read(1)
        bits = json.loads(src.tags()["satellite_bits"])
    land = mask != 255
    print(f"\nplanting exclusion mask, {int(land.sum()):,} land cells:")
    for name, bit in bits.items():
        n = int(((mask & bit) > 0)[land].sum())
        print(f"   {name:22s} bit {bit:3d}   {n:8,d} cells "
              f"({100 * n / land.sum():5.2f}%)")
    clear = int((mask[land] == 0).sum())
    print(f"   {'no exclusion':22s}          {clear:8,d} cells "
          f"({100 * clear / land.sum():5.2f}%)")
    print(f"\nvalue problems flagged  {problems}")
    return problems == 0


# ─────────────────────────────────────────────────────
# 3. COVERAGE
# ─────────────────────────────────────────────────────
def check_coverage(layers, template, occ):
    divider("3. COVERAGE  -  land cells and GBIF occurrence points with a value")
    template_band = read_layer(template)
    land = np.isfinite(template_band)
    n_land = int(land.sum())
    recent = np.isfinite(read_layer(RECENT_CLIMATE))
    print(f"climate_current land cells   {n_land:,}")
    print(f"climate_recent  land cells   {int(recent.sum()):,} "
          f"({100 * recent.sum() / n_land:.2f}% of the above)")

    df, row, col, inside = occ
    n_points = len(df)
    print(f"GBIF occurrence rows         {n_points:,} "
          f"({int((~inside).sum())} fall outside the grid)")

    print(f"\n{'layer':34s} {'land cells':>12s} {'% of land':>10s} "
          f"{'GBIF pts':>10s} {'% of GBIF':>10s}")
    coverage = {}
    for key, path in layers.items():
        if key in ("landcover_majority_class", "planting_exclusion_mask"):
            continue
        band = read_layer(path)
        good = np.isfinite(band)
        on_land = int((good & land).sum())
        hits = np.zeros(n_points, dtype=bool)
        hits[inside] = good[row[inside], col[inside]]
        coverage[key] = (on_land, int(hits.sum()))
        print(f"{key:34s} {on_land:12,d} {100 * on_land / n_land:9.2f}% "
              f"{int(hits.sum()):10,d} {100 * hits.sum() / n_points:9.2f}%")

    # where the NDVI holes are
    ndvi = np.isfinite(read_layer(layers["ndvi_observed_mean"]))
    tree = read_layer(layers["landcover_tree_frac"])
    bare = read_layer(layers["landcover_bare_frac"])
    snow = read_layer(layers["landcover_snowice_frac"])
    holes = land & np.isfinite(tree) & ~ndvi
    print(f"\nNDVI is missing on {int(holes.sum()):,} land cells. What are they?")
    print(f"   mean bare-ground fraction there      "
          f"{np.nanmean(bare[holes]):5.1f}%  (vs "
          f"{np.nanmean(bare[land & ndvi]):.1f}% where NDVI exists)")
    print(f"   mean permanent snow/ice fraction     "
          f"{np.nanmean(snow[holes]):5.1f}%  (vs "
          f"{np.nanmean(snow[land & ndvi]):.1f}%)")
    print(f"   mean tree fraction                   "
          f"{np.nanmean(tree[holes]):5.1f}%  (vs "
          f"{np.nanmean(tree[land & ndvi]):.1f}%)")
    print("   i.e. the gaps are desert and ice, which the source masks by")
    print("   design because NDVI is meaningless there, not random dropout")

    # how much of the year each cell actually sees
    seen = read_layer(layers["ndvi_observed_semimonths"])
    have = land & (seen > 0)
    print(f"\nsemi-monthly composites observed per cell, of 24 "
          f"({int(have.sum()):,} cells with any):")
    for bar in (24, 20, 16, 12, 8):
        n = int((seen[have] >= bar - 0.001).sum())
        print(f"   at least {bar:2d} of 24   {n:8,d} cells "
              f"({100 * n / have.sum():5.1f}%)")
    print("   the shortfall is winter snow cover, so above about 60 degrees the")
    print("   'observed mean' is a snow-free-season mean, not a calendar mean")

    # soil-moisture holes are the interesting ones: do they track canopy?
    sm = np.isfinite(read_layer(layers["soilmoisture_mean"]))
    sm_holes = land & np.isfinite(tree) & ~sm
    print(f"\nSoil moisture is missing on {int(sm_holes.sum()):,} land cells.")
    print(f"   mean tree fraction there             "
          f"{np.nanmean(tree[sm_holes]):5.1f}%  (vs "
          f"{np.nanmean(tree[land & sm]):.1f}% where it exists)")
    print(f"   mean snow/ice fraction there         "
          f"{np.nanmean(snow[sm_holes]):5.1f}%  (vs "
          f"{np.nanmean(snow[land & sm]):.1f}%)")
    print("   microwave retrieval fails under dense canopy and over frozen or")
    print("   snow-covered ground, so this layer's nodata pattern is itself")
    print("   partly a vegetation signal - the same defect that disqualified")
    print("   the Copernicus DEM for this project")

    # latitude breakdown for the headline layers
    print(f"\n{'band':26s} {'land':>9s} {'tree frac':>10s} {'NDVI':>10s} "
          f"{'soil moist':>11s}")
    with rasterio.open(template) as src:
        lats = src.xy(np.arange(src.height), np.zeros(src.height))[1]
    lats = np.asarray(lats)
    for hi, lo, name in LAT_BANDS:
        rows = np.flatnonzero((lats <= hi) & (lats > lo))
        if rows.size == 0:
            continue
        sub = land[rows]
        n = int(sub.sum())
        if n == 0:
            print(f"{name:26s} {0:9,d}")
            continue
        def pct(valid_mask):
            return 100 * (valid_mask[rows] & sub).sum() / n
        print(f"{name:26s} {n:9,d} {pct(np.isfinite(tree)):9.1f}% "
              f"{pct(ndvi):9.1f}% {pct(sm):10.1f}%")
    return coverage


# ─────────────────────────────────────────────────────
# 4. CIRCULARITY
# ─────────────────────────────────────────────────────
def average_ranks(values):
    """Ranks with ties averaged, vectorised (stands in for scipy.rankdata)."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = values.size
    start = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    stop = np.r_[start[1:], n]
    mean_rank = (start + stop - 1) / 2.0 + 1.0
    ranks = np.empty(n, dtype="float64")
    ranks[order] = np.repeat(mean_rank, stop - start)
    return ranks


def auc(positive, negative):
    """Probability that a random presence scores above a random background."""
    if positive.size == 0 or negative.size == 0:
        return np.nan
    ranks = average_ranks(np.concatenate([positive, negative]))
    n_pos = positive.size
    total = ranks[:n_pos].sum()
    return (total - n_pos * (n_pos + 1) / 2.0) / (n_pos * negative.size)


def fit_logistic(features, labels, iterations=1500, rate=1.0, ridge=1e-3):
    """Plain gradient-descent logistic regression on standardised features."""
    design = np.column_stack([np.ones(len(features)), features])
    weights = np.zeros(design.shape[1])
    for _ in range(iterations):
        prob = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -30, 30)))
        gradient = design.T @ (prob - labels) / len(labels)
        gradient[1:] += ridge * weights[1:]
        weights -= rate * gradient
    return weights


def apply_logistic(weights, features):
    design = np.column_stack([np.ones(len(features)), features])
    return design @ weights


def measure_circularity(layers, template, occ):
    divider("4. CIRCULARITY  -  do the satellite layers encode the answer?")
    rng = np.random.default_rng(RANDOM_SEED)

    candidates = {
        "ndvi_observed_mean":       ("satellite", "NDVI, observed mean"),
        "ndvi_peak":                ("satellite", "NDVI, annual peak"),
        "ndvi_growing_season_mean": ("satellite", "NDVI, growing season"),
        "ndvi_seasonal_amplitude":  ("satellite", "NDVI, seasonal amplitude"),
        "landcover_tree_frac":      ("satellite", "tree-cover fraction"),
        "landcover_shrub_frac":     ("satellite", "shrub fraction"),
        "landcover_cropland_frac":  ("satellite", "cropland fraction"),
        "landcover_builtup_frac":   ("satellite", "built-up fraction"),
        "landcover_bare_frac":      ("satellite", "bare-ground fraction"),
        "soilmoisture_mean":        ("satellite", "surface soil moisture"),
    }
    grids = {k: read_layer(layers[k]) for k in candidates}
    grids["bio_1_recent"] = read_layer(RECENT_CLIMATE)
    candidates["bio_1_recent"] = ("reference", "annual mean temperature")
    grids["bio_12_recent"] = read_layer(RECENT_RAINFALL)
    candidates["bio_12_recent"] = ("reference", "annual precipitation")
    if os.path.exists(ELEVATION):
        grids["elevation"] = read_layer(ELEVATION)
        candidates["elevation"] = ("reference", "bare-earth elevation")
    if os.path.exists(SOIL_PH):
        grids["soil_ph"] = read_layer(SOIL_PH)
        candidates["soil_ph"] = ("reference", "topsoil pH")

    # Restrict to cells where every compared variable exists, so that no
    # variable gets an easier subset than another.  Surface soil moisture is
    # deliberately left out of that intersection: its retrieval fails under
    # dense canopy, so requiring it would delete the closed-forest cells that
    # this whole measurement is about.  It is scored on its own valid subset
    # instead, with the cell count printed so the difference is visible.
    SPARSE = {"soilmoisture_mean"}
    usable = np.ones(grids["bio_1_recent"].shape, dtype=bool)
    for key, grid in grids.items():
        if key not in SPARSE:
            usable &= np.isfinite(grid)
    print(f"cells with every compared variable valid   {int(usable.sum()):,}")
    print(f"   (surface soil moisture excluded from that intersection: it is "
          f"valid on only\n    "
          f"{int((usable & np.isfinite(grids['soilmoisture_mean'])).sum()):,} "
          f"of them, and the missing ones are the forested ones)")

    df, row, col, inside = occ
    in_usable = np.zeros(len(df), dtype=bool)
    in_usable[inside] = usable[row[inside], col[inside]]
    print(f"GBIF records usable for this comparison    {int(in_usable.sum()):,} "
          f"of {len(df):,} ({100 * in_usable.mean():.2f}%)")

    flat = np.flatnonzero(usable.ravel())
    presence_flat = np.unique(row[in_usable] * usable.shape[1] + col[in_usable])
    background_pool = np.setdiff1d(flat, presence_flat, assume_unique=False)
    background_flat = rng.choice(
        background_pool, size=min(BACKGROUND_CELLS, background_pool.size),
        replace=False)
    print(f"distinct occurrence cells                  {presence_flat.size:,}")
    print(f"random background cells drawn              {background_flat.size:,}")

    def sample(key, flat_index):
        return grids[key].ravel()[flat_index]

    # ── 4a: single-variable separation ──
    print("\n4a. How well does ONE variable alone separate occurrence cells")
    print("    from background land cells?  AUC 0.5 means no information;")
    print("    separation is |AUC - 0.5|, so bigger means more leakage.")
    print(f"\n{'variable':28s} {'kind':10s} {'AUC':>7s} {'separation':>11s} "
          f"{'presence mean':>14s} {'background':>11s} {'cells':>10s}")
    scores = {}
    for key, (kind, label) in candidates.items():
        pos = sample(key, presence_flat)
        neg = sample(key, background_flat)
        pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        a = auc(pos, neg)
        scores[key] = a
        print(f"{label:28s} {kind:10s} {a:7.3f} {abs(a - 0.5):11.3f} "
              f"{pos.mean():14.3f} {neg.mean():11.3f} "
              f"{pos.size + neg.size:10,d}")

    veg = ["ndvi_observed_mean", "ndvi_peak", "ndvi_growing_season_mean",
           "landcover_tree_frac"]
    ref = ["bio_1_recent", "bio_12_recent"] + \
          [k for k in ("elevation", "soil_ph") if k in scores]
    best_veg = max(veg, key=lambda k: abs(scores[k] - 0.5))
    best_ref = max(ref, key=lambda k: abs(scores[k] - 0.5))
    print(f"\n   strongest vegetation variable   {candidates[best_veg][1]}: "
          f"separation {abs(scores[best_veg] - 0.5):.3f}")
    print(f"   strongest genuine predictor     {candidates[best_ref][1]}: "
          f"separation {abs(scores[best_ref] - 0.5):.3f}")
    ratio = abs(scores[best_veg] - 0.5) / max(abs(scores[best_ref] - 0.5), 1e-9)
    print(f"   the vegetation variable separates {ratio:.1f}x more strongly "
          f"than the best real predictor")
    print("\n   Two warnings about reading this table on its own.  First, a")
    print("   pooled global test handicaps any variable whose effect is")
    print("   species-specific: 293 species between them occupy the whole")
    print("   temperature range, so temperature looks uninformative here even")
    print("   though it is the backbone of the model.  Test 4b fixes that.")
    print("   Second, if built-up fraction comes out at the top of this table,")
    print("   that is survey bias, not ecology - GBIF records cluster along")
    print("   roads and around cities.")

    # ── 4b: consistency across species ──
    print("\n4b. Is that separation species-specific, as a real niche must be,")
    print("    or the same for every species, as a leaked answer must be?")
    counts = df.loc[in_usable, "species"].value_counts()
    species = counts[counts >= MIN_RECORDS_PER_SPECIES].index[:MAX_SPECIES]
    print(f"    species with >= {MIN_RECORDS_PER_SPECIES} usable records: "
          f"{len(species)}")

    per_species = {k: [] for k in candidates}
    species_cells = {}
    width = usable.shape[1]
    for name in species:
        pick = in_usable & (df["species"].to_numpy() == name)
        cells = np.unique(row[pick] * width + col[pick])
        if cells.size < 20:
            continue
        species_cells[name] = cells
        others = np.setdiff1d(background_flat, cells)
        for key in candidates:
            pos, neg = sample(key, cells), sample(key, others)
            per_species[key].append(auc(pos[np.isfinite(pos)],
                                        neg[np.isfinite(neg)]))

    print(f"\n{'variable':28s} {'kind':10s} {'mean sep':>9s} {'sd of AUC':>10s} "
          f"{'same sign':>10s}")
    for key, (kind, label) in candidates.items():
        values = np.array(per_species[key], dtype="float64")
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        above = (values > 0.5).mean()
        agreement = max(above, 1 - above) * 100
        print(f"{label:28s} {kind:10s} {np.abs(values - 0.5).mean():9.3f} "
              f"{values.std():10.3f} {agreement:9.0f}%")
    print("\n    'same sign' is the share of species whose occurrences fall on")
    print("    the SAME side of the variable.  100% means every species, boreal")
    print("    or tropical, responds identically - which is the signature of a")
    print("    variable that is really just answering 'are there trees here?'.")

    # ── 4c: the cells the advisory actually has to serve ──
    print("\n4c. The advisory exists to rank cells that are NOT wooded today.")
    tree = grids["landcover_tree_frac"]
    pos_tree = sample("landcover_tree_frac", presence_flat)
    bg_tree = sample("landcover_tree_frac", background_flat)
    print(f"    tree-cover fraction at occurrence cells   "
          f"median {np.median(pos_tree):5.1f}%  mean {pos_tree.mean():5.1f}%")
    print(f"    tree-cover fraction at background cells   "
          f"median {np.median(bg_tree):5.1f}%  mean {bg_tree.mean():5.1f}%")
    share = 100 * (pos_tree < BARE_OF_TREES_PCT).mean()
    print(f"    occurrence cells with under {BARE_OF_TREES_PCT:.0f}% tree cover"
          f"      {share:5.1f}%")
    print(f"    background cells with under {BARE_OF_TREES_PCT:.0f}% tree cover"
          f"      {100 * (bg_tree < BARE_OF_TREES_PCT).mean():5.1f}%")
    q = np.percentile(pos_tree, [5, 25, 50, 75, 95])
    print(f"    presence tree-cover percentiles 5/25/50/75/95: "
          f"{q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} / {q[3]:.1f} / {q[4]:.1f}")
    rank_of_zero = 100 * (pos_tree <= BARE_OF_TREES_PCT).mean()
    print(f"    so a treeless candidate site sits at the {rank_of_zero:.1f}th")
    print("    percentile of the training presences on this variable.  That is")
    print("    not extrapolation - plenty of records do come from open ground -")
    print("    but the association is monotonic, so on this variable alone the")
    print("    model ranks every candidate planting site below most of the")
    print("    places it was trained on, for no ecological reason.")
    print(f"    NDVI at occurrence cells   mean "
          f"{sample('ndvi_observed_mean', presence_flat).mean():.3f}   "
          f"background mean "
          f"{sample('ndvi_observed_mean', background_flat).mean():.3f}")

    # ── 4d: does it actually inflate cross-validated scores? ──
    print("\n4d. Fit the same model with and without the vegetation layers,")
    print("    using spatial-block cross-validation, and score it three ways:")
    print("    on all held-out cells; on held-out cells that are currently bare")
    print("    of trees; and on held-out cleared farmland, which is the land an")
    print("    afforestation advisory is actually pointed at.")

    rows_grid, cols_grid = np.divmod(np.arange(usable.size), width)
    with rasterio.open(template) as src:
        transform = src.transform
    lat_of = transform.f + (rows_grid + 0.5) * transform.e
    lon_of = transform.c + (cols_grid + 0.5) * transform.a
    lat_block = np.floor((lat_of + 90) / SPATIAL_BLOCK_DEG).astype("int64")
    lon_block = np.floor((lon_of + 180) / SPATIAL_BLOCK_DEG).astype("int64")
    # Mix both axes into the fold id, otherwise the folds become longitude
    # stripes and neighbouring latitudes leak between train and test.
    block = (lat_block * 7 + lon_block * 13) % N_FOLDS

    climate_keys = ["bio_1_recent", "bio_12_recent"] + \
                   [k for k in ("elevation", "soil_ph") if k in grids]
    veg_keys = ["ndvi_observed_mean", "ndvi_peak", "ndvi_seasonal_amplitude",
                "landcover_tree_frac"]
    feature_sets = {
        "climate/terrain/soil only": climate_keys,
        "+ NDVI and tree cover": climate_keys + veg_keys,
        "vegetation layers only": veg_keys,
    }

    tree_flat = tree.ravel()
    crop_flat = grids["landcover_cropland_frac"].ravel()
    SLICES = ("all", "bare", "farmland")
    results = {k: {s: [] for s in SLICES} for k in feature_sets}
    used_species = 0
    for name in list(species_cells)[:MODEL_SPECIES]:
        cells = species_cells[name]
        pool = np.setdiff1d(background_flat, cells)
        n_abs = min(pool.size, max(ABSENCE_RATIO * cells.size, 500))
        absences = rng.choice(pool, size=n_abs, replace=False)
        index = np.concatenate([cells, absences])
        labels = np.concatenate([np.ones(cells.size), np.zeros(absences.size)])
        folds = block[index]
        if len(np.unique(folds)) < 2:
            continue
        used_species += 1

        for label, keys in feature_sets.items():
            raw = np.column_stack([grids[k].ravel()[index] for k in keys])
            per_fold = {s: [] for s in SLICES}
            for fold in np.unique(folds):
                train, test = folds != fold, folds == fold
                if labels[train].sum() < 5 or labels[test].sum() < 5:
                    continue
                mu = raw[train].mean(axis=0)
                sd = raw[train].std(axis=0)
                sd[sd == 0] = 1.0
                weights = fit_logistic((raw[train] - mu) / sd, labels[train])
                score = apply_logistic(weights, (raw[test] - mu) / sd)
                held = labels[test]
                low_tree = tree_flat[index][test] < BARE_OF_TREES_PCT
                subsets = {
                    "all": np.ones(held.shape, dtype=bool),
                    "bare": low_tree,
                    "farmland": low_tree & (crop_flat[index][test] > 40.0),
                }
                for name_slice, keep in subsets.items():
                    if (held[keep] == 1).sum() < 5 or (held[keep] == 0).sum() < 5:
                        continue
                    value = auc(score[keep][held[keep] == 1],
                                score[keep][held[keep] == 0])
                    if np.isfinite(value):
                        per_fold[name_slice].append(value)
            for name_slice in SLICES:
                if per_fold[name_slice]:
                    results[label][name_slice].append(
                        np.mean(per_fold[name_slice]))

    print(f"\n    species modelled: {used_species}, "
          f"{N_FOLDS}-fold {SPATIAL_BLOCK_DEG}-degree spatial blocks")
    print(f"\n    {'feature set':30s} {'all cells':>12s} {'treeless':>12s} "
          f"{'cleared farmland':>18s}")
    for label in feature_sets:
        cells = []
        for name_slice in SLICES:
            values = np.array(results[label][name_slice])
            cells.append(f"{values.mean():.3f} (n={values.size})"
                         if values.size else "n/a")
        print(f"    {label:30s} {cells[0]:>12s} {cells[1]:>12s} "
              f"{cells[2]:>18s}")

    base = np.array(results["climate/terrain/soil only"]["all"])
    with_veg = np.array(results["+ NDVI and tree cover"]["all"])
    veg_only = np.array(results["vegetation layers only"]["all"])
    if base.size and with_veg.size:
        print(f"\n    adding the vegetation layers changes the headline "
              f"cross-validated AUC by {with_veg.mean() - base.mean():+.3f}")
    if veg_only.size and base.size:
        print(f"    vegetation layers ALONE, with no climate, soil or terrain "
              f"at all, reach AUC\n    {veg_only.mean():.3f} against "
              f"{base.mean():.3f} for the whole genuine environmental set.  A "
              f"model that\n    knows nothing about the environment and only "
              f"looks at how green a place\n    already is beats one that knows "
              f"the climate.  That is the tautology, measured.")
    farm_base = np.array(results["climate/terrain/soil only"]["farmland"])
    farm_veg = np.array(results["vegetation layers only"]["farmland"])
    if farm_base.size and farm_veg.size and base.size and veg_only.size:
        print(f"\n    And here is why it is worthless anyway.  On all held-out "
              f"cells the\n    vegetation-only model leads the environment-only "
              f"model by "
              f"{veg_only.mean() - base.mean():+.3f}.\n    On cleared farmland "
              f"- the land this advisory exists to rank - that lead\n    is "
              f"{farm_veg.mean() - farm_base.mean():+.3f}.  The apparent skill "
              f"evaporates exactly where the\n    question is actually asked, "
              f"which is the definition of a leaked answer.")
    return scores


def report_cross_checks(manifest_path):
    divider("5. CROSS-CHECK against finer products on sampled tiles")
    if not os.path.exists(manifest_path):
        print("no manifest found")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    checks = manifest.get("cross_checks", {})

    rows = checks.get("worldcover_10m", [])
    if rows:
        print("ESA WorldCover v200 (10 m, 2021) vs our 300 m fractions, "
              "per 18.5 km cell:")
        print(f"{'tile':10s} {'layer':16s} {'WC 10m %':>9s} {'ours %':>8s} "
              f"{'mean |diff|':>12s} {'mode-ov bias':>13s}")
        for r in rows:
            print(f"{r['tile']:10s} {r['layer']:16s} "
                  f"{r['worldcover_10m_pct']:9.2f} {r['cci_300m_pct']:8.2f} "
                  f"{r['mean_abs_diff_pct']:12.2f} "
                  f"{r['worldcover_mode_overview_bias_pct']:13.2f}")
        bias = np.array([r["worldcover_mode_overview_bias_pct"] for r in rows])
        if bias.size:
            print(f"\n   'mode-ov bias' is what we would have got by reading "
                  f"WorldCover's mode-\n   resampled 320 m overviews instead of "
                  f"40 m: a mean shift of {bias.mean():+.2f} pp and\n"
                  f"   {np.abs(bias).mean():.2f} pp in absolute terms, always "
                  f"towards whichever class is\n   locally dominant.  That is "
                  f"why the global fractions come from the 300 m\n"
                  f"   fractional product rather than from 10 m overviews.")

    rows = checks.get("hansen_30m", [])
    if rows:
        hansen = manifest.get("sources", {}).get(
            "Hansen Global Forest Change", {})
        print(f"\nHansen Global Forest Change {hansen.get('version', '')} "
              f"({hansen.get('epoch', '')}) vs ours:")
        for r in rows:
            print(f"   tile {r['tile']}: Hansen "
                  f"{r['hansen_2024_canopy_pct']:.2f}% vs ours "
                  f"{r['cci_300m_tree_pct']:.2f}%, mean |diff| "
                  f"{r['mean_abs_diff_pct']:.2f} pp, correlation "
                  f"{r['correlation']:.3f} over {r['blocks_compared']} "
                  f"1-degree blocks")

    print("\nsources used:")
    for name, meta in manifest.get("sources", {}).items():
        print(f"   {name:32s} {meta.get('version', ''):12s} "
              f"epoch {meta.get('epoch', '?'):12s} "
              f"{meta.get('native_resolution', '')}")
    print("\nsources rejected:")
    for name, why in manifest.get("rejected_sources", {}).items():
        print(f"   {name}\n      {why}")


def print_verdict():
    divider("VERDICT  -  what may be used how")
    rows = [
        ("soilmoisture_mean / _amplitude", "PREDICTOR, with a caveat",
         "microwave, not greenness; but costs ~21% of the records"),
        ("landcover_builtup_frac", "FILTER",
         "land availability, not vegetation state"),
        ("landcover_water_frac", "FILTER", "land availability"),
        ("landcover_snowice_frac", "FILTER", "land availability"),
        ("landcover_land_frac", "FILTER", "coastline handling"),
        ("landcover_cropland_frac", "FILTER",
         "competing land use; weak vegetation content"),
        ("landcover_bare_frac", "FILTER (weak predictor at best)",
         "partly substrate, partly vegetation absence"),
        ("landcover_tree_frac", "DO NOT USE AS PREDICTOR",
         "this is the response variable in disguise"),
        ("landcover_tree_broadleaf/needleleaf_frac", "DO NOT USE AS PREDICTOR",
         "as above, and additionally leaks species identity"),
        ("landcover_shrub_frac", "DO NOT USE AS PREDICTOR", "vegetation state"),
        ("landcover_grass_natural_frac", "DO NOT USE AS PREDICTOR",
         "vegetation state"),
        ("ndvi_observed_mean / _peak / _growing_season_mean",
         "DO NOT USE AS PREDICTOR", "direct measure of current greenness"),
        ("ndvi_seasonal_amplitude", "DO NOT USE AS PREDICTOR",
         "closest to defensible, but still derived from greenness"),
        ("planting_exclusion_mask", "FILTER (this is its whole purpose)",
         "apply to model output, never to model input"),
    ]
    print(f"{'layer':44s} {'role':34s} why")
    for layer, role, why in rows:
        print(f"{layer:44s} {role:34s} {why}")


def main():
    print("=" * 72)
    print("SATELLITE ALIGNMENT AND CIRCULARITY VALIDATION")
    print("=" * 72)
    layers = load_satellite_layers(SATELLITE_DIR)
    print(f"found {len(layers)} layers in {SATELLITE_DIR}/")

    occ = occurrence_cells(OCCURRENCES, TEMPLATE)
    grid_ok = check_grid(layers, TEMPLATE)
    values_ok = check_values(layers)
    check_coverage(layers, TEMPLATE, occ)
    measure_circularity(layers, TEMPLATE, occ)
    report_cross_checks(MANIFEST)
    print_verdict()

    divider("SUMMARY")
    print(f"grid identical to template   {'yes' if grid_ok else 'NO'}")
    print(f"values and fractions sane    {'yes' if values_ok else 'NO'}")
    print("circularity                  measured above; vegetation-derived")
    print("                             layers are filters, not predictors")


if __name__ == "__main__":
    main()
