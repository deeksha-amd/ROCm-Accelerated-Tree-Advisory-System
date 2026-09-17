"""
Derive PRESENT-DAY climate rasters (19 bioclimatic variables) for recent year
windows, as drop-in replacements for the WorldClim 2.1 1970-2000 normals in
climate_current/.

WHY THIS EXISTS
    climate_current/ holds WorldClim 2.1 bioclim for 1970-2000, but the GBIF
    occurrences have a median year of 2024 and 60% of them fall in 2021-2026.
    This script builds the same 19 variables for modern windows so the model can
    be fitted against the climate the trees were actually observed in.

WHERE THE DATA COMES FROM
    WorldClim does NOT publish ready-made bioclim rasters for a recent window.
    It does publish "historical monthly weather data": monthly minimum
    temperature, maximum temperature and total precipitation for 1950-2024,
    downscaled from CRU TS 4.09 and bias-corrected against WorldClim 2.1.
        https://worldclim.org/data/monthlywth.html
    So we download those monthly grids, average them over the chosen window to
    get a 12-month climatology, and derive the 19 bioclim variables ourselves.

    Staying inside the WorldClim family matters: these grids arrive already on
    the exact project grid (2160 x 1080, EPSG:4326, 10 arc-minutes) and are
    bias-corrected to the same WorldClim 2.1 baseline as climate_current/, so
    the new blocks are directly comparable with the existing one.

    2024 is the last year published, so the most recent window achievable today
    ends in 2024, not 2026.

WINDOWS PRODUCED
    RECENT     2015-2024 -> climate_recent/
    BACKUP     2000-2015 -> climate_2000_2015/
    Both use the same source product and the same bioclim code, so they are
    directly comparable with each other as well as with climate_current/.

BIOCLIM DEFINITIONS
    O'Donnell & Ignizio (2012), the set implemented by dismo::biovars in R.
    Two conventions were pinned down empirically by reproducing WorldClim's own
    published 1970-2000 rasters from WorldClim's own published monthly normals
    (see validate_recent_climate.py):
      - BIO4  uses the SAMPLE standard deviation (n-1 denominator), x100.
      - BIO15 uses the dismo coefficient of variation on precipitation+1, i.e.
              100 * sd(prec, n-1) / (1 + mean(prec)).
    Getting either wrong shifts BIO4 by ~100 units and BIO15 by ~10 percent.
    The four quarter-based variables (BIO8, BIO9, BIO18, BIO19) use rolling
    3-month windows that wrap from December round to January, giving 12 windows.

USAGE
    python3 recent_climate_rasters.py             # build every window in WINDOWS
    python3 recent_climate_rasters.py RECENT      # just 2015-2024
    python3 recent_climate_rasters.py BACKUP      # just 2000-2015
    python3 recent_climate_rasters.py SAMPLE      # quick 2020-2024 smoke test
"""

import os
import sys
import zipfile

import numpy as np
import rasterio
import requests
from tqdm import tqdm

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
SRC_DIR = "climate_monthly_src"          # shared cache of downloaded archives
TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"

RESOLUTION = "10m"        # "10m", "5m", "2.5m" — must stay 10m to match the grid

# ── The windows to build: name -> (first year, last year, output directory) ──
# Add or edit entries here; nothing else needs changing.
WINDOWS = {
    "RECENT": (2015, 2024, "climate_recent"),
    "BACKUP": (2000, 2015, "climate_2000_2015"),
    "SAMPLE": (2020, 2024, "climate_recent_sample"),
}
DEFAULT_WINDOWS = ("RECENT", "BACKUP")   # what a bare run builds

# Output files use the existing project naming so they are drop-in compatible
# with climate_current/: wc2.1_10m_bio_1.tif ... wc2.1_10m_bio_19.tif
OUT_PREFIX = f"wc2.1_{RESOLUTION}_bio"

# ── WorldClim historical monthly weather (CRU TS 4.09 downscaled) ──
HIST_BASE_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/hist/cts4.09"
HIST_TAG = "cruts4.09"
HIST_VARS = ("tmin", "tmax", "prec")

# The monthly archives are bundled in decade blocks; the last one is short
# because the record currently stops at 2024.
DECADE_BLOCKS = [(1950, 1959), (1960, 1969), (1970, 1979), (1980, 1989),
                 (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2024)]
FIRST_YEAR_AVAILABLE = 1950
LAST_YEAR_AVAILABLE = 2024

# ── WorldClim 2.1 1970-2000 monthly normals (used by the validation script) ──
BASE_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"

KEEP_ZIPS = True          # keep archives so re-runs and validation are free;
                          # set False to delete them once every window is built
NODATA = -3.4e38          # project convention: valid cells satisfy arr > -1e30

BIO_LONG_NAMES = {
    1: "Annual mean temperature (C)",
    2: "Mean diurnal temperature range (C)",
    3: "Isothermality (100 * BIO2 / BIO7)",
    4: "Temperature seasonality (sd of monthly mean temp * 100)",
    5: "Max temperature of warmest month (C)",
    6: "Min temperature of coldest month (C)",
    7: "Temperature annual range (BIO5 - BIO6) (C)",
    8: "Mean temperature of wettest quarter (C)",
    9: "Mean temperature of driest quarter (C)",
    10: "Mean temperature of warmest quarter (C)",
    11: "Mean temperature of coldest quarter (C)",
    12: "Annual precipitation (mm)",
    13: "Precipitation of wettest month (mm)",
    14: "Precipitation of driest month (mm)",
    15: "Precipitation seasonality (coefficient of variation)",
    16: "Precipitation of wettest quarter (mm)",
    17: "Precipitation of driest quarter (mm)",
    18: "Precipitation of warmest quarter (mm)",
    19: "Precipitation of coldest quarter (mm)",
}


# ─────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────
def download_file(url, output_path):
    """Download with progress bar. Returns True on success."""
    response = requests.get(url, stream=True, timeout=120)

    if response.status_code != 200:
        print(f"ERROR: could not download (status {response.status_code})")
        print(f"URL tried: {url}")
        return False

    total_size = int(response.headers.get('content-length', 0))
    tmp_path = output_path + ".part"
    with open(tmp_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc=os.path.basename(output_path)[:44]) as pbar:
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                pbar.update(len(chunk))
    os.replace(tmp_path, output_path)
    return True


def blocks_for_years(year_from, year_to):
    """Which decade blocks must be fetched to cover [year_from, year_to]."""
    if year_from < FIRST_YEAR_AVAILABLE or year_to > LAST_YEAR_AVAILABLE:
        raise ValueError(
            f"WorldClim historical monthly weather covers "
            f"{FIRST_YEAR_AVAILABLE}-{LAST_YEAR_AVAILABLE}; "
            f"asked for {year_from}-{year_to}")
    return [b for b in DECADE_BLOCKS if b[0] <= year_to and b[1] >= year_from]


def hist_zip_name(var, block):
    return f"wc2.1_{HIST_TAG}_{RESOLUTION}_{var}_{block[0]}-{block[1]}.zip"


def ensure_hist_zip(var, block):
    """Download one monthly-weather archive if we do not already have it."""
    path = os.path.join(SRC_DIR, hist_zip_name(var, block))
    if os.path.exists(path) and zipfile.is_zipfile(path):
        return path
    if not download_file(f"{HIST_BASE_URL}/{hist_zip_name(var, block)}", path):
        return None
    return path


def ensure_normals_zip(var):
    """Download a WorldClim 2.1 1970-2000 monthly normals archive."""
    name = f"wc2.1_{RESOLUTION}_{var}.zip"
    path = os.path.join(SRC_DIR, name)
    if os.path.exists(path) and zipfile.is_zipfile(path):
        return path
    if not download_file(f"{BASE_URL}/{name}", path):
        return None
    return path


# ─────────────────────────────────────────────────────
# READ MONTHLY GRIDS
# ─────────────────────────────────────────────────────
def _read_tif_in_zip(zip_path, member):
    """Read a single-band GeoTIFF straight out of a zip, nodata -> NaN."""
    with rasterio.open(f"zip://{zip_path}!{member}") as src:
        arr = src.read(1).astype(np.float64)
        nd = src.nodata
    if nd is not None and not np.isnan(nd):
        arr[arr == nd] = np.nan
    arr[arr < -1e30] = np.nan
    return arr


def read_hist_month(var, year, month):
    block = next(b for b in DECADE_BLOCKS if b[0] <= year <= b[1])
    zip_path = os.path.join(SRC_DIR, hist_zip_name(var, block))
    member = f"wc2.1_{HIST_TAG}_{RESOLUTION}_{var}_{year}-{month:02d}.tif"
    return _read_tif_in_zip(zip_path, member)


def read_normals(var):
    """The published WorldClim 2.1 1970-2000 monthly normals, shape (12, H, W)."""
    zip_path = os.path.join(SRC_DIR, f"wc2.1_{RESOLUTION}_{var}.zip")
    return np.stack([
        _read_tif_in_zip(zip_path, f"wc2.1_{RESOLUTION}_{var}_{m:02d}.tif")
        for m in range(1, 13)
    ])


def monthly_climatology(year_from, year_to, verbose=True):
    """
    Average the monthly weather grids over a year window.

    Returns tmin, tmax, prec each shaped (12, H, W): the mean January minimum
    temperature, the mean February minimum temperature, and so on. Precipitation
    is the mean monthly total, not the sum over all years.
    """
    years = list(range(year_from, year_to + 1))
    out = {}
    for var in HIST_VARS:
        stack = []
        for month in tqdm(range(1, 13), desc=f"{var} {year_from}-{year_to}",
                          disable=not verbose):
            msum = None
            mcnt = None
            for year in years:
                a = read_hist_month(var, year, month)
                ok = np.isfinite(a)
                if msum is None:
                    msum = np.where(ok, a, 0.0)
                    mcnt = ok.astype(np.int16)
                else:
                    msum += np.where(ok, a, 0.0)
                    mcnt += ok
            with np.errstate(invalid='ignore', divide='ignore'):
                stack.append(np.where(mcnt > 0, msum / np.maximum(mcnt, 1), np.nan))
        out[var] = np.stack(stack)
    return out["tmin"], out["tmax"], out["prec"]


# ─────────────────────────────────────────────────────
# BIOCLIM
# ─────────────────────────────────────────────────────
def _rolling3(x):
    """
    Rolling 3-month sums over axis 0, wrapping December round to January.

    x is (12, N); result is (12, N) where row j covers months j, j+1, j+2.
    """
    xx = np.concatenate([x, x[:2]], axis=0)
    return xx[0:12] + xx[1:13] + xx[2:14]


def biovars_flat(tmin, tmax, prec):
    """
    The 19 bioclim variables from 12-month climatologies.

    tmin, tmax, prec are (12, N) with no NaNs. Returns dict {1..19: (N,)}.
    Temperatures in degrees C, precipitation in mm.
    """
    n = tmin.shape[1]
    idx = np.arange(n)
    tavg = (tmin + tmax) / 2.0

    prec_q = _rolling3(prec)              # quarterly precipitation totals
    tavg_q = _rolling3(tavg) / 3.0        # quarterly mean temperatures

    wettest = np.argmax(prec_q, axis=0)
    driest = np.argmin(prec_q, axis=0)
    warmest = np.argmax(tavg_q, axis=0)
    coldest = np.argmin(tavg_q, axis=0)

    b = {}
    b[1] = tavg.mean(axis=0)
    b[2] = (tmax - tmin).mean(axis=0)
    b[4] = tavg.std(axis=0, ddof=1) * 100.0    # SAMPLE sd — matches WorldClim
    b[5] = tmax.max(axis=0)
    b[6] = tmin.min(axis=0)
    b[7] = b[5] - b[6]
    b[3] = 100.0 * b[2] / b[7]
    b[8] = tavg_q[wettest, idx]
    b[9] = tavg_q[driest, idx]
    b[10] = tavg_q[warmest, idx]
    b[11] = tavg_q[coldest, idx]
    b[12] = prec.sum(axis=0)
    b[13] = prec.max(axis=0)
    b[14] = prec.min(axis=0)
    # dismo convention: CV of precipitation+1, sample sd. The +1 stops the CV
    # exploding in places where mean monthly rainfall is under 1 mm.
    b[15] = 100.0 * prec.std(axis=0, ddof=1) / (1.0 + prec.mean(axis=0))
    b[16] = prec_q[wettest, idx]
    b[17] = prec_q[driest, idx]
    b[18] = prec_q[warmest, idx]
    b[19] = prec_q[coldest, idx]
    return b


def biovars_grid(tmin, tmax, prec):
    """
    Same as biovars_flat but for (12, H, W) grids that may contain NaN.

    A cell is computed only where all 36 monthly inputs are present.
    Returns (dict {1..19: (H, W) float32}, valid_mask (H, W) bool).
    """
    shape = tmin.shape[1:]
    valid = (np.isfinite(tmin).all(0) & np.isfinite(tmax).all(0)
             & np.isfinite(prec).all(0))
    flat = biovars_flat(tmin[:, valid], tmax[:, valid], prec[:, valid])
    out = {}
    for i in range(1, 20):
        g = np.full(shape, NODATA, dtype=np.float32)
        g[valid] = flat[i].astype(np.float32)
        out[i] = g
    return out, valid


# ─────────────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────────────
def _profile(dtype, nodata):
    with rasterio.open(TEMPLATE) as t:
        profile = t.profile.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata,
                   compress="deflate", predictor=2, tiled=False)
    return profile


def write_bioclim(bio, out_dir, window_label):
    """Write 19 single-band GeoTIFFs on the exact template grid."""
    profile = _profile("float32", NODATA)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i in range(1, 20):
        path = os.path.join(out_dir, f"{OUT_PREFIX}_{i}.tif")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(bio[i], 1)
            dst.update_tags(
                BIOCLIM=str(i),
                DESCRIPTION=BIO_LONG_NAMES[i],
                PERIOD=window_label,
                SOURCE="WorldClim 2.1 historical monthly weather "
                       "(CRU TS 4.09 downscaled, bias-corrected to WorldClim 2.1)",
                METHOD="O'Donnell & Ignizio (2012) / dismo::biovars conventions",
                DERIVED_BY="recent_climate_rasters.py")
            dst.set_band_description(1, BIO_LONG_NAMES[i])
        written.append(path)
    return written


def write_mask(valid, out_dir):
    """Where bioclim could be computed (1) and where it could not (0)."""
    profile = _profile("uint8", None)
    path = os.path.join(out_dir, "coverage_mask.tif")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(valid.astype(np.uint8), 1)
        dst.set_band_description(1, "1 = bioclim computed, 0 = no data")
    return path


# ─────────────────────────────────────────────────────
# BUILD ONE WINDOW
# ─────────────────────────────────────────────────────
def build_window(name, verbose=True):
    year_from, year_to, out_dir = WINDOWS[name]
    label = f"{year_from}-{year_to}"
    n_years = year_to - year_from + 1

    print("-" * 62)
    print(f"WINDOW {name}: {label}  ({n_years} years)  ->  {out_dir}/")
    print("-" * 62)

    blocks = blocks_for_years(year_from, year_to)
    print(f"monthly archives needed: {len(blocks) * len(HIST_VARS)} "
          f"({len(blocks)} decade blocks x {len(HIST_VARS)} variables)")
    for block in blocks:
        for var in HIST_VARS:
            if ensure_hist_zip(var, block) is None:
                return None

    print("averaging monthly grids into a 12-month climatology...")
    tmin, tmax, prec = monthly_climatology(year_from, year_to, verbose=verbose)

    print("computing bioclim...")
    bio, valid = biovars_grid(tmin, tmax, prec)

    paths = write_bioclim(bio, out_dir, label)
    mask_path = write_mask(valid, out_dir)

    print(f"wrote {len(paths)} rasters + {os.path.basename(mask_path)}")
    print(f"valid cells: {int(valid.sum()):,}")
    print(f"BIO1 mean {bio[1][valid].mean():7.3f} C   "
          f"BIO12 mean {bio[12][valid].mean():7.1f} mm")
    return out_dir


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    requested = [a.upper() for a in sys.argv[1:]] or list(DEFAULT_WINDOWS)
    unknown = [w for w in requested if w not in WINDOWS]
    if unknown:
        print(f"Unknown window(s): {unknown}. Choose from {list(WINDOWS)}.")
        sys.exit(2)

    print("=" * 62)
    print("PRESENT-DAY CLIMATE RASTER BUILDER (19 bioclim variables)")
    print("=" * 62)
    print(f"Source:     WorldClim 2.1 historical monthly weather (CRU TS 4.09)")
    print(f"Coverage:   {FIRST_YEAR_AVAILABLE}-{LAST_YEAR_AVAILABLE}")
    print(f"Resolution: {RESOLUTION}   Grid: from {TEMPLATE}")
    print(f"Windows:    {', '.join(requested)}\n")

    os.makedirs(SRC_DIR, exist_ok=True)

    built = []
    for name in requested:
        out_dir = build_window(name)
        if out_dir is None:
            print(f"Aborting: a download failed for window {name}.")
            sys.exit(1)
        built.append((name, out_dir))
        print()

    with rasterio.open(TEMPLATE) as t:
        ref_valid = int((t.read(1) > -1e30).sum())

    print("=" * 62)
    print("DONE")
    print("=" * 62)
    for name, out_dir in built:
        y0, y1, _ = WINDOWS[name]
        print(f"  {name:<8} {y0}-{y1}  ->  {out_dir}/{OUT_PREFIX}_1.tif "
              f"... {OUT_PREFIX}_19.tif")
    print(f"\nTemplate has {ref_valid:,} valid land cells.")
    print("Next: python3 validate_recent_climate.py")

    if not KEEP_ZIPS:
        for f in os.listdir(SRC_DIR):
            if f.endswith(".zip"):
                os.remove(os.path.join(SRC_DIR, f))


if __name__ == "__main__":
    main()
