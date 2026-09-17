"""
Download TOPOGRAPHY / ELEVATION rasters for terrain predictors
Elevation + slope + aspect (sun exposure); affects drainage, frost, sunlight

PRIMARY SOURCE: GEDTM30 v1.2 (released 6 March 2026)
    Global Ensemble Digital Terrain Model, 1 arc-second (~30 m).
    Ho, Grohmann, Lindsay, Reuter, Parente, Witjes & Hengl (2025), PeerJ 13:e19673
    https://doi.org/10.7717/peerj.19673      https://doi.org/10.5281/zenodo.14900181
    Licence: CC-BY-4.0 (permissive, commercial use allowed, no share-alike)

Why GEDTM30 and not the usual suspects:
  * It is a bare-earth DIGITAL TERRAIN model. SRTM, Copernicus DEM, AW3D30 and
    NASADEM are digital SURFACE models: over closed-canopy forest they sit on
    the treetops. For a TREE-species model that is an actively harmful bias -
    the predictor would partly encode the forest we are trying to predict.
  * Trained on ~30 billion ICESat-2 + GEDI ground-return points, it cuts
    Copernicus GLO-30's RMSE by 27.3% where tree cover exceeds 50%.
  * 85N to 65S, so it covers the 8,148 occurrence records above 60N that SRTM
    (60N-56S) simply does not reach.
  * Anonymous HTTPS Cloud-Optimised GeoTIFF with HTTP range reads, so we can
    pull windows out of a 432 GB global mosaic without downloading it.
  * It also publishes SLOPE (and 14 other land-surface parameters) globally at
    six grid spacings - but NOT aspect, which we still have to derive.

Rejected alternatives and why (full table in validate_topography_alignment.py):
  FathomDEM v1.0 (Dec 2024 / Feb 2025) is measurably the most accurate global
    bare-earth DTM, but it is CC BY-NC-SA 4.0 (non-commercial + share-alike)
    and its Zenodo records are ACCESS-RESTRICTED behind a manual request. Not
    scriptable, and the licence would infect every derived layer.
  WorldDEM Neo (Airbus, new global coverage early 2026) is commercial.
  TanDEM-X 30m EDEM (DLR, 15 Nov 2023) needs a scientific-use registration.
  FABDEM V1-2 (18 Jan 2023) is also CC BY-NC-SA and less accurate than GEDTM30.
  Copernicus DEM GLO-30/90, current release 2024_1 (July 2024), is the best
    plain DSM but is not bare-earth; kept here only as a DSM reference so the
    canopy bias can be measured.
  SRTM (Feb 2000), NASADEM (Feb 2020), ASTER GDEM v3 (Aug 2019), GMTED2010,
    MERIT, AW3D30 v4.1 (Apr 2024) - all superseded in accuracy, and the SRTM
    family stops at 60N.

IMPORTANT modelling note (measured in validate_topography_alignment.py):
    Elevation may be averaged when coarsening to 10 arc-minutes, but slope and
    aspect MUST be computed at the DEM's native resolution and only then
    aggregated. Deriving slope from an already-coarsened DEM flattens it by
    roughly an order of magnitude. Aspect is circular, so aggregate it as
    northness = cos(aspect) and eastness = sin(aspect), never as a mean angle.

Default run is a SAMPLE download (~350 MB). See FULL_DOWNLOAD_RECIPE for the
global production route.

Only the regridded global elevation layer is written to topography/. Everything
else here exists to prove the source choice and lands in validation/topography/.
"""

import os
import sys
import zipfile
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from repo_paths import data

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUTPUT_DIR = data("topography")          # production layers only
VALIDATION_DIR = data("validation", "topography")   # evidence, never read by the model
CLIMATE_TEMPLATE = data("climate_current", "wc2.1_10m_bio_1.tif")   # the target grid

MODE = "sample"      # options:
                     #   "sample" = representative windows/tiles (~350 MB) ← default
                     #   "full"   = global production layers (hours, ~50 GB scratch)

# Which sources to fetch.
GET_GEDTM30_GLOBAL = True    # ~10 MB  - global elevation ON the target grid
GET_GEDTM30_NATIVE = True    # ~150 MB - 1 arc-sec windows for slope/aspect
GET_GEDTM30_SLOPE = True     # ~15 MB  - their published slope, to cross-check ours
GET_COPERNICUS_90 = True     # ~23 MB  - DSM reference, for canopy-bias measurement
GET_COPERNICUS_30 = True     # ~42 MB  - resolution-sensitivity check
GET_WORLDCLIM_ELEV = True    # 1.3 MB  - WorldClim's own elev, grid sanity check
GET_GMTED2010 = False        # ~35 MB  - legacy 30 arc-sec, comparison only
GET_SRTM_CGIAR = False       # ~39 MB  - legacy, documents the 60N cutoff

# GDAL settings that make reading a 432 GB remote COG practical.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "200000000",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

# ─────────────────────────────────────────────────────
# SOURCE 1 — GEDTM30 v1.2  (PRIMARY, released 6 March 2026)
# ─────────────────────────────────────────────────────
# One global Cloud-Optimised GeoTIFF per layer, served anonymously over HTTPS
# with byte-range support. Grid: 1,296,009 x 540,009 at exactly 1 arc-second,
# EPSG:4326, float32 metres, nodata 3.4028235e38, vertical datum EGM2008
# (EPSG:3855) embedded in the file. Extent -180.00125/-65.00125 to
# 180.00125/85.00125 -- note the 4.5-pixel overhang, handled in
# gedtm30_to_target_grid() by warping rather than by naive slicing.
GEDTM30_VERSION = "v1.2"
GEDTM30_BASE = "https://s3.opengeohub.org/global/dtm"

GEDTM30_DTM_URL = (f"{GEDTM30_BASE}/{GEDTM30_VERSION}/"
                   "gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif")
# Per-pixel prediction uncertainty (random-forest tree spread), same grid.
GEDTM30_STD_URL = (f"{GEDTM30_BASE}/{GEDTM30_VERSION}/"
                   "gedtm_rf_std_30m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif")

# Published land-surface parameters live under v1.2.0 and are stored as scaled
# integers. Slope is UInt16, scale 100, so degrees = value / 100.
# NOTE: there is NO aspect layer among the 15 parameters, so northness and
# eastness still have to be derived from the DTM itself.
GEDTM30_LSP_BASE = f"{GEDTM30_BASE}/v1.2.0"
GEDTM30_SLOPE_URL = (f"{GEDTM30_LSP_BASE}/"
                     "slope.in.degree_gedtm_m_30m_s_20060101_20151231_"
                     "go_epsg.4326_v1.2.0.tif")
GEDTM30_SLOPE_SCALE = 100.0
GEDTM30_SLOPE_NODATA = 65535

# Machine-readable index of every GEDTM30 layer and grid spacing.
GEDTM30_COG_LIST = ("https://codeberg.org/openlandmap/GEDTM30/raw/branch/main"
                    "/metadata/cog_list.csv")

# Overview level used to build the global 10-arcmin elevation layer.
# Level 5 = decimation factor 32 -> 64 arc-second (~2 km) pixels, which is
# still 9.4x finer than the 600 arc-second target, so block-averaging it loses
# nothing that survives at 10 arc-minutes. Reading the native 1 arc-second
# mosaic globally would mean ~2.7 million HTTP block fetches.
GEDTM30_GLOBAL_OVERVIEW = 5

# Sample windows: 1 deg x 1 deg, chosen to exercise every hard case we have.
# (name, west, south)  -- each window is 1 degree on a side.
SAMPLE_WINDOWS = [
    ("alps_N46E008",     8.0,  46.0),   # Alps: extreme relief, slope/aspect test
    ("finnmark_N70E025", 25.0,  70.0),  # 70N, past SRTM's 60N limit
    ("taymyr_N72E101",  101.0,  72.0),  # 72N, the Betula glandulosa cluster
    ("amazon_S03W060",  -60.0,  -3.0),  # dense rainforest: DSM canopy bias
    ("capetown_S34E018", 18.0, -34.0),  # southern hemisphere + coastal ocean
]

# ─────────────────────────────────────────────────────
# SOURCE 2 — Copernicus DEM GLO-90 / GLO-30 (DSM reference)
# ─────────────────────────────────────────────────────
# AWS Open Data, anonymous HTTPS, no Earthdata login. Current release 2024_1
# (July 2024). TanDEM-X 2011-2015. EGM2008. LAND TILES ONLY: an all-ocean
# tile returns HTTP 404, and ocean pixels inside a land tile are a literal
# 0.0 m with no nodata flag set.
# Kept so we can MEASURE the surface-vs-terrain canopy bias, not as the
# elevation source.
COP90_BASE = "https://copernicus-dem-90m.s3.amazonaws.com"
COP30_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
COP90_TILELIST = f"{COP90_BASE}/tileList.txt"   # 26,475 tiles
COP30_TILELIST = f"{COP30_BASE}/tileList.txt"

COP90_SAMPLE_TILES = [
    "N46_00_E008_00",   # Alps        - pairs with alps_N46E008
    "N70_00_E025_00",   # Finnmark    - pairs with finnmark_N70E025
    "N72_00_E101_00",   # Taymyr      - pairs with taymyr_N72E101
    "S03_00_W060_00",   # Amazon      - pairs with amazon_S03W060
    "S34_00_E018_00",   # Cape Town   - pairs with capetown_S34E018
    "N00_00_E037_00",   # Mount Kenya - equatorial relief
]
COP30_SAMPLE_TILES = ["N46_00_E008_00"]


def copernicus_url(tile, resolution):
    """Build an anonymous-HTTPS Copernicus DEM tile URL.

    The "30"/"10" in the product name is arc-seconds, NOT metres:
    COG_30 = 3 arc-sec = GLO-90, COG_10 = 1 arc-sec = GLO-30.
    """
    if resolution == 90:
        name = f"Copernicus_DSM_COG_30_{tile}_DEM"
        return f"{COP90_BASE}/{name}/{name}.tif", f"{name}.tif"
    name = f"Copernicus_DSM_COG_10_{tile}_DEM"
    return f"{COP30_BASE}/{name}/{name}.tif", f"{name}.tif"


# ─────────────────────────────────────────────────────
# SOURCE 3 — WorldClim 2.1 elevation (grid reference only)
# ─────────────────────────────────────────────────────
# Already pixel-identical to climate_current/, which makes it the perfect
# control for proving our GEDTM30 regrid landed on the right cells. Its own
# heritage is SRTM + GMTED, i.e. a surface model, so it is NOT the production
# elevation layer -- it is the ruler we measure with.
WORLDCLIM_RESOLUTION = "10m"
WORLDCLIM_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
WORLDCLIM_ELEV_FILE = f"wc2.1_{WORLDCLIM_RESOLUTION}_elev.zip"
WORLDCLIM_ELEV_URL = f"{WORLDCLIM_BASE}/{WORLDCLIM_ELEV_FILE}"

# ─────────────────────────────────────────────────────
# SOURCE 4 — legacy products (off by default, for comparison)
# ─────────────────────────────────────────────────────
GMTED_BASE = ("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo"
              "/downloads/GMTED/Global_tiles_GMTED")
GMTED_ARCSEC = "300darcsec"
GMTED_STAT = "mea"
GMTED_SAMPLE_TILES = [("30N000E", "E000"), ("70N000E", "E000")]
GMTED_GLOBAL_ZIP = ("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/topo"
                    "/downloads/GMTED/Grid_ZipFiles/mn30_grd.zip")

# CGIAR-CSI rehosts void-filled SRTM v4.1 with no Earthdata login.
# 5 deg tiles: x = floor((lon+180)/5)+1, y = floor((60-lat)/5)+1, y max 24,
# so there is no tile north of 60N at all.
SRTM_BASE = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF"
SRTM_SAMPLE_TILES = ["srtm_38_03"]   # 5-10E, 45-50N, the same Alps footprint


def gmted_url(tile, lon_dir):
    fname = f"{tile}_20101117_gmted_{GMTED_STAT}{GMTED_ARCSEC[:3]}.tif"
    return f"{GMTED_BASE}/{GMTED_ARCSEC}/{GMTED_STAT}/{lon_dir}/{fname}", fname


def srtm_url(tile):
    return f"{SRTM_BASE}/{tile}.zip", f"{tile}.zip"


# ─────────────────────────────────────────────────────
# DOWNLOAD HELPERS
# ─────────────────────────────────────────────────────
def download_file(url, output_path, desc="Downloading"):
    """Download with progress bar. Returns True on success."""
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"ERROR: could not download (status {response.status_code})")
        print(f"URL tried: {url}")
        return False

    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return True


def download_and_unzip(url, zip_path, extract_dir, desc="Downloading"):
    """Download a zip, extract it, delete the zip."""
    if not download_file(url, zip_path, desc=desc):
        return False
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    os.remove(zip_path)
    return True


def skip_if_present(path, label):
    """Avoid re-downloading on repeat runs."""
    if os.path.exists(path):
        print(f"  [have] {label}  ({os.path.getsize(path) / 1e6:.1f} MB)")
        return True
    return False


def write_geotiff(path, array, transform, crs, nodata, dtype=None):
    """Write a single-band DEFLATE-compressed GeoTIFF."""
    array = array if dtype is None else array.astype(dtype)
    with rasterio.open(path, "w", driver="GTiff",
                       height=array.shape[0], width=array.shape[1], count=1,
                       dtype=array.dtype, crs=crs, transform=transform,
                       nodata=nodata, compress="deflate", predictor=2,
                       tiled=True, blockxsize=512, blockysize=512) as dst:
        dst.write(array, 1)
    return path


# ─────────────────────────────────────────────────────
# GEDTM30 READERS  (COG range reads, no bulk download)
# ─────────────────────────────────────────────────────
def gedtm30_window(url, west, south, east, north, out_path,
                   nodata=None, desc="window"):
    """Pull one lon/lat window out of the global GEDTM30 COG at NATIVE 1 arc-sec.

    Only the intersecting internal COG blocks travel over the network, so a
    1 deg window costs a few MB out of a 432 GB file.
    """
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(f"/vsicurl/{url}") as src:
            win = from_bounds(west, south, east, north, src.transform)
            win = win.round_offsets().round_lengths()
            arr = src.read(1, window=win)
            tr = src.window_transform(win)
            nd = src.nodata if nodata is None else nodata
            crs = src.crs
    print(f"  {desc}: {arr.shape[1]} x {arr.shape[0]} px at 1 arc-sec")
    return write_geotiff(out_path, arr, tr, crs, nd)


def gedtm30_to_target_grid(url, template_path, out_path,
                           overview_level=GEDTM30_GLOBAL_OVERVIEW):
    """Regrid global GEDTM30 onto the EXACT climate grid by area-averaging.

    Reads a decimated overview of the global COG and reprojects it onto the
    template's transform with Resampling.average. Using reproject (rather than
    slicing) is what correctly absorbs GEDTM30's 4.5-pixel grid overhang and
    its 85N/65S extent, and guarantees the output is cell-for-cell on the
    climate grid.

    Averaging is the right reduction for ELEVATION. It is the WRONG reduction
    for slope and aspect, which is why those are handled separately from the
    native-resolution windows.
    """
    with rasterio.open(template_path) as tmpl:
        dst_transform, dst_crs = tmpl.transform, tmpl.crs
        dst_h, dst_w = tmpl.height, tmpl.width

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(f"/vsicurl/{url}",
                           OVERVIEW_LEVEL=overview_level) as src:
            print(f"  source overview {overview_level}: {src.width} x "
                  f"{src.height} ({src.res[0] * 3600:.0f} arc-sec)")
            src_arr = src.read(1)
            src_transform, src_crs, src_nodata = (src.transform, src.crs,
                                                  src.nodata)

    # Mask nodata before averaging so ocean does not drag land values down.
    src_arr = np.where(src_arr > 1e30, np.nan, src_arr).astype("float32")

    dst_arr = np.full((dst_h, dst_w), np.nan, dtype="float32")
    reproject(source=src_arr, destination=dst_arr,
              src_transform=src_transform, src_crs=src_crs,
              src_nodata=np.nan,
              dst_transform=dst_transform, dst_crs=dst_crs,
              dst_nodata=np.nan,
              resampling=Resampling.average)

    filled = np.where(np.isnan(dst_arr), -32768.0, dst_arr)
    print(f"  regridded to {dst_w} x {dst_h} on the climate grid; "
          f"{100.0 * np.isfinite(dst_arr).mean():.2f}% of cells have terrain")
    return write_geotiff(out_path, filled, dst_transform, dst_crs, -32768.0,
                         dtype="float32")


# ─────────────────────────────────────────────────────
# SAMPLE MODE
# ─────────────────────────────────────────────────────
def fetch_sample():
    """Download a small, representative set (~350 MB)."""
    got = []
    ged_dir = os.path.join(OUTPUT_DIR, "gedtm30")
    os.makedirs(ged_dir, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)

    if GET_GEDTM30_GLOBAL:
        print("\n[1/6] GEDTM30 v1.2 -> global elevation on the climate grid")
        out = os.path.join(ged_dir, "gedtm30_v1.2_elev_10m.tif")
        if not skip_if_present(out, os.path.basename(out)):
            print(f"  URL: {GEDTM30_DTM_URL}")
            gedtm30_to_target_grid(GEDTM30_DTM_URL, CLIMATE_TEMPLATE, out)
        got.append(out)

    if GET_GEDTM30_NATIVE:
        print(f"\n[2/6] GEDTM30 v1.2 -> {len(SAMPLE_WINDOWS)} native "
              "1 arc-sec windows (for slope/aspect)")
        sub = os.path.join(VALIDATION_DIR, "gedtm30_native")
        os.makedirs(sub, exist_ok=True)
        for name, west, south in SAMPLE_WINDOWS:
            out = os.path.join(sub, f"gedtm30_{name}.tif")
            if not skip_if_present(out, os.path.basename(out)):
                gedtm30_window(GEDTM30_DTM_URL, west, south,
                               west + 1.0, south + 1.0, out, desc=name)
            got.append(out)

    if GET_GEDTM30_SLOPE:
        print("\n[3/6] GEDTM30 v1.2 -> published SLOPE window "
              "(cross-check for our own derivation)")
        sub = os.path.join(VALIDATION_DIR, "gedtm30_slope_published")
        os.makedirs(sub, exist_ok=True)
        name, west, south = SAMPLE_WINDOWS[0]
        out = os.path.join(sub, f"gedtm30_slope_{name}.tif")
        if not skip_if_present(out, os.path.basename(out)):
            print(f"  URL: {GEDTM30_SLOPE_URL}")
            print(f"  (UInt16, degrees = value / {GEDTM30_SLOPE_SCALE:.0f})")
            gedtm30_window(GEDTM30_SLOPE_URL, west, south,
                           west + 1.0, south + 1.0, out,
                           nodata=GEDTM30_SLOPE_NODATA, desc=f"slope {name}")
        got.append(out)

    if GET_COPERNICUS_90:
        print(f"\n[4/6] Copernicus DEM GLO-90 release 2024_1 "
              f"({len(COP90_SAMPLE_TILES)} tiles, DSM reference)")
        sub = os.path.join(VALIDATION_DIR, "copernicus_glo90")
        os.makedirs(sub, exist_ok=True)
        for tile in COP90_SAMPLE_TILES:
            url, fname = copernicus_url(tile, 90)
            out = os.path.join(sub, fname)
            if not skip_if_present(out, fname):
                download_file(url, out, desc=tile)
            got.append(out)

    if GET_COPERNICUS_30:
        print(f"\n[5/6] Copernicus DEM GLO-30 release 2024_1 "
              f"({len(COP30_SAMPLE_TILES)} tiles)")
        sub = os.path.join(VALIDATION_DIR, "copernicus_glo30")
        os.makedirs(sub, exist_ok=True)
        for tile in COP30_SAMPLE_TILES:
            url, fname = copernicus_url(tile, 30)
            out = os.path.join(sub, fname)
            if not skip_if_present(out, fname):
                download_file(url, out, desc=tile)
            got.append(out)

    if GET_WORLDCLIM_ELEV:
        print("\n[6/6] WorldClim 2.1 elevation (grid control)")
        out = os.path.join(VALIDATION_DIR,
                           f"wc2.1_{WORLDCLIM_RESOLUTION}_elev.tif")
        if not skip_if_present(out, os.path.basename(out)):
            download_and_unzip(WORLDCLIM_ELEV_URL,
                               os.path.join(VALIDATION_DIR,
                                            WORLDCLIM_ELEV_FILE),
                               VALIDATION_DIR, desc="wc elev")
        got.append(out)

    if GET_GMTED2010:
        print(f"\n[extra] GMTED2010 {GMTED_ARCSEC}/{GMTED_STAT} (legacy)")
        sub = os.path.join(VALIDATION_DIR, "gmted2010")
        os.makedirs(sub, exist_ok=True)
        for tile, lon_dir in GMTED_SAMPLE_TILES:
            url, fname = gmted_url(tile, lon_dir)
            out = os.path.join(sub, fname)
            if not skip_if_present(out, fname):
                download_file(url, out, desc=tile)
            got.append(out)

    if GET_SRTM_CGIAR:
        print("\n[extra] CGIAR-CSI SRTM 90m v4.1 (legacy, 60N-56S only)")
        sub = os.path.join(VALIDATION_DIR, "srtm_cgiar")
        os.makedirs(sub, exist_ok=True)
        for tile in SRTM_SAMPLE_TILES:
            url, zname = srtm_url(tile)
            out = os.path.join(sub, f"{tile}.tif")
            if not skip_if_present(out, f"{tile}.tif"):
                download_and_unzip(url, os.path.join(sub, zname), sub,
                                   desc=tile)
            got.append(out)

    return got


# ─────────────────────────────────────────────────────
# FULL MODE
# ─────────────────────────────────────────────────────
FULL_DOWNLOAD_RECIPE = """
FULL-GLOBAL PRODUCTION RECIPE
=============================

You never need to download GEDTM30 in bulk. The global mosaic is a 432 GB
Cloud-Optimised GeoTIFF with HTTP range support, so the production job is a
loop of windowed reads:

  DTM   {dtm}
  slope {slope}
  index {coglist}

Elevation (cheap, minutes):
  Read overview level {ovr} (64 arc-second) and area-average onto the climate
  grid. That is exactly what gedtm30_to_target_grid() does, and MODE="sample"
  already produces the final global product - there is nothing extra to do.

Slope / aspect / ruggedness (the expensive part, hours):
  Iterate 1 deg x 1 deg windows over land, and for EACH window:
    1. read a 1-arc-second window with a >=1 pixel HALO on all four sides
    2. compute slope and aspect per native pixel (Horn 3x3), using
         dx = res_lon * 111320 * cos(lat)     dy = res_lat * 110540
    3. reduce each 10-arcmin cell (600 x 600 native pixels) to
         elev_mean, elev_std, elev_range,
         slope_mean, slope_p90,
         northness = sum(sin(slope)*cos(aspect)) / sum(sin(slope)),
         eastness  = sum(sin(slope)*sin(aspect)) / sum(sin(slope))
    4. DISCARD the window before moving on - never hold a global 1 arc-sec
       mosaic in memory or on disk
  At 600 x 600 = 360,000 native pixels per target cell there is no sampling
  noise left; you could decimate to every 2nd pixel and halve the runtime
  with no measurable change in the aggregates.

  Shortcut: GEDTM30 also publishes SLOPE globally at 30/60/120/240/480/960 m.
  Those layers are on DIFFERENT grids from the DTM (the 30 m slope layer is
  84N-57S, and the 960 m layer is 32 arc-second, which does not divide the
  600 arc-second target evenly), so they must be warped, not sliced. Use them
  to validate your own slope, as validate_topography_alignment.py does. There
  is NO published aspect layer, so northness/eastness must be derived either
  way - which means you are running the loop above regardless.

Optional extras worth their bytes:
  * GEDTM30 uncertainty layer {std}
    - flags cells where the terrain prediction is itself unreliable
      (dense canopy, steep slopes); useful as a QA mask or a model weight.
  * Copernicus DEM GLO-90 {cop90list}
    - 26,475 land tiles, ~350 GB. Only needed if you want to quantify the
      canopy bias, which the sample already does on five windows.

DO NOT bother with:
  Copernicus GLO-30 global (~1.5 TB) at a 10-arcmin target - pointless.
  The GEDTM30 240 m Zenodo files (10.1 GB each) - the COG route is smaller
  and gives you native resolution where it actually matters.
""".format(dtm=GEDTM30_DTM_URL, slope=GEDTM30_SLOPE_URL,
           coglist=GEDTM30_COG_LIST, std=GEDTM30_STD_URL,
           cop90list=COP90_TILELIST, ovr=GEDTM30_GLOBAL_OVERVIEW)


def fetch_full():
    """Global elevation layer plus the GEDTM30 uncertainty layer."""
    print("\nFULL MODE")
    ged_dir = os.path.join(OUTPUT_DIR, "gedtm30")
    os.makedirs(ged_dir, exist_ok=True)
    got = []

    print("\n[1/2] global elevation on the climate grid")
    out = os.path.join(ged_dir, "gedtm30_v1.2_elev_10m.tif")
    if not skip_if_present(out, os.path.basename(out)):
        gedtm30_to_target_grid(GEDTM30_DTM_URL, CLIMATE_TEMPLATE, out)
    got.append(out)

    print("\n[2/2] global prediction uncertainty on the climate grid")
    out = os.path.join(ged_dir, "gedtm30_v1.2_elev_std_10m.tif")
    if not skip_if_present(out, os.path.basename(out)):
        gedtm30_to_target_grid(GEDTM30_STD_URL, CLIMATE_TEMPLATE, out)
    got.append(out)

    print("\nSlope/aspect still need the native-resolution loop:")
    print(FULL_DOWNLOAD_RECIPE)
    return got


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("TOPOGRAPHY RASTER DOWNLOADER")
    print("=" * 50)
    print(f"Primary:    GEDTM30 {GEDTM30_VERSION} (released 2026-03-06, CC-BY-4.0)")
    print("            bare-earth DTM, 1 arc-sec, 85N-65S")
    print(f"Mode:       {MODE}")
    print(f"Output dir: {OUTPUT_DIR}/ (production), "
          f"{VALIDATION_DIR}/ (evidence)")
    print(f"Target grid: {CLIMATE_TEMPLATE}")
    print("Variables:  elevation (regridded), slope + aspect (derived native)")

    if not os.path.exists(CLIMATE_TEMPLATE):
        print(f"\nERROR: {CLIMATE_TEMPLATE} not found - it defines the target "
              "grid.\nRun current_climate_rasters.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paths = fetch_full() if MODE == "full" else fetch_sample()

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)

    total = 0
    for d in (OUTPUT_DIR, VALIDATION_DIR):
        for root, _, files in os.walk(d):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
    for p in paths:
        if os.path.isfile(p):
            print(f"  {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    print(f"\nTotal in {OUTPUT_DIR}/ + {VALIDATION_DIR}/: "
          f"{total / 1e6:.1f} MB")

    print("\nNext: python3 validate_topography_alignment.py")
    print("      (grid alignment, GBIF/land-cell coverage, canopy bias, and")
    print("       the correct native-resolution slope/aspect derivation)")

    if MODE != "full":
        print(FULL_DOWNLOAD_RECIPE)


if __name__ == "__main__":
    main()
