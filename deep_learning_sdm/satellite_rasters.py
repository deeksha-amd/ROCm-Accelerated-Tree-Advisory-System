"""
Download SATELLITE land-cover and vegetation rasters and align them to the
WorldClim 10-arcminute grid

Land cover  : ESA CCI Land Cover "Plant Functional Type" v2.0.81, epoch 2020.
              300 m, EPSG:4326, and already expressed as *per-class cover
              fractions* rather than class codes, which is what an 18.5 km cell
              needs -- averaging a fraction is exact, averaging a class code is
              meaningless.  14 fractions sum to exactly 100 on every land pixel.
Vegetation  : PKU GIMMS NDVI v1.2 (AVHRR+MODIS consolidated), semi-monthly,
              1/12 degree, epoch 2015-2022.  Summarised into four temporal
              statistics, because a single annual mean throws away the
              seasonality that separates an evergreen forest from a monsoon
              grassland.
Soil moist. : ESA CCI Soil Moisture v09.2 COMBINED, daily 0.25 degree,
              epoch 2015-2024.  Included because it is the one satellite
              variable here that is *not* a vegetation proxy, so it is the only
              one that is safe to feed the model as a predictor (see
              validate_satellite_alignment.py for the measurement behind that).
Cross-check : ESA WorldCover v200 (10 m, 2021) and Hansen Global Forest Change
              v1.12 (30 m, 2024) on a sample of tiles.  Neither can be
              aggregated globally inside a 4 GB download budget, so they are
              used to verify the 300 m fractions rather than to produce them.

None of these needs credentials.  Google Earth Engine, which the design
document suggests, does -- and pip is blocked here -- so everything below is
plain HTTPS.

READ THE CIRCULARITY WARNING: tree-cover fraction and every NDVI layer written
by this script describe *where trees already grow*.  They are provided as
filters and as diagnostics for "current green gaps", not as model predictors.
"""

import os
import json
import struct
import warnings
import zlib
import concurrent.futures as futures

import numpy as np
import rasterio
import requests
from affine import Affine
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

# The netCDF sources carry their grid in coordinate variables rather than in a
# geotransform, so rasterio warns on every open.  We set the transform by hand.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUTPUT_DIR = "satellite"
TEMPLATE = "climate_current/wc2.1_10m_bio_1.tif"   # defines the target grid

MODE = "sample"      # "sample" = 2 NDVI years, 1 soil-moisture day/month over
                     #            3 years, 4 WorldCover tiles, 1 Hansen tile
                     #            (~2.1 GB download, ~25 min)
                     # "full"   = 8 NDVI years, 2 days/month over 10 years,
                     #            12 WorldCover tiles, 2 Hansen tiles
                     #            (~3.0 GB download, ~45 min)

DOWNLOAD_THREADS = 12    # parallel HTTP range requests per large file
WORKER_THREADS = 16      # parallel small-file fetches

# ── Land cover: ESA CCI LC Plant Functional Types (the global product) ──
# Chosen over ESA WorldCover (10 m) and GLC_FCS30D (30 m) because it is the only
# public, credential-free global product that ships *pre-computed fractions*.
# WorldCover's internal overviews are mode-resampled, so deriving fractions from
# them under-reports minority classes badly (measured: tree -17%, built-up -33%
# at 320 m); reading WorldCover fine enough to avoid that costs >8 GB.
PFT_BASE = ("https://dap.ceda.ac.uk/neodc/esacci/land_cover/data/pft/v2.0.81")
PFT_VERSION = "v2.0.81"
PFT_YEAR = 2020          # last year published; sits inside climate_recent
PFT_FILE = f"ESACCI-LC-L4-PFT-Map-300m-P1Y-{PFT_YEAR}-{PFT_VERSION}.nc"
PFT_WIDTH, PFT_HEIGHT = 129600, 64800     # 300 m global, north-up, -180/90
PFT_BLOCK = 8100         # native rows/cols per processing block.  8100 is the
                         # least common multiple of the file's 2025-pixel HDF5
                         # chunk and the 60-pixel target cell, so every chunk is
                         # decompressed exactly once and no target cell is split.

# Our layer name -> the PFT variables that make it up.
PFT_COMPONENTS = {
    "tree":            ["TREES-BD", "TREES-BE", "TREES-ND", "TREES-NE"],
    "tree_broadleaf":  ["TREES-BD", "TREES-BE"],
    "tree_needleleaf": ["TREES-ND", "TREES-NE"],
    "shrub":           ["SHRUBS-BD", "SHRUBS-BE", "SHRUBS-ND", "SHRUBS-NE"],
    "grass_natural":   ["GRASS-NAT"],
    "cropland":        ["GRASS-MAN"],   # "managed grass" is CCI's cropland class
    "builtup":         ["BUILT"],
    "bare":            ["BARE"],
    "water":           ["WATER_INLAND"],
    "snowice":         ["SNOWICE"],
}
# The subset that partitions the land surface, i.e. sums to 100.  Used for the
# majority class, for purity, and as the sanity check in validation.
PFT_PARTITION = ["tree", "shrub", "grass_natural", "cropland",
                 "builtup", "bare", "water", "snowice"]
# Codes follow ESA WorldCover so the numbers mean the same thing to a reader who
# has seen that product.
CLASS_CODES = {"tree": 10, "shrub": 20, "grass_natural": 30, "cropland": 40,
               "builtup": 50, "bare": 60, "snowice": 70, "water": 80}

# ── Vegetation index: PKU GIMMS NDVI v1.2 ──
# Public Zenodo record, CC-BY 4.0, no credentials.  MODIS MOD13/MYD13 and VIIRS
# VNP13 are finer but sit behind a NASA Earthdata login; Copernicus Global Land
# NDVI needs a VITO account.  Individual GeoTIFFs are pulled out of the Zenodo
# zip with HTTP range requests so the 775 MB archive is never downloaded whole.
NDVI_ZIP = ("https://zenodo.org/records/8253971/files/"
            "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_2011_2022.zip")
NDVI_VERSION = "V1.2"
NDVI_SCALE = 0.001       # stored as uint16; 65535 is the fill value
NDVI_FILL = 65535
NDVI_WIDTH, NDVI_HEIGHT = 4320, 2160     # 1/12 degree, exactly 2x the target
NDVI_YEARS_FULL = list(range(2015, 2023))    # overlaps climate_recent 2015-2024
NDVI_YEARS_SAMPLE = [2021, 2022]
# Growing season = the greenest half of the year.  Defined per cell rather than
# per calendar month so it works in both hemispheres and in monsoon climates.
NDVI_GROWING_SEMIMONTHS = 12   # of 24
# The source masks a composite wherever NDVI is not retrievable: snow, ice,
# water and barren desert.  Above about 60 degrees latitude that removes most of
# the winter, so a cell there legitimately has only 10 to 14 of the 24
# composites.  Demanding a full year would silently delete the boreal zone and
# with it the treeline records this project cares about, so the bar is one
# four-month green season.  ndvi_observed_semimonths_10m.tif records the real
# count per cell, and the layer is named "observed mean" rather than "annual
# mean" because outside the tropics that is honestly what it is.
NDVI_MIN_SEMIMONTHS = 8        # of 24

# ── Surface soil moisture: ESA CCI Soil Moisture v09.2 COMBINED ──
# The non-vegetation satellite variable.  Daily 0.25 degree files are ~2.5 MB,
# so a fixed day-of-month sample gives a climatology cheaply.  SMAP and MODIS
# snow cover would both need a NASA Earthdata login.
SM_BASE = ("https://dap.ceda.ac.uk/neodc/esacci/soil_moisture/data/"
           "daily_files/COMBINED/v09.2")
SM_VERSION = "v09.2"
SM_WIDTH, SM_HEIGHT = 1440, 720          # 0.25 degree, north-up, -180/90
SM_YEARS_FULL = list(range(2015, 2025))
SM_YEARS_SAMPLE = [2022, 2023, 2024]
SM_DAYS_FULL = [8, 22]                   # day-of-month sample
SM_DAYS_SAMPLE = [15]

# ── Cross-check sources (sampled tiles only, never global) ──
WC_BASE = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
           "v200/2021/map")
WC_VERSION = "v200 (2021)"
WC_OVERVIEW = 4          # read the 40 m overview; at 320 m the mode-resampled
                         # overviews distort fractions too much to compare
# 3x3 degree tiles named by their south-west corner, spread across biomes.
WC_TILES_FULL = ["N00E018", "N39W006", "N51E000", "N60E024", "N18E078",
                 "S03W060", "S24E027", "N33W087", "S33E147", "N21W102",
                 "N63W123", "S09W072"]
WC_TILES_SAMPLE = ["N00E018", "N39W006", "N60E024", "S03W060"]
WC_CLASS_TO_LAYER = {10: "tree", 95: "tree", 20: "shrub", 30: "grass_natural",
                     90: "grass_natural", 100: "grass_natural", 40: "cropland",
                     50: "builtup", 60: "bare", 70: "snowice", 80: "water"}

HANSEN_VERSION = "GFC-2025-v1.13"     # newest; tiles published March 2026,
                                      # forest loss now runs through 2025
HANSEN_BASE = ("https://storage.googleapis.com/earthenginepartners-hansen/"
               f"{HANSEN_VERSION}")
# 10x10 degree tiles named by their north-west corner.  Tile size swings from
# 20 MB over ocean to 741 MB over the Congo, because canopy cover only
# compresses well when it is uniform, so the tiles are picked for a wide canopy
# gradient rather than for maximum forest: 40N_010W runs from bare Morocco to
# Iberian woodland, 30N_070E from desert to irrigated plain.
HANSEN_TILES_FULL = ["40N_010W", "30N_070E"]
HANSEN_TILES_SAMPLE = ["40N_010W"]
HANSEN_DECIMATE = 8     # read every 8th pixel (~220 m); the tiles are striped
                        # rather than tiled so a full read is 20 MB either way

# ── Planting-exclusion mask thresholds (percent of the cell) ──
# This is the *intended* use of vegetation-derived satellite data in this
# project: not as a predictor, but to strike out land that is unavailable.
EXCLUDE_OCEAN_BELOW_LAND_PCT = 50.0
EXCLUDE_WATER_ABOVE_PCT = 50.0
EXCLUDE_SNOWICE_ABOVE_PCT = 50.0
EXCLUDE_BUILTUP_ABOVE_PCT = 50.0
EXCLUDE_CLOSED_FOREST_ABOVE_PCT = 60.0
FLAG_CROPLAND_ABOVE_PCT = 50.0

MASK_BITS = {
    "ocean": 1,           # cell is mostly sea
    "inland_water": 2,
    "permanent_snow_ice": 4,
    "built_up": 8,
    "closed_canopy_forest": 16,
    "cropland": 32,       # advisory only: competing land use, not unplantable
}


def download_file(url, output_path, threads=1, attempts=5, quiet=False):
    """Download with a progress bar, resuming or parallelising by byte range.

    CEDA gives about 0.9 MB/s on one connection, so large files are split into
    `threads` ranges.  A file already on disk at the right size is left alone,
    which makes reruns cheap.
    """
    try:
        head = requests.head(url, allow_redirects=True, timeout=60)
    except requests.exceptions.RequestException as exc:
        if not quiet:
            print(f"ERROR: {type(exc).__name__} on HEAD {url}")
        return False
    if head.status_code != 200:
        if not quiet:
            print(f"ERROR: could not download (status {head.status_code})")
            print(f"URL tried: {url}")
        return False
    total_size = int(head.headers.get("content-length", 0))

    if os.path.exists(output_path) and total_size and \
            os.path.getsize(output_path) == total_size:
        return True

    desc = os.path.basename(output_path)[:38]
    if threads > 1 and total_size:
        chunk = (total_size + threads - 1) // threads

        def grab(index):
            start = index * chunk
            stop = min(start + chunk, total_size) - 1
            for _ in range(attempts):
                try:
                    r = requests.get(url, timeout=600,
                                     headers={"Range": f"bytes={start}-{stop}"})
                    if len(r.content) == stop - start + 1:
                        return index, r.content
                except requests.exceptions.RequestException:
                    continue
            raise RuntimeError(f"range {start}-{stop} failed for {url}")

        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc,
                  disable=quiet) as pbar:
            parts = {}
            with futures.ThreadPoolExecutor(threads) as pool:
                for index, blob in pool.map(grab, range(threads)):
                    parts[index] = blob
                    pbar.update(len(blob))
            with open(output_path, "wb") as f:
                for index in range(threads):
                    f.write(parts[index])
        return os.path.getsize(output_path) == total_size

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc,
              disable=quiet) as pbar:
        for attempt in range(attempts):
            done = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            if total_size and done >= total_size:
                return True
            pbar.n = done
            pbar.refresh()
            headers = {"Range": f"bytes={done}-"} if done else {}
            try:
                r = requests.get(url, stream=True, timeout=300, headers=headers)
                with open(output_path, "ab" if done else "wb") as f:
                    for block in r.iter_content(chunk_size=1 << 16):
                        f.write(block)
                        pbar.update(len(block))
            except requests.exceptions.RequestException as exc:
                print(f"  retry {attempt + 1}/{attempts} after {type(exc).__name__}")
                continue
            if not total_size or os.path.getsize(output_path) >= total_size:
                return True

    if not quiet:
        print(f"ERROR: incomplete download after {attempts} attempts: {url}")
    return False


# ─────────────────────────────────────────────────────
# GRID ALIGNMENT
# ─────────────────────────────────────────────────────
def read_template(path):
    """Return the target grid definition taken from a WorldClim raster."""
    with rasterio.open(path) as src:
        return {"width": src.width, "height": src.height,
                "transform": src.transform, "crs": src.crs}


def write_aligned(array, path, grid, tags, dtype="float32", nodata=np.nan):
    """Write one layer on the target grid."""
    with rasterio.open(path, "w", driver="GTiff", height=grid["height"],
                       width=grid["width"], count=1, dtype=dtype,
                       crs=grid["crs"], transform=grid["transform"],
                       nodata=nodata, compress="deflate", tiled=True) as dst:
        dst.write(array.astype(dtype), 1)
        dst.update_tags(**tags)


def block_mean(array, factor):
    """Average an array down by an exact integer factor in both directions."""
    h, w = array.shape
    return array.reshape(h // factor, factor,
                         w // factor, factor).mean(axis=(1, 3))


# ─────────────────────────────────────────────────────
# SOURCE 1: ESA CCI LC plant functional types -> class fractions
# ─────────────────────────────────────────────────────
def pft_subdataset(path, variable):
    return f'netCDF:"{path}":{variable}'


def aggregate_landcover(path, grid, cache=None):
    """Area-weight the 300 m PFT fractions onto the target grid.

    Takes about twenty minutes because it decompresses the whole 300 m file once
    per variable, so the result is cached: only the land-cover stage is
    expensive, and reruns usually only change the NDVI or soil-moisture epoch.

    Each target cell covers exactly 60x60 native pixels, so this is a plain
    average over the pixels that are not sea.

    The one subtlety is which pixels those are.  The file offers two candidate
    masks and only one of them is right.  LAND is 100 minus *all* water, so a
    lake pixel has LAND = 0; weighting by it silently deletes inland water and
    reported Finland as 2 percent lake against WorldCover's 19 percent.
    WATER_OCEAN is the correct mask: at 300 m it is strictly 0 or 100, the
    fourteen class fractions sum to exactly 100 wherever it is 0 and to 0
    wherever it is 100, so "not sea" is unambiguous and inland water stays in
    as a class of the land surface.
    """
    if cache and os.path.exists(cache):
        print(f"  reusing cached aggregation {cache}")
        with np.load(cache) as stored:
            return {k: stored[k] for k in stored.files}

    factor = PFT_HEIGHT // grid["height"]          # 60
    assert factor == PFT_WIDTH // grid["width"]
    out_h, out_w = grid["height"], grid["width"]
    cells = PFT_BLOCK // factor                    # target cells per block edge

    variables = sorted({v for group in PFT_COMPONENTS.values() for v in group})
    totals = {v: np.zeros((out_h, out_w), dtype="float64") for v in variables}
    pixels = np.zeros((out_h, out_w), dtype="int64")

    blocks = [(r, c) for r in range(0, PFT_HEIGHT, PFT_BLOCK)
              for c in range(0, PFT_WIDTH, PFT_BLOCK)]

    handles = {v: rasterio.open(pft_subdataset(path, v)) for v in variables}
    handles["WATER_OCEAN"] = rasterio.open(
        pft_subdataset(path, "WATER_OCEAN"))
    try:
        for row0, col0 in tqdm(blocks, desc="ESA CCI LC PFT", unit="block"):
            window = ((row0, row0 + PFT_BLOCK), (col0, col0 + PFT_BLOCK))
            keep = handles["WATER_OCEAN"].read(1, window=window) == 0
            tr, tc = row0 // factor, col0 // factor
            sl = (slice(tr, tr + cells), slice(tc, tc + cells))

            counts = keep.reshape(cells, factor, cells, factor).sum(
                axis=(1, 3), dtype="int64")
            pixels[sl] += counts
            if counts.max() == 0:
                continue                      # whole block is open sea

            for v in variables:
                comp = handles[v].read(1, window=window).astype("int16")
                comp *= keep                  # sea pixels contribute nothing
                totals[v][sl] += comp.reshape(
                    cells, factor, cells, factor).sum(axis=(1, 3), dtype="int64")
    finally:
        for h in handles.values():
            h.close()

    valid = pixels > 0
    layers = {}
    for name, group in PFT_COMPONENTS.items():
        total = np.zeros((out_h, out_w), dtype="float64")
        for v in group:
            total += totals[v]
        frac = np.full((out_h, out_w), np.nan, dtype="float32")
        frac[valid] = (total[valid] / pixels[valid]).astype("float32")
        layers[name] = frac

    # Percent of the cell that is not sea.  Inland water counts towards it and
    # is then reported separately by the water class.
    land_frac = np.full((out_h, out_w), np.nan, dtype="float32")
    land_frac[valid] = (100.0 * pixels[valid]
                        / float(factor * factor)).astype("float32")
    layers["land"] = land_frac
    if cache:
        np.savez_compressed(cache, **layers)
    return layers


def derive_majority(layers):
    """Majority class code plus its purity, as an alternative to the fractions.

    Both representations are written.  The fractions are the primary product --
    a majority code alone hides that a cell is 45 percent forest and 40 percent
    cropland -- but a single code is convenient for maps and for grouping.
    """
    names = PFT_PARTITION
    stack = np.stack([layers[n] for n in names])
    valid = np.isfinite(stack[0])
    filled = np.where(np.isfinite(stack), stack, -1.0)
    winner = filled.argmax(axis=0)

    code = np.zeros(winner.shape, dtype="uint8")
    for i, n in enumerate(names):
        code[valid & (winner == i)] = CLASS_CODES[n]
    purity = np.full(winner.shape, np.nan, dtype="float32")
    purity[valid] = filled.max(axis=0)[valid]
    return code, purity


def derive_planting_mask(layers):
    """Bit-coded mask of land that is not available for planting.

    This is the non-circular use of land cover: it never enters the model as a
    predictor, it removes cells from the *output* of the model.  Bit 16 (already
    closed-canopy forest) is the one that matters most -- those are the cells a
    vegetation predictor would otherwise cause the model to recommend, which is
    exactly backwards for an afforestation advisory.
    """
    valid = np.isfinite(layers["tree"])
    mask = np.zeros(layers["tree"].shape, dtype="uint8")
    # Every layer is a percentage; a comparison against NaN is already False,
    # so ocean cells simply fail each test and are marked nodata at the end.
    tests = [
        ("ocean", layers["land"] < EXCLUDE_OCEAN_BELOW_LAND_PCT),
        ("inland_water", layers["water"] > EXCLUDE_WATER_ABOVE_PCT),
        ("permanent_snow_ice", layers["snowice"] > EXCLUDE_SNOWICE_ABOVE_PCT),
        ("built_up", layers["builtup"] > EXCLUDE_BUILTUP_ABOVE_PCT),
        ("closed_canopy_forest",
         layers["tree"] > EXCLUDE_CLOSED_FOREST_ABOVE_PCT),
        ("cropland", layers["cropland"] > FLAG_CROPLAND_ABOVE_PCT),
    ]
    for name, test in tests:
        mask[valid & test] |= MASK_BITS[name]
    mask[~valid] = 255
    return mask


# ─────────────────────────────────────────────────────
# SOURCE 2: PKU GIMMS NDVI -> four temporal statistics
# ─────────────────────────────────────────────────────
class RemoteZip:
    """Index a remote zip once, then pull single members by byte range.

    The Zenodo archive is 775 MB and holds 288 GeoTIFFs; we only want the 48 to
    192 that fall in our epoch, so reading the central directory and fetching
    members individually saves most of the transfer.
    """

    def __init__(self, url):
        self.url = url
        head = requests.head(url, allow_redirects=True, timeout=60)
        self.total = int(head.headers["content-length"])
        tail = self._range(max(0, self.total - 200000), self.total - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd < 0:
            raise RuntimeError(f"no zip end-of-central-directory in {url}")
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
        cd = self._range(cd_off, cd_off + cd_size - 1)

        self.entries = {}
        pos = 0
        while pos < len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
            method, = struct.unpack("<H", cd[pos + 10:pos + 12])
            csize, = struct.unpack("<I", cd[pos + 20:pos + 24])
            nlen, elen, clen = struct.unpack("<HHH", cd[pos + 28:pos + 34])
            local, = struct.unpack("<I", cd[pos + 42:pos + 46])
            name = cd[pos + 46:pos + 46 + nlen].decode()
            self.entries[name] = (local, csize, method)
            pos += 46 + nlen + elen + clen

    def _range(self, start, stop):
        r = requests.get(self.url, timeout=600,
                         headers={"Range": f"bytes={start}-{stop}"})
        r.raise_for_status()
        return r.content

    def member(self, name):
        local, csize, method = self.entries[name]
        header = self._range(local, local + 29)
        nlen, elen = struct.unpack("<HH", header[26:30])
        start = local + 30 + nlen + elen
        blob = self._range(start, start + csize - 1)
        return zlib.decompress(blob, -15) if method == 8 else blob


def ndvi_member_names(zip_index, years):
    """Semi-monthly member names for the requested years, in temporal order."""
    wanted = []
    for name in zip_index.entries:
        if not name.endswith(".tif"):
            continue
        stamp = os.path.basename(name).rsplit("_", 1)[-1][:-4]   # YYYYMMHH
        year, month, half = int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8])
        if year in years:
            wanted.append((year, month, half, name))
    wanted.sort()
    return [(y, (m - 1) * 2 + (h - 1), n) for y, m, h, n in wanted]


def fetch_ndvi_members(zip_index, members, raw_dir):
    """Cache each wanted member on disk so reruns do not re-download."""
    def grab(item):
        _, _, name = item
        path = os.path.join(raw_dir, os.path.basename(name))
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            blob = zip_index.member(name)
            with open(path, "wb") as f:
                f.write(blob)
        return path

    paths = []
    with futures.ThreadPoolExecutor(WORKER_THREADS) as pool:
        for path in tqdm(pool.map(grab, members), total=len(members),
                         desc="PKU GIMMS NDVI", unit="file"):
            paths.append(path)
    return paths


def aggregate_ndvi(zip_index, years, raw_dir, grid):
    """Build a semi-monthly NDVI climatology, then summarise it four ways.

    One annual mean is not enough: it cannot tell an evergreen rainforest from a
    monsoon grassland that is bare for eight months, and it is dragged down by
    the dormant season in exactly the seasonal biomes where trees are most
    constrained.  So we keep four statistics:

      observed_mean        average greenness over the composites that exist
      peak                 highest greenness of the year, the productive ceiling
      growing_season_mean  average over the greenest half of the year, which is
                           when growth actually happens
      seasonal_amplitude   peak minus trough, which separates evergreen from
                           deciduous and monsoon systems

    The statistics are computed at the native 1/12 degree resolution and only
    then averaged onto the 18.5 km grid.  That order matters: this project
    already learned with slope that a non-linear summary taken after spatial
    averaging is understated.  Averaging first would clip the peak and flatten
    the amplitude wherever a cell mixes biomes.
    """
    members = ndvi_member_names(zip_index, years)
    if not members:
        raise RuntimeError("no NDVI members matched the requested years")
    paths = fetch_ndvi_members(zip_index, members, raw_dir)

    total = np.zeros((24, NDVI_HEIGHT, NDVI_WIDTH), dtype="float32")
    count = np.zeros((24, NDVI_HEIGHT, NDVI_WIDTH), dtype="uint8")
    for (_, slot, _), path in tqdm(list(zip(members, paths)),
                                   desc="NDVI climatology", unit="file"):
        with rasterio.open(path) as src:
            raw = src.read(1)
        good = raw != NDVI_FILL
        total[slot] += np.where(good, raw, 0).astype("float32") * NDVI_SCALE
        count[slot] += good

    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    del total, count

    observed = np.isfinite(clim).sum(axis=0)
    usable = observed >= NDVI_MIN_SEMIMONTHS

    with np.errstate(invalid="ignore"):
        mean_native = np.nanmean(clim, axis=0)
        max_native = np.nanmax(clim, axis=0)
        min_native = np.nanmin(clim, axis=0)
        ranked = np.sort(np.where(np.isfinite(clim), clim, -np.inf), axis=0)
        green = ranked[-NDVI_GROWING_SEMIMONTHS:]
        growing_native = np.where(np.isfinite(green), green, np.nan)
        growing_native = np.nanmean(growing_native, axis=0)
    amplitude_native = max_native - min_native

    factor = NDVI_HEIGHT // grid["height"]         # 2
    layers = {}
    for name, native in [("observed_mean", mean_native),
                         ("peak", max_native),
                         ("growing_season_mean", growing_native),
                         ("seasonal_amplitude", amplitude_native)]:
        field = np.where(usable, native, np.nan).astype("float32")
        good = np.isfinite(field)
        num = block_mean(np.where(good, field, 0.0), factor)
        den = block_mean(good.astype("float32"), factor)
        with np.errstate(invalid="ignore", divide="ignore"):
            layers[name] = np.where(den > 0, num / den, np.nan).astype("float32")
    layers["_observed_semimonths"] = block_mean(
        observed.astype("float32"), factor).astype("float32")
    return layers, len(members)


# ─────────────────────────────────────────────────────
# SOURCE 3: ESA CCI surface soil moisture (the non-vegetation variable)
# ─────────────────────────────────────────────────────
def soil_moisture_urls(years, days):
    urls = []
    for year in years:
        for month in range(1, 13):
            for day in days:
                stamp = f"{year}{month:02d}{day:02d}"
                urls.append((month, f"{SM_BASE}/{year}/"
                             f"ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-"
                             f"{stamp}000000-fv{SM_VERSION[1:]}.nc"))
    return urls


def aggregate_soil_moisture(years, days, raw_dir, grid):
    """Monthly climatology of surface soil moisture, then mean and amplitude.

    Retrievals are missing wherever the soil is frozen, snow covered or under a
    dense canopy, so a per-month valid count is kept and reported: this layer
    has real holes, and they are not randomly placed.
    """
    jobs = soil_moisture_urls(years, days)

    def grab(job):
        month, url = job
        path = os.path.join(raw_dir, os.path.basename(url))
        if not download_file(url, path, quiet=True):
            return month, None
        return month, path

    total = np.zeros((12, SM_HEIGHT, SM_WIDTH), dtype="float64")
    count = np.zeros((12, SM_HEIGHT, SM_WIDTH), dtype="uint16")
    missing = 0
    with futures.ThreadPoolExecutor(WORKER_THREADS) as pool:
        for month, path in tqdm(pool.map(grab, jobs), total=len(jobs),
                                desc="ESA CCI soil moisture", unit="day"):
            if path is None:
                missing += 1
                continue
            with rasterio.open(f'netCDF:"{path}":sm') as src:
                band = src.read(1)
            good = np.isfinite(band) & (band > -1.0)
            total[month - 1] += np.where(good, band, 0.0)
            count[month - 1] += good

    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    with np.errstate(invalid="ignore"):
        mean_native = np.nanmean(clim, axis=0)
        span_native = np.nanmax(clim, axis=0) - np.nanmin(clim, axis=0)
    months_seen = np.isfinite(clim).sum(axis=0).astype("float32")

    src_transform = Affine(360.0 / SM_WIDTH, 0.0, -180.0,
                           0.0, -180.0 / SM_HEIGHT, 90.0)
    layers = {}
    for name, native in [("mean", mean_native), ("amplitude", span_native),
                         ("_months_observed", months_seen)]:
        # The source is coarser than the target (28 km vs 18.5 km), so this step
        # replicates values rather than refining them.  Nearest keeps the
        # numbers honest instead of inventing a gradient.
        out = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
        reproject(source=np.asarray(native, dtype="float32"), destination=out,
                  src_transform=src_transform, src_crs="EPSG:4326",
                  src_nodata=np.nan, dst_transform=grid["transform"],
                  dst_crs=grid["crs"], dst_nodata=np.nan,
                  resampling=Resampling.nearest, num_threads=4)
        layers[name] = out
    return layers, len(jobs) - missing, missing


# ─────────────────────────────────────────────────────
# CROSS-CHECKS: 10 m and 30 m products on sampled tiles
# ─────────────────────────────────────────────────────
def cross_check_worldcover(tiles, layers, grid, overview=WC_OVERVIEW):
    """Compare our 300 m fractions against ESA WorldCover 10 m on a few tiles.

    WorldCover is the newest and finest global land cover, but its internal
    overviews are mode-resampled, so fractions read from them drift toward the
    locally dominant class.  Reading at 40 m keeps that drift small; the drift
    itself is also measured, by re-reading at 320 m.
    """
    pixel = abs(grid["transform"].a)
    results = []
    for name in tqdm(tiles, desc="WorldCover 10 m check", unit="tile"):
        url = (f"/vsicurl/{WC_BASE}/"
               f"ESA_WorldCover_10m_2021_v200_{name}_Map.tif")
        try:
            with rasterio.open(url) as src:
                fine = src.read(1, out_shape=(src.height // overview,
                                              src.width // overview))
                coarse = src.read(1, out_shape=(src.height // 32,
                                                src.width // 32))
                left, top = src.bounds.left, src.bounds.top
        except rasterio.errors.RasterioIOError as exc:
            print(f"  skipped {name}: {str(exc)[:80]}")
            continue

        # A 3 degree tile holds exactly 18 target cells, and the 40 m overview
        # holds exactly 500 pixels per cell, so the two grids nest cleanly.
        cells = int(round(3.0 / pixel))
        row0 = int(round((90.0 - top) / pixel))
        col0 = int(round((left + 180.0) / pixel))
        ours_tile = layers["tree"][row0:row0 + cells, col0:col0 + cells]
        if not np.isfinite(ours_tile).any():
            continue

        for layer in ("tree", "builtup", "cropland", "water"):
            wanted = [c for c, l in WC_CLASS_TO_LAYER.items() if l == layer]
            hit = fractions_in_blocks(np.isin(fine, wanted), cells)
            hit32 = fractions_in_blocks(np.isin(coarse, wanted), cells)
            ours = layers[layer][row0:row0 + cells, col0:col0 + cells]
            good = np.isfinite(ours)
            results.append({
                "tile": name, "layer": layer,
                "worldcover_10m_pct": round(float(hit[good].mean()), 2),
                "cci_300m_pct": round(float(ours[good].mean()), 2),
                "mean_abs_diff_pct": round(
                    float(np.abs(hit[good] - ours[good]).mean()), 2),
                "worldcover_mode_overview_bias_pct": round(
                    float((hit32[good] - hit[good]).mean()), 2),
            })
    return results


def fractions_in_blocks(hits, cells):
    """Percentage of True per cells x cells block, trimming any ragged edge."""
    step_r = hits.shape[0] // cells
    step_c = hits.shape[1] // cells
    trimmed = hits[:cells * step_r, :cells * step_c]
    return trimmed.reshape(cells, step_r, cells,
                           step_c).mean(axis=(1, 3)) * 100.0


def cross_check_hansen(tiles, layers, grid, raw_dir):
    """Compare against Hansen tree canopy cover, updated to 2024 with the loss layer.

    Hansen measures canopy cover in 2000; blanking the pixels flagged as lost
    gives a 2024 canopy layer.  The tiles are row-striped rather than tiled, so
    there is no cheap decimated read -- hence one or two tiles only.

    Hansen's 0.00025 degree pixel does not divide the 1/6 degree target cell, so
    the comparison is made on 1 degree blocks, which both grids do divide
    exactly (4000 Hansen pixels, 6 target cells).
    """
    pixel = abs(grid["transform"].a)
    per_degree = int(round(1.0 / pixel))             # 6 target cells
    results = []
    for name in tqdm(tiles, desc="Hansen 30 m check", unit="tile"):
        arrays, bounds = {}, None
        for layer in ("treecover2000", "lossyear"):
            fname = f"Hansen_{HANSEN_VERSION}_{layer}_{name}.tif"
            path = os.path.join(raw_dir, fname)
            if not download_file(f"{HANSEN_BASE}/{fname}", path, quiet=True):
                arrays = None
                break
            with rasterio.open(path) as src:
                arrays[layer] = src.read(
                    1, out_shape=(src.height // HANSEN_DECIMATE,
                                  src.width // HANSEN_DECIMATE))
                bounds = src.bounds
        if not arrays:
            continue

        cover = arrays["treecover2000"].astype("float32")
        cover[arrays["lossyear"] > 0] = 0.0          # cleared 2001-2024
        degrees = int(round(bounds.top - bounds.bottom))
        hansen = block_mean(cover[:cover.shape[0] // degrees * degrees,
                                  :cover.shape[1] // degrees * degrees],
                            cover.shape[0] // degrees)

        cells = degrees * per_degree
        row0 = int(round((90.0 - bounds.top) / pixel))
        col0 = int(round((bounds.left + 180.0) / pixel))
        ours_cells = layers["tree"][row0:row0 + cells, col0:col0 + cells]
        good_cells = np.isfinite(ours_cells)
        num = block_mean(np.where(good_cells, ours_cells, 0.0), per_degree)
        den = block_mean(good_cells.astype("float32"), per_degree)
        with np.errstate(invalid="ignore", divide="ignore"):
            ours = np.where(den > 0, num / den, np.nan)

        good = np.isfinite(ours)
        if good.sum() < 2:
            continue
        results.append({
            "tile": name,
            "comparison_resolution": "1 degree blocks",
            "blocks_compared": int(good.sum()),
            "hansen_2024_canopy_pct": round(float(hansen[good].mean()), 2),
            "cci_300m_tree_pct": round(float(ours[good].mean()), 2),
            "mean_abs_diff_pct": round(
                float(np.abs(hansen[good] - ours[good]).mean()), 2),
            "correlation": round(float(np.corrcoef(
                hansen[good], ours[good])[0, 1]), 3),
        })
    return results


def main():
    print("=" * 50)
    print("SATELLITE RASTER DOWNLOADER")
    print("=" * 50)

    full = MODE == "full"
    ndvi_years = NDVI_YEARS_FULL if full else NDVI_YEARS_SAMPLE
    sm_years = SM_YEARS_FULL if full else SM_YEARS_SAMPLE
    sm_days = SM_DAYS_FULL if full else SM_DAYS_SAMPLE
    wc_tiles = WC_TILES_FULL if full else WC_TILES_SAMPLE
    hansen_tiles = HANSEN_TILES_FULL if full else HANSEN_TILES_SAMPLE

    print(f"Mode:        {MODE}")
    print(f"Land cover:  ESA CCI LC PFT {PFT_VERSION}, {PFT_YEAR}, 300 m "
          f"(global, fractions)")
    print(f"NDVI:        PKU GIMMS {NDVI_VERSION}, 1/12 deg, semi-monthly, "
          f"{ndvi_years[0]}-{ndvi_years[-1]}")
    print(f"Soil moist.: ESA CCI SM {SM_VERSION}, 0.25 deg, "
          f"{sm_years[0]}-{sm_years[-1]}, day(s) {sm_days} of each month")
    print(f"Cross-check: WorldCover 10 m x{len(wc_tiles)} tiles, "
          f"Hansen 30 m x{len(hansen_tiles)} tiles")

    raw_dir = os.path.join(OUTPUT_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    grid = read_template(TEMPLATE)
    print(f"\nTarget grid: {grid['width']}x{grid['height']} {grid['crs']} "
          f"pixel {abs(grid['transform'].a):.10f} deg")

    # ── land cover ──
    print("\nDownloading ESA CCI LC plant functional types (global, 1.8 GB)...")
    pft_path = os.path.join(raw_dir, PFT_FILE)
    if not download_file(f"{PFT_BASE}/{PFT_FILE}", pft_path,
                         threads=DOWNLOAD_THREADS):
        raise RuntimeError("could not retrieve the PFT land-cover file")
    print("Aggregating 300 m fractions onto the target grid...")
    layers = aggregate_landcover(
        pft_path, grid,
        cache=os.path.join(raw_dir, f"landcover_aggregated_{PFT_YEAR}.npz"))

    # ── NDVI ──
    print("\nIndexing the PKU GIMMS NDVI archive (range requests only)...")
    zip_index = RemoteZip(NDVI_ZIP)
    ndvi, n_composites = aggregate_ndvi(zip_index, ndvi_years, raw_dir, grid)

    # ── soil moisture ──
    print("\nDownloading ESA CCI surface soil moisture...")
    soil_moisture, n_days, n_missing = aggregate_soil_moisture(
        sm_years, sm_days, raw_dir, grid)
    if n_missing:
        print(f"  {n_missing} of {n_days + n_missing} daily files unavailable")

    # ── derived products ──
    print("\nDeriving majority class, purity and the planting-exclusion mask...")
    majority, purity = derive_majority(layers)
    exclusion = derive_planting_mask(layers)

    # ── write ──
    print("Writing aligned layers...")
    written = []
    provenance = ("ESA CCI Land Cover PFT " + PFT_VERSION +
                  f", epoch {PFT_YEAR}, 300 m")
    for name in list(PFT_COMPONENTS) + ["land"]:
        path = os.path.join(OUTPUT_DIR, f"landcover_{name}_frac_10m.tif")
        write_aligned(layers[name], path, grid, {
            "satellite_layer": f"{name} cover fraction",
            "satellite_units": ("percent of the cell that is not sea"
                                if name == "land" else
                                "percent of the non-sea part of the cell"),
            "satellite_source": provenance,
            "satellite_epoch": str(PFT_YEAR),
            "satellite_model_role": (
                "FILTER ONLY - vegetation state, do not use as a predictor"
                if name in ("tree", "tree_broadleaf", "tree_needleleaf",
                            "shrub", "grass_natural")
                else "land availability constraint, safe as a filter"),
        })
        written.append(os.path.basename(path))

    for name, array in ndvi.items():
        if name.startswith("_"):
            continue
        path = os.path.join(OUTPUT_DIR, f"ndvi_{name}_10m.tif")
        write_aligned(array, path, grid, {
            "satellite_layer": f"NDVI {name.replace('_', ' ')}",
            "satellite_units": "NDVI, dimensionless -1 to 1",
            "satellite_source": f"PKU GIMMS NDVI {NDVI_VERSION}, 1/12 deg",
            "satellite_epoch": f"{ndvi_years[0]}-{ndvi_years[-1]}",
            "satellite_model_role": (
                "DIAGNOSTIC ONLY - measures current greenness, circular as a "
                "predictor of where trees can grow"),
        })
        written.append(os.path.basename(path))

    path = os.path.join(OUTPUT_DIR, "ndvi_observed_semimonths_10m.tif")
    write_aligned(ndvi["_observed_semimonths"], path, grid, {
        "satellite_layer": "semi-monthly composites with data, of 24",
        "satellite_source": f"PKU GIMMS NDVI {NDVI_VERSION}",
    })
    written.append(os.path.basename(path))

    for name in ("mean", "amplitude"):
        path = os.path.join(OUTPUT_DIR, f"soilmoisture_{name}_10m.tif")
        write_aligned(soil_moisture[name], path, grid, {
            "satellite_layer": f"surface soil moisture {name}",
            "satellite_units": "m3/m3 (top ~5 cm)",
            "satellite_source": f"ESA CCI Soil Moisture {SM_VERSION} COMBINED",
            "satellite_epoch": f"{sm_years[0]}-{sm_years[-1]}",
            "satellite_native_resolution": "0.25 deg (~28 km), coarser than grid",
            "satellite_model_role": (
                "CANDIDATE PREDICTOR - not a vegetation index, but retrieval "
                "gaps track canopy density, so check coverage first"),
        })
        written.append(os.path.basename(path))

    path = os.path.join(OUTPUT_DIR, "soilmoisture_months_observed_10m.tif")
    write_aligned(soil_moisture["_months_observed"], path, grid, {
        "satellite_layer": "months of the year with any retrieval, of 12",
        "satellite_source": f"ESA CCI Soil Moisture {SM_VERSION}",
    })
    written.append(os.path.basename(path))

    path = os.path.join(OUTPUT_DIR, "landcover_majority_class_10m.tif")
    write_aligned(majority, path, grid, {
        "satellite_layer": "majority land-cover class code",
        "satellite_classes": json.dumps(CLASS_CODES),
        "satellite_source": provenance,
    }, dtype="uint8", nodata=0)
    written.append(os.path.basename(path))

    path = os.path.join(OUTPUT_DIR, "landcover_purity_10m.tif")
    write_aligned(purity, path, grid, {
        "satellite_layer": "percent of the cell held by its majority class",
        "satellite_source": provenance,
    })
    written.append(os.path.basename(path))

    path = os.path.join(OUTPUT_DIR, "planting_exclusion_mask_10m.tif")
    write_aligned(exclusion, path, grid, {
        "satellite_layer": "bit-coded planting exclusion mask",
        "satellite_bits": json.dumps(MASK_BITS),
        "satellite_source": provenance,
        "satellite_model_role": (
            "POST-HOC FILTER - apply to model output, never as a predictor"),
    }, dtype="uint8", nodata=255)
    written.append(os.path.basename(path))

    # ── cross-checks ──
    print("\nCross-checking against 10 m and 30 m products on sampled tiles...")
    wc_results = cross_check_worldcover(wc_tiles, layers, grid)
    hansen_results = cross_check_hansen(hansen_tiles, layers, grid, raw_dir)

    # Count only what actually crossed the network: the aggregation cache is
    # derived locally and would otherwise inflate the download figure.
    downloaded = sum(os.path.getsize(os.path.join(raw_dir, f))
                     for f in os.listdir(raw_dir) if not f.endswith(".npz"))
    manifest = {
        "target_grid": {"width": grid["width"], "height": grid["height"],
                        "crs": str(grid["crs"]),
                        "transform": list(grid["transform"])[:6],
                        "template": TEMPLATE},
        "mode": MODE,
        "bytes_downloaded": downloaded,
        "layers": sorted(written),
        "class_codes": CLASS_CODES,
        "exclusion_mask_bits": MASK_BITS,
        "sources": {
            "ESA CCI Land Cover PFT": {
                "version": PFT_VERSION,
                "published": "annual updates; v2.0.81 runs to 2020, the last "
                             "year published as of September 2026",
                "catalogue": ("https://catalogue.ceda.ac.uk/uuid/"
                              "313854aedcb04a5eb56f711401a87396"),
                "epoch": str(PFT_YEAR),
                "native_crs": "EPSG:4326",
                "native_resolution": "300 m",
                "licence": "ESA CCI Land Cover terms: free use with attribution",
                "url": f"{PFT_BASE}/{PFT_FILE}",
                "variables": sorted({v for g in PFT_COMPONENTS.values()
                                     for v in g}) + ["LAND"],
                "role": ("global land-cover class fractions; the only "
                         "credential-free global product that publishes "
                         "fractions rather than class codes"),
                "aggregation": ("mean of the 60x60 native pixels per target "
                                "cell that are not sea, identified by "
                                "WATER_OCEAN == 0; the LAND variable is not "
                                "used as the weight because it excludes inland "
                                "water and so erases lakes"),
            },
            "PKU GIMMS NDVI": {
                "version": NDVI_VERSION,
                "published": "2023-08-17 (Zenodo 8253971)",
                "doi": "10.5281/zenodo.8253971",
                "record_coverage": "1982-2022, semi-monthly",
                "method": ("biome-specific neural-network correction of GIMMS "
                           "NDVI3g against 3.6 million Landsat samples, then "
                           "fused with MODIS MOD13C1 to reach 2022"),
                "epoch": f"{ndvi_years[0]}-{ndvi_years[-1]}",
                "native_crs": "EPSG:4326",
                "native_resolution": "1/12 deg (~9.3 km)",
                "temporal_resolution": "semi-monthly (24 per year)",
                "composites_used": n_composites,
                "licence": "CC-BY 4.0",
                "url": NDVI_ZIP,
                "role": ("vegetation index; diagnostic and gap-finding only, "
                         "excluded from the predictor set as circular"),
                "aggregation": ("semi-monthly climatology and its four summary "
                                "statistics computed at 1/12 deg, then averaged "
                                "2x2 onto the target grid"),
                "statistics": ["observed_mean", "peak", "growing_season_mean",
                               "seasonal_amplitude"],
                "validity_rule": (
                    f"a cell needs >= {NDVI_MIN_SEMIMONTHS} of 24 semi-monthly "
                    "composites; the source masks snow, ice, water and barren "
                    "desert, which removes most of the winter above 60 degrees, "
                    "so the mean is over observed composites and is named "
                    "observed_mean rather than annual_mean"),
            },
            "ESA CCI Soil Moisture": {
                "version": SM_VERSION,
                "published": "v09.2, daily files available 1978-2024",
                "catalogue": ("https://catalogue.ceda.ac.uk/uuid/"
                              "d4e66299f5054129b8076fb7502949e1"),
                "epoch": f"{sm_years[0]}-{sm_years[-1]}",
                "native_crs": "EPSG:4326",
                "native_resolution": "0.25 deg (~28 km)",
                "temporal_sample": f"day(s) {sm_days} of each month",
                "daily_files_used": n_days,
                "licence": "ESA CCI data policy: free use with attribution",
                "url": SM_BASE,
                "role": ("the one satellite variable here that is not a "
                         "vegetation proxy, so the only predictor candidate"),
                "aggregation": ("monthly climatology at 0.25 deg, then nearest "
                                "onto the finer target grid without inventing "
                                "detail"),
            },
            "ESA WorldCover": {
                "version": WC_VERSION,
                "published": "2022-10-28 (v200)",
                "doi": "10.5281/zenodo.7254221",
                "epoch": "2021",
                "native_resolution": "10 m",
                "licence": "CC-BY 4.0",
                "url": WC_BASE,
                "role": "cross-check on sampled tiles only",
                "why_not_global": (
                    "2651 tiles; fractions must come from a read fine enough "
                    "to escape the mode-resampled overviews, which costs about "
                    "8 GB at 80 m and 26 GB at 40 m, against a 4 GB budget"),
                "tiles_checked": wc_tiles,
            },
            "Hansen Global Forest Change": {
                "version": HANSEN_VERSION,
                "published": "2026-03 (tiles dated 16-19 March 2026)",
                "epoch": "2000 canopy cover, annual loss through 2025",
                "native_resolution": "30 m (0.00025 deg)",
                "licence": "CC-BY 4.0",
                "url": HANSEN_BASE,
                "role": "cross-check on sampled tiles only",
                "why_not_global": (
                    "504 tiles, row-striped rather than tiled, so a decimated "
                    "read still transfers the whole tile; measured by HEAD "
                    "request, canopy cover and loss come to 68.1 GB globally "
                    "(58.9 + 9.2), against a 4 GB budget"),
                "tiles_checked": hansen_tiles,
            },
        },
        "rejected_sources": {
            "Google Earth Engine": (
                "needs a registered account and the earthengine-api package; "
                "pip is blocked by PEP 668 on this machine, so the stack the "
                "design document recommends is unusable here"),
            "MODIS MCD12Q1 / MCD12C1, MOD13 / MYD13, VIIRS VNP13": (
                "NASA Earthdata login required; anonymous access to "
                "e4ftl01.cr.usgs.gov now returns 404"),
            "C3S Land Cover 2016-2022 (the newer ESA CCI LC years)":
                "Copernicus Climate Data Store API key required",
            "Copernicus Global Land Cover 100 m (CGLS-LC100 v3.0.1, 2020-09-08)":
                "public on Zenodo and does publish cover fractions, which would "
                "have been ideal, but the GeoTIFFs carry no overviews, so a "
                "global read costs 5.6 GB for the tree layer alone",
            "GLC_FCS30D (30 m, annual 1985-2022, Zenodo 8239305)":
                "public and CC-BY 4.0 but 194 GB of per-tile archives",
            "ESRI Land Cover v003 (10 m, annual to 2023)":
                "public on Azure blob storage but ~700 UTM tiles of ~150 MB "
                "each, and would need reprojection per tile",
            "Google Dynamic World": "Earth Engine only",
            "SMAP surface soil moisture, MODIS snow cover (MOD10CM)":
                "both need a NASA Earthdata login; ESA CCI Soil Moisture is the "
                "credential-free equivalent and is used instead",
        },
        "cross_checks": {"worldcover_10m": wc_results,
                         "hansen_30m": hansen_results},
        "circularity_note": (
            "Tree, shrub, grass and every NDVI layer describe current "
            "vegetation state and must not enter the predictor set: they encode "
            "the answer to 'can trees grow here'. Use them through "
            "planting_exclusion_mask_10m.tif and as the 'current green gaps' "
            "diagnostic. Built-up, water, permanent snow/ice and cropland "
            "describe land availability and are safe as filters. Surface soil "
            "moisture is the only candidate predictor. See "
            "validate_satellite_alignment.py for the measurement."),
    }
    with open(os.path.join(OUTPUT_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Wrote {len(written)} layers to {OUTPUT_DIR}/")
    for name in sorted(written):
        print(f"  {name}")
    print(f"\nDownloaded {downloaded / 1e9:.2f} GB into {raw_dir}/")
    print(f"Manifest: {OUTPUT_DIR}/MANIFEST.json")
    print("Next: run validate_satellite_alignment.py")


if __name__ == "__main__":
    main()
