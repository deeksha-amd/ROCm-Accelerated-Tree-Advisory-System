"""
Build the country-scoped CLIMATE block at 30 arcsec (~1 km): 19 bioclimatic
variables for 2015-2024, plus the water-demand variables the project has been
missing.

WHY 30 ARCSEC AND NOT 10 ARCMIN
    The global blocks are 10 arcmin (~18.5 km) because the model was global.
    DATA_OVERVIEW.md already concedes assumption 4: "One cell can hold a
    mountain and a valley; the model sees the average." For one country, and
    especially for a deep SDM that is supposed to read spatial structure, that
    assumption is the main thing standing in the way.

    WorldClim 2.1 publishes 30 arcsec grids PER ISO3 COUNTRY, which changes
    the economics completely: a few hundred MB of direct downloads instead of
    the 10.4 GB global bioclim zip. The files land on exactly the global 30
    arcsec lattice, and 1/6 deg is exactly 20 x 1/120 deg, so the fine grid
    nests 20 x 20 inside the existing blocks and the two remain comparable.

THE EPOCH PROBLEM, AND HOW IT IS SOLVED
    WorldClim's per-country 30 arcsec files are 1970-2000 normals. The
    occurrences have a median year of 2024, and the project measured +1.28 degC
    between those epochs, so training on 1970-2000 is the error climate_recent/
    was built to avoid. But the historical monthly weather product that
    climate_recent/ derives from is published at 10 ARCMIN ONLY -- there is no
    30 arcsec version to download. Checked, not assumed: the 30s path under
    hist/cts4.09 returns 404.

    So this script uses the delta (change-factor) method, which is the standard
    way to move a fine-resolution climatology to another epoch and is the same
    logic WorldClim itself uses to downscale CMIP6:

        1. From the monthly archives already in climate_monthly_src/, compute
           the 2015-2024 and 1970-2000 monthly climatologies at 10 arcmin.
        2. Take the anomaly between them -- additive for temperature,
           multiplicative for precipitation.
        3. Interpolate the ANOMALY to 30 arcsec and apply it to WorldClim's
           30 arcsec 1970-2000 normals.
        4. Derive the 19 bioclim variables from the result.

    What makes this legitimate is the division of labour: the fine spatial
    structure comes from the 30 arcsec normals, which genuinely resolve it,
    while the anomaly field supplies only the epoch shift. Interpolating the
    anomaly is defensible because a 55-year climate change signal is smooth
    over tens of km in a way that the climate itself is not -- it has no
    coastline and no mountain range in it. Interpolating the climatology
    directly would be the mistake.

    Both source epochs are taken from the SAME cts4.09 product, so any
    systematic bias in it cancels in the difference. The 1970-2000 baseline
    derived this way is also compared against WorldClim's published 1970-2000
    normals and the agreement is reported, which is what turns that claim into
    a measurement.

    The bioclim code is imported from recent_climate_rasters.py rather than
    reimplemented. That module's conventions (sample sd for BIO4, dismo's
    CV-of-precipitation+1 for BIO15) were pinned down by reproducing
    WorldClim's own published rasters bit-for-bit, and a second copy of that
    logic would be a second chance to get it wrong.

CLOSING THE WATER-DEMAND GAP
    The existing blocks carry temperature and precipitation but no solar
    radiation, wind or vapour pressure, so the model sees water SUPPLY and not
    water DEMAND -- it cannot tell a cool wet 800 mm climate from a hot windy
    one. Two potential-evapotranspiration estimates are produced:

    PET_HARGREAVES is the primary. Hargreaves-Samani needs only tmin, tmax and
        extraterrestrial radiation, and Ra is pure orbital geometry computed
        analytically from latitude and day of year. So it is entirely
        consistent with the 2015-2024 window and needs no extra download.

    PET_PENMAN is FAO-56 Penman-Monteith, which is the better formulation
        where the inputs exist. Here it has to mix epochs: solar radiation,
        wind and vapour pressure are published as 1970-2000 normals ONLY (this
        was checked -- srad, wind and vapr are absent from the historical
        monthly product), so those three come from 1970-2000 while the
        temperatures are 2015-2024. That is acceptable for radiation, which is
        orbital and cloud-climatological, and reasonable for wind. It is
        weakest for vapour pressure, which really does rise with temperature,
        and the effect is to understate vapour pressure deficit and so
        understate PET. The two estimates are compared in the output and the
        ratio is reported; treat a large disagreement as a reason to prefer
        Hargreaves.

    vapr is also the one input with no per-country 30 arcsec file at all, so it
    is downscaled from the 10 arcmin normals. It is therefore the only layer
    in this block whose real information content is coarser than its grid, and
    it is tagged as such.

    Derived from those: aridity index (P / PET, the UNEP definition), annual
    climatic water deficit, and vapour pressure deficit.

OUTPUTS  (country_data/<ISO>/climate_30s/)
    wc2.1_30s_bio_1.tif ... _19.tif    2015-2024, delta-downscaled
    baseline_1970_2000/                the untouched WorldClim normals
    srad_annual_mean.tif, wind_annual_mean.tif, vapr_annual_mean.tif
    pet_hargreaves_annual.tif, pet_penman_annual.tif
    aridity_index.tif, water_deficit_annual.tif, vpd_annual_mean.tif
    elevation_worldclim.tif            WorldClim's own elev, a grid control
    MANIFEST.json

    The grid these land on is defined by the WorldClim per-country extent,
    which is the country's bounding box snapped outward to 0.5 deg. 0.5 deg is
    3 x 1/6 deg and 60 x 1/120 deg, so that extent is simultaneously on the
    10 arcmin and 30 arcsec global lattices. Every other country block is built
    on the grid this script writes to country_grid.json, which is what keeps
    the blocks mutually aligned.

USAGE
    python3 country_climate_rasters.py IND
    python3 country_climate_rasters.py IND --keep-baseline
    python3 country_climate_rasters.py IND --stage download   # fetch only
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm

import recent_climate_rasters as rcr
from country_clip import (RES_10M, RES_30S, country_grid, country_mask,
                          download_file, snap_bounds)

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUT_ROOT = "country_data"
CACHE_ROOT = os.path.join("country_src_cache", "worldclim")
MONTHLY_SRC = "climate_monthly_src"     # read-only: the existing 10 arcmin cache

# The grid. 0.5 deg is the lattice WorldClim snaps its per-country files to,
# and it is a whole multiple of both project resolutions.
WC_SNAP_DEG = 0.5
TARGET_RES = RES_30S

# Epochs. RECENT is the training window; BASELINE is what the 30 arcsec
# normals represent and what the anomaly is measured against.
RECENT = (2015, 2024)
BASELINE = (1970, 2000)

# ── WorldClim per-country 30 arcsec ──
WC_ISO_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso"

# What is actually needed. The published `bio` file is deliberately NOT in this
# list: we derive all 19 variables ourselves from the monthly grids, and it is
# by far the largest file on offer (380 MB for India, 2.4 GB for the USA). It
# is worth fetching only to cross-check the derivation, behind --crosscheck-bio.
WC_ISO_VARS = ("tmin", "tmax", "prec", "srad", "wind", "elev")
WC_ISO_REQUIRED = ("tmin", "tmax", "prec")
WC_BIO_BANDS = 19

# ── vapr: 10 arcmin only, there is no per-country 30 arcsec file ──
WC_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
VAPR_10M_ZIP = "wc2.1_10m_vapr.zip"

# Unit conversions out of WorldClim's storage units.
SRAD_KJ_TO_MJ = 1e-3            # kJ m-2 day-1 -> MJ m-2 day-1
WIND_10M_TO_2M = 0.748          # FAO-56 log profile, 10 m -> 2 m

# Precipitation anomaly is a ratio, which explodes where the baseline is
# nearly zero. Below this monthly total the anomaly is applied additively
# instead, and the ratio is clamped regardless.
PREC_RATIO_FLOOR_MM = 1.0
PREC_RATIO_CLAMP = (0.2, 5.0)

NODATA = rcr.NODATA             # -3.4e38, the project convention
DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MID_MONTH_DOY = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])

# FAO-56 constants.
SOLAR_CONSTANT = 0.0820         # MJ m-2 min-1
ALBEDO = 0.23
STEFAN_BOLTZMANN = 4.903e-9     # MJ K-4 m-2 day-1
LATENT_HEAT = 2.45              # MJ kg-1, so MJ m-2 day-1 / 2.45 = mm day-1

SEP = "=" * 70


def nanmean(array, axis=None):
    """np.nanmean without the all-NaN warning.

    Over a country window most cells are sea, so whole blocks and whole
    columns are legitimately empty; the warning is noise, not information.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(array, axis=axis)


# ─────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────
def wc_iso_url(iso, var):
    return f"{WC_ISO_BASE}/{iso}_wc2.1_30s_{var}.tif"


def fetch_worldclim_country(iso, cache_dir, variables=WC_ISO_VARS):
    """Download the per-country 30 arcsec files, returning {var: path}.

    A few country/variable combinations are broken upstream -- some are served
    as zero-byte files and some 404 -- so each one is validated after download
    and a missing variable is reported rather than silently producing an empty
    layer. tmin, tmax and prec are the only genuinely required ones: bio is a
    cross-check (we derive our own), and srad and wind only feed the secondary
    PET.
    """
    os.makedirs(cache_dir, exist_ok=True)
    paths, missing = {}, []

    for var in variables:
        path = os.path.join(cache_dir, f"{iso}_wc2.1_30s_{var}.tif")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"  [have] {os.path.basename(path)}  "
                  f"({os.path.getsize(path) / 1e6:.1f} MB)")
            paths[var] = path
            continue

        url = wc_iso_url(iso, var)
        if not download_file(url, path, desc=f"{iso} {var}"):
            missing.append(var)
            continue
        if os.path.getsize(path) < 1000:
            print(f"  WARNING: {url} served {os.path.getsize(path)} bytes; "
                  f"WorldClim's {var} file for {iso} is broken upstream")
            os.remove(path)
            missing.append(var)
            continue
        paths[var] = path

    return paths, missing


def fetch_vapr_10m(cache_dir):
    """The 10 arcmin vapour pressure normals; there is no 30 arcsec country file."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, VAPR_10M_ZIP)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"  [have] {VAPR_10M_ZIP}  "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")
        return path
    if not download_file(f"{WC_BASE}/{VAPR_10M_ZIP}", path, desc="vapr 10m"):
        return None
    return path


# ─────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────
def grid_from_worldclim(path, iso, margin_deg, area_frac, max_gap_deg,
                        bbox=None):
    """Adopt the downloaded file's extent as the authoritative country grid.

    Using WorldClim's own extent rather than our own snapped box guarantees
    every target cell is backed by real data: a wider box would only add a
    nodata margin. The extent is asserted to be on both project lattices, so
    the 20 x 20 nesting relationship with the existing 10 arcmin blocks holds.

    `bbox` restricts that extent to (west, south, east, north), which is the
    escape hatch for a country whose per-country file is mostly somewhere else.
    WorldClim files a country's whole sovereign span, so USA reaches from Guam
    across the antimeridian to Maine (43140 x 6540, 282 M cells, overwhelmingly
    ocean) and ESP reaches down to the Canaries. A bbox turns the adoption into
    "WorldClim's lattice, our extent": the box is snapped OUTWARD to the
    10 arcmin lattice and then intersected with what WorldClim actually has, so
    the result is still a whole number of coarse cells and still entirely
    backed by real data. The resulting read window is carried on the grid as
    `wc_window`, and every read of a per-country file goes through it.
    """
    with rasterio.open(path) as src:
        transform, width, height = src.transform, src.width, src.height
        res = src.transform.a

    if abs(res - TARGET_RES) > 1e-9:
        raise RuntimeError(f"{path} has pixel size {res}, expected {TARGET_RES}")

    west, north = transform.c, transform.f
    east, south = west + width * res, north - height * res
    full = (west, south, east, north)
    window = rasterio.windows.Window(0, 0, width, height)

    if bbox is not None:
        want = snap_bounds(bbox, margin_deg=0.0, snap_res=RES_10M)
        west = max(full[0], want[0])
        south = max(full[1], want[1])
        east = min(full[2], want[2])
        north = min(full[3], want[3])
        if east <= west or north <= south:
            raise RuntimeError(
                f"--bbox {bbox} does not intersect WorldClim's extent for "
                f"{iso}, which is {full[0]:g},{full[1]:g},{full[2]:g},"
                f"{full[3]:g}")
        col_off = int(round((west - full[0]) / res))
        row_off = int(round((full[3] - north) / res))
        width = int(round((east - west) / res))
        height = int(round((north - south) / res))
        window = rasterio.windows.Window(col_off, row_off, width, height)

    for value, name in ((west, "west"), (east, "east"),
                        (south, "south"), (north, "north")):
        on_fine = abs(value / TARGET_RES - round(value / TARGET_RES)) < 1e-6
        on_coarse = abs(value / RES_10M - round(value / RES_10M)) < 1e-6
        if not (on_fine and on_coarse):
            raise RuntimeError(
                f"the {name} edge {value} is not on both the 30 arcsec "
                "and 10 arcmin lattices, so the blocks would not nest")

    # Reuse country_grid for the polygon and metadata, then override the extent.
    grid = country_grid(iso, res_deg=TARGET_RES, margin_deg=margin_deg,
                        area_frac=area_frac, max_gap_deg=max_gap_deg)
    source = os.path.basename(path)
    if bbox is not None:
        source += f" cropped to {west:g},{south:g},{east:g},{north:g}"
    grid.update(transform=Affine(res, 0.0, west, 0.0, -res, north),
                width=width, height=height, bounds=(west, south, east, north),
                res_deg=TARGET_RES, grid_source=source, wc_window=window,
                wc_full_bounds=full)
    return grid


def coarse_window(grid):
    """The matching window on the global 10 arcmin grid, as integer offsets."""
    west, south, east, north = grid["bounds"]
    col = (west + 180.0) / RES_10M
    row = (90.0 - north) / RES_10M
    w = (east - west) / RES_10M
    h = (north - south) / RES_10M
    for value in (col, row, w, h):
        if abs(value - round(value)) > 1e-6:
            raise RuntimeError("the country extent is not a whole number of "
                               "10 arcmin cells; the blocks would not nest")
    return rasterio.windows.Window(round(col), round(row), round(w), round(h))


def write_layer(path, array, grid, tags, dtype="float32", nodata=NODATA):
    """One single-band GeoTIFF on the country grid."""
    data = np.where(np.isfinite(array), array, nodata).astype(dtype)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=grid["width"],
                       height=grid["height"], count=1, dtype=dtype,
                       crs=grid["crs"], transform=grid["transform"],
                       nodata=nodata, compress="deflate", predictor=2,
                       tiled=True, blockxsize=256, blockysize=256,
                       BIGTIFF="IF_SAFER") as dst:
        dst.write(data, 1)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


# ─────────────────────────────────────────────────────
# READ THE 30 ARCSEC NORMALS
# ─────────────────────────────────────────────────────
def read_wc_country(path, grid, indexes=None):
    """Read a WorldClim per-country file on the country grid, NaN for nodata.

    Always read through grid["wc_window"], which is the whole file unless
    grid_from_worldclim() was given a bbox. Every per-country variable is filed
    on one extent, so one window serves all of them -- and the shape is checked
    against the grid afterwards rather than assumed, because a variable served
    on a different extent would otherwise be silently misregistered.
    """
    window = grid.get("wc_window")
    with rasterio.open(path) as src:
        full = (src.width, src.height)
        expected = ((window.width, window.height) if window is not None
                    else (grid["width"], grid["height"]))
        if window is not None and (window.col_off + window.width > src.width
                                   or window.row_off + window.height > src.height):
            raise RuntimeError(
                f"{path} is {full[0]}x{full[1]}, too small for the "
                f"{window.width}x{window.height} window at "
                f"({window.col_off},{window.row_off}); WorldClim's per-country "
                "files for one country should all share an extent")
        if expected != (grid["width"], grid["height"]):
            raise RuntimeError(
                f"{path} yields {expected[0]}x{expected[1]} but the grid is "
                f"{grid['width']}x{grid['height']}")
        arr = src.read(indexes=indexes, window=window).astype("float32")
        nodata = src.nodata
    arr[arr < -1e30] = np.nan
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    return arr


def read_wc_monthly(path, grid):
    """A 12-band WorldClim country file as (12, H, W) float32 with NaN nodata."""
    arr = read_wc_country(path, grid)
    if arr.shape[0] != 12:
        raise RuntimeError(f"{path} has {arr.shape[0]} bands, expected 12")
    return arr


# ─────────────────────────────────────────────────────
# THE 10 ARCMIN ANOMALY
# ─────────────────────────────────────────────────────
def window_climatology(var, year_from, year_to, window, verbose=True):
    """Mean monthly grids over a year window, read only inside `window`.

    Returns (12, h, w). Reading a window straight out of the zipped GeoTIFFs
    keeps this to a few MB per month instead of the full 2160 x 1080 global
    grid, which matters because the baseline epoch is 31 years x 12 months.
    """
    out = []
    for month in tqdm(range(1, 13), desc=f"{var} {year_from}-{year_to}",
                      disable=not verbose, leave=False):
        total = None
        count = None
        for year in range(year_from, year_to + 1):
            block = next(b for b in rcr.DECADE_BLOCKS if b[0] <= year <= b[1])
            zip_path = os.path.join(MONTHLY_SRC,
                                    rcr.hist_zip_name(var, block))
            member = (f"wc2.1_{rcr.HIST_TAG}_{rcr.RESOLUTION}_{var}_"
                      f"{year}-{month:02d}.tif")
            with rasterio.open(f"zip://{zip_path}!{member}") as src:
                arr = src.read(1, window=window).astype("float64")
                nodata = src.nodata
            arr[arr < -1e30] = np.nan
            if nodata is not None and np.isfinite(nodata):
                arr[arr == nodata] = np.nan

            ok = np.isfinite(arr)
            if total is None:
                total, count = np.where(ok, arr, 0.0), ok.astype(np.int16)
            else:
                total += np.where(ok, arr, 0.0)
                count += ok
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(count > 0, total / np.maximum(count, 1),
                                np.nan))
    return np.stack(out)


def upsample(coarse, coarse_transform, grid):
    """Interpolate a 10 arcmin field onto the 30 arcsec grid.

    Bilinear, then a nearest-neighbour pass to fill what bilinear cannot.
    The second pass is not cosmetic: a 30 arcsec coastal land cell can have
    ocean (nodata) in three of the four 10 arcmin cells it interpolates from,
    and bilinear would leave it empty. Since this is an ANOMALY field -- smooth,
    with no coastline in it -- taking the nearest value there is right, and it
    keeps the coastline from eroding by up to one coarse cell.
    """
    filled = np.full((grid["height"], grid["width"]), np.nan, dtype="float32")
    source = coarse.astype("float32")
    for resampling in (Resampling.bilinear, Resampling.nearest):
        pass_out = np.full_like(filled, np.nan)
        reproject(source=source, destination=pass_out,
                  src_transform=coarse_transform, src_crs=grid["crs"],
                  src_nodata=np.nan,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  dst_nodata=np.nan, resampling=resampling)
        filled = np.where(np.isfinite(filled), filled, pass_out)
    return filled


def delta_downscale(iso_paths, grid, verbose=True):
    """The 2015-2024 monthly climatology at 30 arcsec.

    Returns (tmin, tmax, prec) each (12, H, W), plus a dict of diagnostics.
    """
    window = coarse_window(grid)
    coarse_transform = Affine(RES_10M, 0.0, grid["bounds"][0],
                              0.0, -RES_10M, grid["bounds"][3])

    print(f"\n  10 arcmin anomaly window: {window.width} x {window.height} "
          f"cells  ({grid['width']} x {grid['height']} at 30 arcsec, "
          f"{int(round(RES_10M / TARGET_RES))}x refinement)")

    diagnostics = {}
    out = {}
    for var in ("tmin", "tmax", "prec"):
        recent = window_climatology(var, *RECENT, window, verbose)
        baseline = window_climatology(var, *BASELINE, window, verbose)

        fine = read_wc_monthly(iso_paths[var], grid)
        if var == "prec":
            # Multiplicative, because a 10% wetter month means something
            # different in a rainforest and in a desert. Additive below
            # PREC_RATIO_FLOOR_MM, where a ratio is meaningless.
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.clip(recent / np.maximum(baseline, 1e-6),
                                *PREC_RATIO_CLAMP)
            ratio = np.where(baseline >= PREC_RATIO_FLOOR_MM, ratio, np.nan)
            additive = recent - baseline

            scaled = np.stack([
                fine[m] * upsample(ratio[m], coarse_transform, grid)
                for m in range(12)])
            shifted = np.stack([
                fine[m] + upsample(additive[m], coarse_transform, grid)
                for m in range(12)])
            adjusted = np.where(np.isfinite(scaled), scaled, shifted)
            adjusted = np.maximum(adjusted, 0.0)      # rainfall cannot be < 0
            diagnostics["prec_ratio_mean"] = float(nanmean(ratio))
        else:
            anomaly = recent - baseline
            adjusted = np.stack([
                fine[m] + upsample(anomaly[m], coarse_transform, grid)
                for m in range(12)])
            diagnostics[f"{var}_anomaly_mean_C"] = float(nanmean(anomaly))

        # How well the cts4.09-derived baseline reproduces the published
        # normals. This is the check that makes the anomaly trustworthy: the
        # two epochs come from the same product, so bias cancels, but only if
        # the baseline really does match.
        coarse_fine = block_reduce_mean(fine, int(round(RES_10M / TARGET_RES)))
        overlap = np.isfinite(coarse_fine) & np.isfinite(baseline)
        if overlap.any():
            diff = (coarse_fine - baseline)[overlap]
            diagnostics[f"{var}_baseline_bias"] = float(diff.mean())
            diagnostics[f"{var}_baseline_absmax"] = float(np.abs(diff).max())

        out[var] = adjusted.astype("float32")

    return out["tmin"], out["tmax"], out["prec"], diagnostics


def block_reduce_mean(arr, factor):
    """Average (12, H, W) down by an integer factor, ignoring NaN."""
    n, h, w = arr.shape
    nh, nw = h // factor, w // factor
    trimmed = arr[:, :nh * factor, :nw * factor]
    return nanmean(trimmed.reshape(n, nh, factor, nw, factor), axis=(2, 4))


# ─────────────────────────────────────────────────────
# WATER DEMAND
# ─────────────────────────────────────────────────────
def latitude_grid(grid):
    """Cell-centre latitude for every row, broadcast over columns."""
    north = grid["transform"].f
    rows = np.arange(grid["height"]) + 0.5
    return (north - rows * TARGET_RES)[:, None]


def extraterrestrial_radiation(grid):
    """Ra in MJ m-2 day-1 for each month, shape (12, H, 1). FAO-56 eq. 21.

    Pure orbital geometry -- latitude and day of year, nothing measured -- so
    this is epoch-free and costs no download. That is exactly why Hargreaves
    is the primary PET here.
    """
    phi = np.deg2rad(latitude_grid(grid))
    doy = MID_MONTH_DOY[:, None, None]

    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    decl = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)

    # Clipped for the polar day and polar night, where the sun never sets or
    # never rises and the unclipped arccos argument leaves [-1, 1].
    sunset = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0))

    return ((24.0 * 60.0 / np.pi) * SOLAR_CONSTANT * dr
            * (sunset * np.sin(phi) * np.sin(decl)
               + np.cos(phi) * np.cos(decl) * np.sin(sunset)))


def saturation_vapour_pressure(t_celsius):
    """es in kPa. FAO-56 eq. 11."""
    return 0.6108 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))


def pet_hargreaves(tmin, tmax, grid):
    """Monthly PET in mm. Hargreaves-Samani, FAO-56 eq. 52.

    Needs only the temperature range as a proxy for radiation and humidity,
    which makes it the one PET fully consistent with the 2015-2024 window.
    """
    ra_mm = extraterrestrial_radiation(grid) / LATENT_HEAT
    tmean = (tmin + tmax) / 2.0
    span = np.maximum(tmax - tmin, 0.0)
    daily = 0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(span)
    return np.maximum(daily, 0.0) * DAYS_IN_MONTH[:, None, None]


def pet_penman(tmin, tmax, srad, wind, vapr, elevation, grid):
    """Monthly reference ET0 in mm. FAO-56 eq. 6.

    srad in MJ m-2 day-1, wind at 2 m in m s-1, vapr in kPa, elevation in m.
    Soil heat flux is taken as zero, which is standard at monthly resolution.
    """
    tmean = (tmin + tmax) / 2.0
    es = (saturation_vapour_pressure(tmax)
          + saturation_vapour_pressure(tmin)) / 2.0
    ea = np.minimum(vapr, es)                 # humidity cannot exceed saturation
    deficit = np.maximum(es - ea, 0.0)

    slope = (4098.0 * saturation_vapour_pressure(tmean)
             / (tmean + 237.3) ** 2)
    pressure = 101.3 * ((293.0 - 0.0065 * elevation) / 293.0) ** 5.26
    gamma = 0.665e-3 * pressure

    ra = extraterrestrial_radiation(grid)
    rso = (0.75 + 2e-5 * elevation) * ra
    rns = (1.0 - ALBEDO) * srad
    with np.errstate(invalid="ignore", divide="ignore"):
        cloud = np.clip(1.35 * srad / np.maximum(rso, 1e-6) - 0.35, 0.05, 1.0)
    rnl = (STEFAN_BOLTZMANN
           * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2.0
           * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0))) * cloud)
    rn = rns - rnl

    numerator = (0.408 * slope * rn
                 + gamma * (900.0 / (tmean + 273.0)) * wind * deficit)
    denominator = slope + gamma * (1.0 + 0.34 * wind)
    daily = np.maximum(numerator / denominator, 0.0)
    return daily * DAYS_IN_MONTH[:, None, None], deficit


# ─────────────────────────────────────────────────────
# BIOCLIM
# ─────────────────────────────────────────────────────
def derive_bioclim(tmin, tmax, prec):
    """The 19 bioclim variables, using the validated project implementation."""
    valid = (np.isfinite(tmin).all(0) & np.isfinite(tmax).all(0)
             & np.isfinite(prec).all(0))
    shape = tmin.shape[1:]

    flat = rcr.biovars_flat(tmin[:, valid].astype("float64"),
                            tmax[:, valid].astype("float64"),
                            prec[:, valid].astype("float64"))
    out = {}
    for i in range(1, 20):
        grid_i = np.full(shape, np.nan, dtype="float32")
        grid_i[valid] = flat[i].astype("float32")
        out[i] = grid_i
    return out, valid


# ─────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────
def crosscheck_published_bio(path, grid, our_baseline_bio, valid):
    """Compare our derived 1970-2000 bioclim against WorldClim's published bio.

    This is the end-to-end check on the derivation: same epoch, same source
    monthly grids, independent implementation. Agreement to rounding says the
    19 variables are right; a systematic gap says a convention is wrong.
    """
    with rasterio.open(path) as src:
        if src.count != WC_BIO_BANDS:
            print(f"  note: {path} has {src.count} bands, expected "
                  f"{WC_BIO_BANDS}; skipping the cross-check")
            return {}
    published = read_wc_country(path, grid)

    print(f"  {'variable':>10} {'ours':>12} {'published':>12} "
          f"{'mean diff':>12} {'max |diff|':>12}")
    out = {}
    for i in (1, 4, 5, 6, 12, 15, 17):
        ours = our_baseline_bio[i]
        theirs = published[i - 1]
        both = valid & np.isfinite(theirs)
        if not both.any():
            continue
        diff = ours[both] - theirs[both]
        out[f"bio{i}_mean_diff"] = float(diff.mean())
        out[f"bio{i}_absmax_diff"] = float(np.abs(diff).max())
        print(f"  {'BIO' + str(i):>10} {ours[both].mean():>12.4f} "
              f"{theirs[both].mean():>12.4f} {diff.mean():>+12.5f} "
              f"{np.abs(diff).max():>12.5f}")
    return out


def build(iso, margin_deg, area_frac, max_gap_deg, keep_baseline, stage,
          crosscheck_bio, bbox=None):
    iso = iso.upper()
    cache_dir = os.path.join(CACHE_ROOT, iso)
    out_dir = os.path.join(OUT_ROOT, iso, "climate_30s")

    print(SEP)
    print("COUNTRY CLIMATE BLOCK  --  30 arcsec (~1 km)")
    print(SEP)
    print(f"country     {iso}")
    print(f"recent      {RECENT[0]}-{RECENT[1]}  (the training window)")
    print(f"baseline    {BASELINE[0]}-{BASELINE[1]}  (what the 30s normals are)")
    print(f"cache       {cache_dir}/")
    print(f"output      {out_dir}/")

    print("\n[1/6] WorldClim 2.1 per-country 30 arcsec files")
    wanted = WC_ISO_VARS + (("bio",) if crosscheck_bio else ())
    paths, missing = fetch_worldclim_country(iso, cache_dir, wanted)
    required = set(WC_ISO_REQUIRED)
    if required - set(paths):
        print(f"\nERROR: missing required variables {sorted(required - set(paths))}. "
              "The 19 bioclim variables cannot be derived without them.")
        return None
    if missing:
        print(f"  note: {missing} unavailable for {iso}. bio is only a "
              "cross-check and srad/wind only feed the secondary PET, so the "
              "block is still buildable -- the affected outputs are skipped.")

    vapr_zip = fetch_vapr_10m(CACHE_ROOT)

    if stage == "download":
        print("\n--stage download: stopping before any processing.")
        return None

    print("\n[2/6] grid")
    grid = grid_from_worldclim(paths["tmin"], iso, margin_deg, area_frac,
                              max_gap_deg, bbox)
    inside = country_mask(grid)
    print(f"  extent    {grid['bounds'][0]:.4f} {grid['bounds'][1]:.4f} "
          f"{grid['bounds'][2]:.4f} {grid['bounds'][3]:.4f}  "
          f"(from {grid['grid_source']})")
    print(f"  size      {grid['width']} x {grid['height']} = "
          f"{grid['width'] * grid['height']:,} cells at 30 arcsec")
    print(f"  in country {int(inside.sum()):,} cells "
          f"({100.0 * inside.mean():.1f}% of the window)")
    print("  lattice   on both the 30 arcsec and 10 arcmin global grids, so "
          "this nests 20x20 inside the existing blocks")

    print("\n[3/6] delta-downscaling the monthly climatology to 2015-2024")
    tmin, tmax, prec, diagnostics = delta_downscale(paths, grid)
    for key, value in diagnostics.items():
        print(f"  {key:<26} {value:+.4f}")
    print("  ^ baseline_bias is how far the cts4.09-derived 1970-2000 mean "
          "sits from\n    WorldClim's published normals, aggregated to "
          "10 arcmin. Near zero means the\n    anomaly is measured between "
          "consistent epochs and systematic bias cancels.")

    print("\n[4/6] the 19 bioclim variables")
    bio, valid = derive_bioclim(tmin, tmax, prec)
    print(f"  computed on {int(valid.sum()):,} cells "
          f"({100.0 * valid.mean():.1f}% of the window, "
          f"{100.0 * (valid & inside.astype(bool)).sum() / max(inside.sum(), 1):.1f}% "
          "of the country)")
    print(f"  BIO1 mean {np.nanmean(bio[1][valid]):7.3f} C     "
          f"BIO12 mean {np.nanmean(bio[12][valid]):8.1f} mm")

    written = []
    common = {"PERIOD": f"{RECENT[0]}-{RECENT[1]}",
              "COUNTRY": iso,
              "RESOLUTION": "30 arcsec (~1 km)",
              "DERIVED_BY": "country_climate_rasters.py"}
    for i in range(1, 20):
        written.append(write_layer(
            os.path.join(out_dir, f"wc2.1_30s_bio_{i}.tif"), bio[i], grid,
            dict(common, BIOCLIM=i, DESCRIPTION=rcr.BIO_LONG_NAMES[i],
                 METHOD="delta-downscaled from WorldClim 30s 1970-2000 "
                        "normals using a 10 arcmin cts4.09 anomaly; bioclim "
                        "per O'Donnell & Ignizio (2012) / dismo::biovars",
                 SOURCE="WorldClim 2.1")))

    if keep_baseline or crosscheck_bio:
        base_bio, base_valid = derive_bioclim(
            read_wc_monthly(paths["tmin"], grid),
            read_wc_monthly(paths["tmax"], grid),
            read_wc_monthly(paths["prec"], grid))
        shift = np.nanmean(bio[1][valid & base_valid]
                           - base_bio[1][valid & base_valid])
        print(f"  BIO1 moved {shift:+.3f} C between {BASELINE[0]}-"
              f"{BASELINE[1]} and {RECENT[0]}-{RECENT[1]}")

        if crosscheck_bio and "bio" in paths:
            print("\n  cross-check: our 1970-2000 derivation vs WorldClim's "
                  "published 30 arcsec bio")
            diagnostics.update(crosscheck_published_bio(
                paths["bio"], grid, base_bio, base_valid))

        if keep_baseline:
            base_dir = os.path.join(out_dir, "baseline_1970_2000")
            for i in range(1, 20):
                written.append(write_layer(
                    os.path.join(base_dir, f"wc2.1_30s_bio_{i}.tif"),
                    base_bio[i], grid,
                    dict(common, PERIOD="1970-2000", BIOCLIM=i,
                         DESCRIPTION=rcr.BIO_LONG_NAMES[i],
                         METHOD="derived from the untouched WorldClim 30s "
                                "normals",
                         SOURCE="WorldClim 2.1")))

    print("\n[5/6] water demand")
    elevation = None
    if "elev" in paths:
        elevation = read_wc_country(paths["elev"], grid, indexes=1)
        written.append(write_layer(
            os.path.join(out_dir, "elevation_worldclim.tif"), elevation, grid,
            dict(common, PERIOD="n/a", DESCRIPTION="WorldClim elevation (m); "
                 "a grid control, NOT the production terrain layer -- "
                 "its heritage is SRTM/GMTED, which are surface models",
                 SOURCE="WorldClim 2.1")))

    pet_h = pet_hargreaves(tmin, tmax, grid)
    pet_h_annual = pet_h.sum(axis=0)
    prec_annual = prec.sum(axis=0)
    written.append(write_layer(
        os.path.join(out_dir, "pet_hargreaves_annual.tif"), pet_h_annual, grid,
        dict(common, DESCRIPTION="annual potential evapotranspiration (mm), "
             "Hargreaves-Samani",
             METHOD="FAO-56 eq. 52 from 2015-2024 tmin/tmax and analytic "
                    "extraterrestrial radiation; fully epoch-consistent",
             SOURCE="derived")))

    print(f"  PET Hargreaves  mean {np.nanmean(pet_h_annual[valid]):8.1f} mm/yr")

    aridity_sources = [("hargreaves", pet_h_annual)]
    if {"srad", "wind"} <= set(paths) and elevation is not None and vapr_zip:
        srad = read_wc_monthly(paths["srad"], grid) * SRAD_KJ_TO_MJ
        wind = read_wc_monthly(paths["wind"], grid) * WIND_10M_TO_2M

        coarse_vapr = np.stack([
            rcr._read_tif_in_zip(vapr_zip, f"wc2.1_10m_vapr_{m:02d}.tif")
            for m in range(1, 13)])
        window = coarse_window(grid)
        coarse_transform = Affine(RES_10M, 0.0, grid["bounds"][0],
                                  0.0, -RES_10M, grid["bounds"][3])
        sliced = coarse_vapr[:, window.row_off:window.row_off + window.height,
                             window.col_off:window.col_off + window.width]
        vapr = np.stack([upsample(sliced[m], coarse_transform, grid)
                         for m in range(12)])

        pet_p, vpd = pet_penman(tmin, tmax, srad, wind, vapr, elevation, grid)
        pet_p_annual = pet_p.sum(axis=0)
        aridity_sources.append(("penman", pet_p_annual))

        ratio = np.nanmean(pet_p_annual[valid]) / np.nanmean(pet_h_annual[valid])
        print(f"  PET Penman      mean {np.nanmean(pet_p_annual[valid]):8.1f} "
              f"mm/yr   (Penman / Hargreaves = {ratio:.3f})")
        print("  ^ Penman mixes epochs: 1970-2000 srad/wind/vapr with "
              "2015-2024 temperature.\n    vapour pressure rises with warming, "
              "so using the older normals understates\n    the deficit and "
              "therefore understates PET. Prefer Hargreaves where they "
              "disagree.")

        for name, array, desc, method in (
            ("pet_penman_annual", pet_p_annual,
             "annual reference evapotranspiration (mm), FAO-56 "
             "Penman-Monteith",
             "FAO-56 eq. 6; MIXED EPOCH -- srad/wind/vapr are 1970-2000 "
             "normals, temperature is 2015-2024"),
            ("vpd_annual_mean", nanmean(vpd, axis=0),
             "annual mean vapour pressure deficit (kPa)",
             "mean of es(tmax), es(tmin) minus actual vapour pressure"),
            ("srad_annual_mean", nanmean(srad, axis=0),
             "annual mean solar radiation (MJ m-2 day-1)",
             "WorldClim 2.1 30 arcsec 1970-2000 normals"),
            ("wind_annual_mean", nanmean(wind, axis=0),
             "annual mean wind speed at 2 m (m s-1)",
             "WorldClim 2.1 30 arcsec 1970-2000 normals, 10 m scaled by "
             f"{WIND_10M_TO_2M} per the FAO-56 log profile"),
            ("vapr_annual_mean", nanmean(vapr, axis=0),
             "annual mean vapour pressure (kPa)",
             "WorldClim 2.1 10 ARCMIN 1970-2000 normals, interpolated to 30 "
             "arcsec -- no per-country 30 arcsec vapr file exists, so this is "
             "the one layer here whose information content is coarser than "
             "its grid"),
        ):
            written.append(write_layer(
                os.path.join(out_dir, f"{name}.tif"), array, grid,
                dict(common, DESCRIPTION=desc, METHOD=method,
                     SOURCE="WorldClim 2.1 / derived")))

        deficit = np.maximum(pet_p - prec, 0.0).sum(axis=0)
        written.append(write_layer(
            os.path.join(out_dir, "water_deficit_annual.tif"), deficit, grid,
            dict(common, DESCRIPTION="annual climatic water deficit (mm)",
                 METHOD="sum over months of max(0, PET_penman - "
                        "precipitation)", SOURCE="derived")))
    else:
        print("  skipping Penman PET, VPD and water deficit: needs srad, "
              "wind, elevation and vapr")

    for name, pet_annual in aridity_sources:
        with np.errstate(invalid="ignore", divide="ignore"):
            aridity = prec_annual / np.maximum(pet_annual, 1e-6)
        suffix = "" if len(aridity_sources) == 1 else f"_{name}"
        written.append(write_layer(
            os.path.join(out_dir, f"aridity_index{suffix}.tif"), aridity, grid,
            dict(common, DESCRIPTION="aridity index, annual precipitation / "
                 "annual PET (UNEP definition; <0.65 is dryland)",
                 METHOD=f"prec 2015-2024 divided by PET ({name})",
                 SOURCE="derived")))
        print(f"  aridity ({name:<10}) mean "
              f"{np.nanmean(aridity[valid]):6.3f}   "
              f"drylands (<0.65) {100.0 * np.nanmean(aridity[valid] < 0.65):.1f}% "
              "of cells")

    print("\n[6/6] manifest and grid definition")
    manifest = {
        "country": iso,
        "block": "climate_30s",
        "resolution": "30 arcsec (1/120 deg, ~1 km)",
        "grid": {"crs": "EPSG:4326",
                 "transform": list(grid["transform"])[:6],
                 "width": grid["width"], "height": grid["height"],
                 "bounds": list(grid["bounds"]),
                 "defined_by": grid["grid_source"],
                 "nests_in_10arcmin": True,
                 "refinement_factor": int(round(RES_10M / TARGET_RES))},
        "recent_window": list(RECENT),
        "baseline_window": list(BASELINE),
        "method": ("WorldClim 2.1 per-country 30 arcsec 1970-2000 normals, "
                   "moved to 2015-2024 by a 10 arcmin delta (change-factor) "
                   "anomaly from the cts4.09 historical monthly product; "
                   "bioclim per recent_climate_rasters.py"),
        "diagnostics": diagnostics,
        "missing_worldclim_variables": missing,
        "layers": sorted(os.path.relpath(p, out_dir) for p in written),
        "licence": ("WorldClim 2.1: free for academic and non-commercial use; "
                    "do not redistribute"),
        "caveats": [
            "The 2015-2024 climatology is delta-downscaled, not measured at "
            "30 arcsec. Fine spatial structure is 1970-2000; only the epoch "
            "shift is interpolated.",
            "srad, wind and vapr are 1970-2000 normals. WorldClim does not "
            "publish them in the historical monthly product, so no recent "
            "version exists without credentials.",
            "vapr has no per-country 30 arcsec file and is interpolated from "
            "10 arcmin.",
            "pet_penman therefore mixes epochs and understates PET; "
            "pet_hargreaves is epoch-consistent and is the primary.",
        ],
    }
    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    grid_path = os.path.join(OUT_ROOT, iso, "country_grid.json")
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump({"country": iso, "res_deg": TARGET_RES,
                   "crs": "EPSG:4326",
                   "transform": list(grid["transform"])[:6],
                   "width": grid["width"], "height": grid["height"],
                   "bounds": list(grid["bounds"]),
                   "margin_deg": margin_deg, "area_frac": area_frac,
                   "max_gap_deg": max_gap_deg,
                   "bbox": list(bbox) if bbox else None,
                   "defined_by": grid["grid_source"]}, f, indent=2)

    print(f"  {manifest_path}")
    print(f"  {grid_path}   <- every other country block reads this")

    total = sum(os.path.getsize(p) for p in written)
    print("\n" + SEP)
    print(f"DONE  --  {len(written)} layers, {total / 1e6:.1f} MB")
    print(SEP)
    print("Next: python3 country_topography_rasters.py "
          f"{iso}\n      python3 country_soil_rasters.py {iso}"
          f"\n      python3 verify_country_alignment.py {iso}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build the 30 arcsec country climate block for 2015-2024.")
    parser.add_argument("iso", help="3-letter country code")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="margin in degrees for the polygon metadata; the "
                             "extent itself comes from WorldClim (default 0)")
    parser.add_argument("--area-frac", type=float, default=1.0)
    parser.add_argument("--max-gap", type=float, default=10.0)
    parser.add_argument("--keep-baseline", action="store_true",
                        help="also write the untouched 1970-2000 bioclim, for "
                             "measuring the epoch shift")
    parser.add_argument("--crosscheck-bio", action="store_true",
                        help="also fetch WorldClim's published 30 arcsec bio "
                             "and compare it against our own derivation "
                             "(the largest download on offer)")
    parser.add_argument("--stage", choices=("download", "all"), default="all",
                        help="'download' fetches the sources and stops")
    parser.add_argument("--bbox", default=None, metavar="W,S,E,N",
                        help="restrict WorldClim's per-country extent to this "
                             "box, in degrees. Needed wherever the sovereign "
                             "span is not the modelling extent: USA reaches "
                             "Guam to Maine, ESP reaches the Canaries")
    args = parser.parse_args()

    bbox = None
    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))
        if len(bbox) != 4:
            parser.error("--bbox needs exactly four numbers: W,S,E,N")

    if not os.path.isdir(MONTHLY_SRC):
        print(f"ERROR: {MONTHLY_SRC}/ not found. It holds the 10 arcmin "
              "monthly archives the anomaly is measured from; run "
              "recent_climate_rasters.py first.")
        sys.exit(2)

    result = build(args.iso, args.margin, args.area_frac, args.max_gap,
                   args.keep_baseline, args.stage, args.crosscheck_bio, bbox)
    if result is None and args.stage != "download":
        sys.exit(1)


if __name__ == "__main__":
    main()
