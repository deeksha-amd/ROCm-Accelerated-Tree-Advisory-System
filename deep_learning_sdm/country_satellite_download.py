"""
DOWNLOAD ONLY. Fetch NDVI and land cover for one country and stop there.

READ THIS BEFORE USING THE OUTPUT
    These layers are NOT predictors, and this script deliberately does not put
    them anywhere they could become predictors. It does not write onto the
    country predictor grid, it does not touch country_data/<ISO>/*_30s/, and it
    builds no combined table. Everything lands under
    country_data/<ISO>/satellite_download/ at each source's OWN native
    resolution, on a grid that does not match the predictor stack, so merging
    them is something you would have to do on purpose.

WHY THE SEPARATION IS STRUCTURAL AND NOT JUST ADVISORY
    Earlier work in this project measured that vegetation layers are CIRCULAR
    for this problem, and the numbers are worth restating because they are the
    reason for every awkward choice in this file:

    * A model given ONLY NDVI and tree-cover fraction -- no climate, no soil,
      no terrain -- reached cross-validated AUC 0.791, against 0.711 for the
      entire genuine environmental predictor set.
    * That 0.080 advantage collapsed to 0.001 on cleared farmland, which is
      exactly the land an afforestation advisory has to rank. The lead is not
      ecological skill; it is the model reading off where trees already are.
    * Tree-cover fraction separated occurrences from background in the same
      direction for 90% of 40 species, against 52% for annual mean
      temperature. Separating almost every species the same way is the
      signature of a variable answering "are there trees here?" rather than
      describing an ecological niche.

    So they are retained as OUTPUT FILTERS and DIAGNOSTICS: use them to mask
    recommendations away from water, ice, built-up land and existing closed
    forest, and to sanity-check predictions. Never as model input.

EPOCHS, AND WHERE THEY CANNOT MATCH
    The target is climate_recent's 2015-2024 window, and the wider MODE="full"
    settings from satellite_rasters.py are used rather than the sample
    defaults.

    NDVI            2015-2022. PKU GIMMS v1.2 ends in 2022, so this covers 8
                    of the 10 years.
    soil moisture   2015-2024. ESA CCI v09.2 covers the window in full.
    land cover      2020, AND THAT IS THE HONEST LIMIT. ESA CCI Land Cover PFT
                    is published only to 2020, and there is no
                    credential-free land-cover product past 2022 at all --
                    Copernicus Global Land needs a VITO account, and the
                    MODIS and VIIRS products need a NASA Earthdata login. So
                    land cover cannot be matched to 2015-2024 as tightly as
                    NDVI can. 2020 does at least sit inside the window, which
                    is the best available position for a single-year product.

    Surface soil moisture is the one layer here that is NOT a vegetation
    proxy -- it is a microwave measurement, not greenness -- so it is the only
    genuine predictor candidate in the set. It is still written here rather
    than into the predictor block, because promoting it is a modelling decision
    for the user to make deliberately, and because it reaches only ~79% of
    occurrence points, with the holes sitting under dense canopy.

OUTPUTS  (country_data/<ISO>/satellite_download/)
    landcover_pft_2020/     per-PFT fractional cover, native 300 m
    ndvi_pku_<year>.tif     24 semi-monthly composites per year, native 1/12 deg
    soilmoisture_cci/       the sampled daily files, native 0.25 deg
    DO_NOT_TRAIN_ON_THIS.md
    MANIFEST.json

USAGE
    python3 country_satellite_download.py IND
    python3 country_satellite_download.py IND --skip soilmoisture
    python3 country_satellite_download.py IND --ndvi-years 2021 2022
"""

import argparse
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from tqdm import tqdm

import satellite_rasters as sat
from country_clip import country_grid

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUT_ROOT = "country_data"
BLOCK_NAME = "satellite_download"

# MODE="full" epochs from satellite_rasters.py, not the sample defaults.
NDVI_YEARS = sat.NDVI_YEARS_FULL           # 2015-2022
SM_YEARS = sat.SM_YEARS_FULL               # 2015-2024
SM_DAYS = sat.SM_DAYS_FULL                 # days 8 and 22 of each month
PFT_YEAR = sat.PFT_YEAR                    # 2020, the last year published

# Which PFT components to keep. All of them are written, but the tree ones are
# tagged as filters-and-diagnostics-only in their metadata, so the restriction
# travels with the file rather than living only in a document.
PFT_KEEP = ("tree", "tree_broadleaf", "tree_needleleaf", "shrub",
            "grass_natural", "cropland", "builtup", "bare", "water", "snowice")
PFT_FILTER_SAFE = ("builtup", "water", "snowice", "bare", "cropland")

# A margin around the country, in degrees, so the filters still apply at the
# border. These are not model inputs, so exact alignment is not required and
# deliberately not provided.
MARGIN_DEG = 0.25

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "200000000",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

WARNING_TEXT = """# Do not train on anything in this directory

These layers are **output filters and diagnostics**, not predictors.

Measured in this project:

- NDVI and tree-cover fraction alone -- no climate, soil or terrain -- reach
  cross-validated **AUC 0.791**, against **0.711** for the entire genuine
  environmental predictor set.
- That advantage **collapses to +0.001 on cleared farmland**, which is the land
  an afforestation advisory actually has to rank. The lead is the model reading
  off where trees already grow, not ecological skill.
- Tree-cover fraction separates occurrences from background **in the same
  direction for 90% of 40 species** (annual mean temperature: 52%) -- the
  signature of a variable answering "are there trees here?" rather than
  describing a niche.

So: predicting where trees *can* grow from where trees *already* grow is
circular, and it fails precisely on the land the advisory is for.

## What these are for

- **Filters on model OUTPUT.** Mask recommendations away from water, permanent
  ice, built-up land and existing closed forest.
- **Diagnostics.** Check whether predictions look sane against observed cover.

## Why they are not on the predictor grid

Deliberately. Each layer is at its source's native resolution, which does not
match `country_data/<ISO>/climate_30s`, `soil_30s` or `topography_30s`. Nothing
here can be stacked with the predictors by accident.

`soilmoisture_cci/` is the one exception in kind: microwave soil moisture is
not a vegetation proxy and is a legitimate predictor candidate. It is still
here rather than in the predictor block because promoting it should be a
deliberate decision, and because it reaches only ~79% of occurrence points,
with the gaps under dense canopy.

## Epoch limits

`landcover` is **2020 only**. ESA CCI Land Cover PFT stops there and no
credential-free product runs past 2022, so land cover cannot match the
2015-2024 climate window as tightly as NDVI (2015-2022) does.
"""

SEP = "=" * 70


# ─────────────────────────────────────────────────────
# WINDOWS
# ─────────────────────────────────────────────────────
def country_bounds(iso, margin_deg, area_frac, max_gap_deg, bbox=None):
    """The country's extent, plus a margin. Not snapped to any project grid.

    `bbox` overrides the geometry-derived extent, for the same reason
    country_clip.py offers one: the distance filter cannot separate the
    Canaries from mainland Spain (7.1 deg apart, inside any threshold that
    keeps a real archipelago whole) or Alaska from the lower 48 (5.5 deg).
    """
    grid = country_grid(iso, res_deg=1.0 / 120.0, margin_deg=margin_deg,
                        area_frac=area_frac, max_gap_deg=max_gap_deg,
                        bbox=bbox)
    return grid["bounds"], grid["name"]


def write_window(src, bounds, out_path, band=1, tags=None, descriptions=None,
                 dtype=None):
    """Copy a native-resolution window of an open dataset to a GeoTIFF."""
    west, south, east, north = bounds
    window = from_bounds(west, south, east, north,
                         src.transform).round_offsets().round_lengths()
    window = window.intersection(
        rasterio.windows.Window(0, 0, src.width, src.height))

    bands = [band] if isinstance(band, int) else list(band)
    data = src.read(bands, window=window)
    transform = src.window_transform(window)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff",
                       width=data.shape[2], height=data.shape[1],
                       count=data.shape[0], dtype=dtype or data.dtype,
                       crs=src.crs or "EPSG:4326", transform=transform,
                       nodata=src.nodata, compress="deflate", tiled=True,
                       blockxsize=256, blockysize=256,
                       BIGTIFF="IF_SAFER") as dst:
        dst.write(data.astype(dtype or data.dtype))
        dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
        for index, text in enumerate(descriptions or [], start=1):
            dst.set_band_description(index, text)
    return out_path, data.shape


# ─────────────────────────────────────────────────────
# LAND COVER
# ─────────────────────────────────────────────────────
def fetch_landcover(bounds, out_dir, iso):
    """Per-PFT fractional cover for the country, at native 300 m.

    Read straight out of the remote NetCDF through GDAL's netCDF driver, one
    subdataset per plant functional type, so nothing but the country window
    travels out of a 1.8 GB file.
    """
    url = f"{sat.PFT_BASE}/{sat.PFT_FILE}"
    target = os.path.join(out_dir, f"landcover_pft_{PFT_YEAR}")
    os.makedirs(target, exist_ok=True)

    print(f"  source {url}")
    print(f"  ESA CCI Land Cover PFT {PFT_YEAR}, 300 m -- the last year "
          "published")

    written = []
    variables = sorted({v for name in PFT_KEEP
                        for v in sat.PFT_COMPONENTS[name]})
    with rasterio.Env(**GDAL_ENV):
        for variable in tqdm(variables, desc="land cover PFT", unit="layer"):
            subdataset = sat.pft_subdataset(f"/vsicurl/{url}", variable)
            out_path = os.path.join(target, f"pft_{variable.lower()}.tif")
            try:
                with rasterio.open(subdataset) as src:
                    role = ("filter and diagnostic only -- CIRCULAR as a "
                            "predictor" if variable.startswith(("TREES",
                                                                "SHRUBS"))
                            else "filter and diagnostic only")
                    path, shape = write_window(
                        src, bounds, out_path,
                        tags={"COUNTRY": iso, "PFT": variable,
                              "YEAR": PFT_YEAR,
                              "NATIVE_RESOLUTION": "300 m",
                              "SOURCE": f"ESA CCI Land Cover PFT "
                                        f"{sat.PFT_VERSION}",
                              "ROLE": role,
                              "NOT_A_PREDICTOR": "true",
                              "EPOCH_LIMIT": "published to 2020 only; cannot "
                                             "match the 2015-2024 climate "
                                             "window",
                              "DOWNLOADED_BY":
                                  "country_satellite_download.py"},
                        descriptions=[f"{variable} fractional cover (%)"])
                written.append(path)
            except rasterio.errors.RasterioIOError as exc:
                print(f"    {variable}: could not read ({str(exc)[:90]})")

    if written:
        with rasterio.open(written[0]) as src:
            print(f"  window {src.width} x {src.height} at 300 m, "
                  f"{len(written)} PFT layers")
    return written


# ─────────────────────────────────────────────────────
# NDVI
# ─────────────────────────────────────────────────────
def fetch_ndvi(bounds, out_dir, iso, years, raw_dir):
    """Semi-monthly NDVI composites for the country, one file per year.

    The members are pulled out of the 775 MB Zenodo archive with HTTP range
    requests, so the archive is never downloaded whole.
    """
    print(f"  source {sat.NDVI_ZIP}")
    print(f"  PKU GIMMS NDVI {sat.NDVI_VERSION}, 1/12 deg, "
          f"{years[0]}-{years[-1]} (source ends 2022)")
    os.makedirs(raw_dir, exist_ok=True)

    index = sat.RemoteZip(sat.NDVI_ZIP)
    members = sat.ndvi_member_names(index, set(years))
    if not members:
        print("  no matching composites in the archive")
        return []
    print(f"  {len(members)} semi-monthly composites")

    paths = sat.fetch_ndvi_members(index, members, raw_dir)
    by_year = {}
    for (year, slot, _), path in zip(members, paths):
        by_year.setdefault(year, []).append((slot, path))

    written = []
    for year in sorted(by_year):
        entries = sorted(by_year[year])
        stack, transform, crs, nodata = None, None, None, None
        for position, (slot, path) in enumerate(entries):
            with rasterio.open(path) as src:
                window = from_bounds(*bounds,
                                     src.transform).round_offsets(
                                         ).round_lengths()
                window = window.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height))
                band = src.read(1, window=window)
                if stack is None:
                    stack = np.zeros((len(entries),) + band.shape,
                                     dtype=band.dtype)
                    transform = src.window_transform(window)
                    crs, nodata = src.crs, src.nodata
            stack[position] = band

        out_path = os.path.join(out_dir, f"ndvi_pku_{year}.tif")
        with rasterio.open(out_path, "w", driver="GTiff",
                           width=stack.shape[2], height=stack.shape[1],
                           count=stack.shape[0], dtype=stack.dtype,
                           crs=crs or "EPSG:4326", transform=transform,
                           nodata=sat.NDVI_FILL, compress="deflate",
                           tiled=True, blockxsize=256, blockysize=256,
                           BIGTIFF="IF_SAFER") as dst:
            dst.write(stack)
            dst.update_tags(
                COUNTRY=iso, YEAR=year,
                SOURCE=f"PKU GIMMS NDVI {sat.NDVI_VERSION}",
                URL=sat.NDVI_ZIP,
                NATIVE_RESOLUTION="1/12 deg (~9.3 km)",
                SCALE=sat.NDVI_SCALE, FILL_VALUE=sat.NDVI_FILL,
                COMPOSITES=len(entries),
                ROLE="filter and diagnostic only -- CIRCULAR as a predictor",
                NOT_A_PREDICTOR="true",
                DOWNLOADED_BY="country_satellite_download.py")
            for position, (slot, _) in enumerate(entries, start=1):
                half = "first" if slot % 2 == 0 else "second"
                dst.set_band_description(
                    position, f"{year}-{slot // 2 + 1:02d} {half} half, "
                              f"NDVI x {1 / sat.NDVI_SCALE:.0f}")
        written.append(out_path)
        print(f"  {os.path.basename(out_path)}  {stack.shape[0]} bands, "
              f"{stack.shape[2]} x {stack.shape[1]}")

    return written


# ─────────────────────────────────────────────────────
# SOIL MOISTURE
# ─────────────────────────────────────────────────────
def fetch_soil_moisture(bounds, out_dir, iso, years, days, raw_dir):
    """The sampled daily surface soil moisture files, clipped to the country.

    The only non-vegetation layer here, so the only legitimate predictor
    candidate -- but see DO_NOT_TRAIN_ON_THIS.md before promoting it.
    """
    print(f"  source {sat.SM_BASE}")
    print(f"  ESA CCI Soil Moisture {sat.SM_VERSION}, 0.25 deg, "
          f"{years[0]}-{years[-1]}, days {days} of each month")
    target = os.path.join(out_dir, "soilmoisture_cci")
    os.makedirs(target, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    urls = sat.soil_moisture_urls(years, days)
    written, failed = [], 0
    for _, url in tqdm(urls, desc="soil moisture", unit="file"):
        name = os.path.basename(url)
        raw_path = os.path.join(raw_dir, name)
        if not os.path.exists(raw_path) or os.path.getsize(raw_path) < 1000:
            if not sat.download_file(url, raw_path, desc=name[-24:]):
                failed += 1
                continue

        out_path = os.path.join(target, name.replace(".nc", ".tif"))
        if os.path.exists(out_path):
            written.append(out_path)
            continue
        try:
            with rasterio.open(sat.pft_subdataset(raw_path, "sm")) as src:
                path, _ = write_window(
                    src, bounds, out_path,
                    tags={"COUNTRY": iso, "SOURCE": "ESA CCI Soil Moisture "
                                                    f"{sat.SM_VERSION}",
                          "NATIVE_RESOLUTION": "0.25 deg",
                          "ROLE": "microwave measurement, NOT a vegetation "
                                  "proxy; the one legitimate predictor "
                                  "candidate here",
                          "NOT_A_PREDICTOR": "not integrated by decision",
                          "DOWNLOADED_BY": "country_satellite_download.py"},
                    descriptions=["surface soil moisture (m3 m-3)"])
            written.append(path)
        except rasterio.errors.RasterioIOError:
            failed += 1

    print(f"  {len(written)} files clipped, {failed} unavailable")
    return written


# ─────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────
def build(iso, margin_deg, area_frac, max_gap_deg, skip, ndvi_years,
          bbox=None):
    iso = iso.upper()
    out_dir = os.path.join(OUT_ROOT, iso, BLOCK_NAME)
    raw_root = os.path.join("country_src_cache", "satellite", iso)

    print(SEP)
    print("COUNTRY SATELLITE DOWNLOAD  --  DOWNLOAD ONLY, NOT INTEGRATED")
    print(SEP)
    bounds, name = country_bounds(iso, margin_deg, area_frac, max_gap_deg,
                                  bbox)
    print(f"country     {name} ({iso})")
    print(f"bounds      {bounds[0]:.3f} {bounds[1]:.3f} {bounds[2]:.3f} "
          f"{bounds[3]:.3f}  (+{margin_deg} deg margin)")
    print(f"output      {out_dir}/")
    print("\nThese layers are output FILTERS and DIAGNOSTICS. They are written")
    print("at each source's native resolution, NOT on the predictor grid, so")
    print("they cannot be stacked with climate, soil or terrain by accident.")
    print("Vegetation indices are circular for this problem: NDVI and tree")
    print("cover alone score AUC 0.791 against 0.711 for the whole genuine")
    print("environmental set, and that lead falls to +0.001 on cleared")
    print("farmland -- the land the advisory actually has to rank.")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "DO_NOT_TRAIN_ON_THIS.md"), "w",
              encoding="utf-8") as f:
        f.write(WARNING_TEXT)

    results = {}

    if "landcover" not in skip:
        print(f"\n[1/3] LAND COVER  ({PFT_YEAR} only -- see the epoch note)")
        results["landcover"] = fetch_landcover(bounds, out_dir, iso)
    else:
        print("\n[1/3] LAND COVER  skipped")
        results["landcover"] = []

    if "ndvi" not in skip:
        print(f"\n[2/3] NDVI  ({ndvi_years[0]}-{ndvi_years[-1]})")
        results["ndvi"] = fetch_ndvi(bounds, out_dir, iso, ndvi_years,
                                     os.path.join(raw_root, "ndvi"))
    else:
        print("\n[2/3] NDVI  skipped")
        results["ndvi"] = []

    if "soilmoisture" not in skip:
        print(f"\n[3/3] SOIL MOISTURE  ({SM_YEARS[0]}-{SM_YEARS[-1]})")
        results["soilmoisture"] = fetch_soil_moisture(
            bounds, out_dir, iso, SM_YEARS, SM_DAYS,
            os.path.join(raw_root, "soilmoisture"))
    else:
        print("\n[3/3] SOIL MOISTURE  skipped")
        results["soilmoisture"] = []

    manifest = {
        "country": iso,
        "block": BLOCK_NAME,
        "status": "DOWNLOAD ONLY -- deliberately not integrated",
        "on_predictor_grid": False,
        "bounds": list(bounds),
        "margin_deg": margin_deg,
        "why_not_integrated": {
            "vegetation_only_auc": 0.791,
            "environmental_set_auc": 0.711,
            "advantage_on_cleared_farmland": 0.001,
            "tree_cover_same_direction_species": "90% of 40 (temperature: 52%)",
            "conclusion": ("vegetation layers answer 'are there trees here?', "
                           "not 'what niche is this?'. Predicting where trees "
                           "CAN grow from where they DO grow is circular, and "
                           "the advantage vanishes on exactly the cleared land "
                           "an afforestation advisory has to rank."),
            "permitted_use": "filters on model output, and diagnostics",
        },
        "sources": {
            "landcover": {
                "name": f"ESA CCI Land Cover PFT {sat.PFT_VERSION}",
                "year": PFT_YEAR,
                "native_resolution": "300 m",
                "files": len(results["landcover"]),
                "epoch_limit": ("published to 2020 only. No credential-free "
                                "land-cover product runs past 2022: "
                                "Copernicus Global Land needs a VITO account, "
                                "MODIS and VIIRS need NASA Earthdata. So land "
                                "cover CANNOT match 2015-2024 as tightly as "
                                "NDVI does; 2020 at least falls inside the "
                                "window."),
                "role": "filter and diagnostic only",
            },
            "ndvi": {
                "name": f"PKU GIMMS NDVI {sat.NDVI_VERSION}",
                "years": list(ndvi_years),
                "native_resolution": "1/12 deg (~9.3 km)",
                "temporal_resolution": "semi-monthly, 24 per year",
                "files": len(results["ndvi"]),
                "epoch_note": (f"{ndvi_years[0]}-{ndvi_years[-1]}; the source "
                               "ends in 2022, so this covers 8 of the 10 "
                               "years of the 2015-2024 climate window"),
                "role": "filter and diagnostic only -- circular as a predictor",
                "licence": "CC-BY 4.0",
            },
            "soilmoisture": {
                "name": f"ESA CCI Soil Moisture {sat.SM_VERSION} COMBINED",
                "years": list(SM_YEARS),
                "day_sample": list(SM_DAYS),
                "native_resolution": "0.25 deg",
                "files": len(results["soilmoisture"]),
                "epoch_note": "2015-2024, matching the climate window in full",
                "role": ("microwave, not greenness -- the only legitimate "
                         "predictor candidate here, left unintegrated by "
                         "decision; reaches only ~79% of occurrence points, "
                         "with gaps under dense canopy"),
            },
        },
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total = 0
    for root, _, files in os.walk(out_dir):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)

    print("\n" + SEP)
    print(f"DONE  --  {sum(len(v) for v in results.values())} files, "
          f"{total / 1e6:.1f} MB in {out_dir}/")
    print(SEP)
    print("NOT integrated, by design. These are output filters and")
    print("diagnostics; see DO_NOT_TRAIN_ON_THIS.md in that directory.")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download NDVI and land cover for one country. Filters "
                    "and diagnostics only -- never merged into the predictor "
                    "stack.")
    parser.add_argument("iso", help="3-letter country code")
    parser.add_argument("--margin", type=float, default=MARGIN_DEG)
    parser.add_argument("--area-frac", type=float, default=1.0)
    parser.add_argument("--max-gap", type=float, default=10.0)
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=("landcover", "ndvi", "soilmoisture"))
    parser.add_argument("--bbox", default=None, metavar="W,S,E,N",
                        help="override the geometry-derived extent, in "
                             "degrees; the distance filter cannot cut the "
                             "Canaries off Spain or Alaska off the lower 48")
    parser.add_argument("--ndvi-years", nargs="+", type=int,
                        default=NDVI_YEARS,
                        help=f"default {NDVI_YEARS[0]}-{NDVI_YEARS[-1]}, the "
                             "widest the source allows")
    args = parser.parse_args()

    bbox = None
    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))
        if len(bbox) != 4:
            parser.error("--bbox needs exactly four numbers: W,S,E,N")

    if build(args.iso, args.margin, args.area_frac, args.max_gap,
             set(args.skip), sorted(args.ndvi_years), bbox) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
