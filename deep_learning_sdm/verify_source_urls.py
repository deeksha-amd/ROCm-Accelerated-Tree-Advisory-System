"""
HEAD-check every data source this project downloads from, before anyone starts
a multi-gigabyte transfer.

WHY THIS EXISTS
    The country-scoped acquisition pulls from eight independent institutions.
    Any one of them can re-version a path, move a bucket or retire a release,
    and the failure mode is a job that dies forty minutes in with half a block
    written. A HEAD request costs nothing and answers three questions at once:
    is the URL live, does it need credentials, and how big is it really --
    which is the number the resolution decision actually turns on.

    Sizes here are measured, not estimated. Content-Length on a HEAD is what
    the server will really send.

WHAT IT REPORTS
    status      HTTP status after redirects. 200/206 is good. 401/403 means a
                credential wall, which disqualifies a source for this project.
    size        Content-Length, so per-country totals can be summed.
    ranges      whether the server advertises byte ranges, which is what makes
                a windowed read of a 432 GB COG possible instead of a download.

    Totals separate the two access modes, because summing them would be
    nonsense. GEDTM30's DTM is a 432 GB file and OpenLandMap's layers are 5-18
    GB each, but none of them is ever downloaded: only the country window
    travels over the network, a few MB. Only the FETCH total is bytes that will
    actually land on disk.

USAGE
    python3 verify_source_urls.py                  # country-independent sources
    python3 verify_source_urls.py --iso IND        # plus every per-country file
    python3 verify_source_urls.py --iso IND --group climate
    python3 verify_source_urls.py --iso IND --quiet   # totals only
"""

import argparse
import collections
import concurrent.futures as futures

import requests

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
TIMEOUT = 60
THREADS = 16
RETRIES = 2

# Some servers answer HEAD with 403/405 but honour a ranged GET perfectly well,
# so a HEAD failure is retried as a 2-byte range request before being believed.
FALLBACK_STATUSES = (403, 405, 501, 400)

# How a source is consumed. FETCH bytes land on disk; WINDOW bytes do not --
# only the country's window is range-read out of a much larger remote file.
FETCH = "fetch"
WINDOW = "window"

# ── Boundaries ──
NE_BASE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
           "/v5.1.2/geojson")

# ── WorldClim 2.1 ──
WC_BASE = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
WC_HIST = "https://geodata.ucdavis.edu/climate/worldclim/2_1/hist/cts4.09"
WC_ISO = "https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso"

# Per-country 30 arcsec products. NOTE the set is bio/elev/prec/srad/tavg/
# tmax/tmin/wind -- there is no per-country vapr, which is why VAPR_GLOBAL
# below exists.
WC_ISO_VARS = ("bio", "elev", "prec", "srad", "tavg", "tmax", "tmin", "wind")

# Decade blocks of the historical monthly weather product. Already on disk in
# climate_monthly_src/, checked here so a fresh clone knows they are fetchable.
WC_HIST_BLOCKS = ((1970, 1979), (1980, 1989), (1990, 1999), (2000, 2009),
                  (2010, 2019), (2020, 2024))
WC_HIST_VARS = ("tmin", "tmax", "prec")

# ── Terrain: GEDTM30 v1.2 ──
GEDTM30_BASE = "https://s3.opengeohub.org/global/dtm"

# ── Soil ──
SG_250M = "https://files.isric.org/soilgrids/latest/data"
SG_5KM = "https://files.isric.org/soilgrids/latest/data_aggregated/5000m"
OLM_S3 = "https://s3.opengeohub.org/global-soil"

# ── Satellite (download-only, never a predictor) ──
PFT_BASE = "https://dap.ceda.ac.uk/neodc/esacci/land_cover/data/pft/v2.0.81"
NDVI_ZIP = ("https://zenodo.org/records/8253971/files/"
            "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_2011_2022.zip")
SM_BASE = ("https://dap.ceda.ac.uk/neodc/esacci/soil_moisture/data/"
           "daily_files/COMBINED/v09.2")

# ── Considered and not chosen; checked so the comparison stays honest ──
CHELSA_BASE = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies"


# ─────────────────────────────────────────────────────
# URL TABLE
# ─────────────────────────────────────────────────────
def country_independent():
    """(group, label, url, mode) for everything independent of the country."""
    rows = []

    for scale in ("10m", "50m"):
        rows.append(("boundary", f"Natural Earth admin-0 {scale}",
                     f"{NE_BASE}/ne_{scale}_admin_0_countries.geojson", FETCH))

    # Solar radiation, wind and vapour pressure: WorldClim publishes these as
    # 1970-2000 monthly normals ONLY. They are absent from the historical
    # monthly weather product, so there is no 2015-2024 version to download.
    for var in ("srad", "wind", "vapr"):
        rows.append(("climate", f"WorldClim 10m {var} normals (1970-2000)",
                     f"{WC_BASE}/wc2.1_10m_{var}.zip", FETCH))
    rows.append(("climate", "WorldClim 30s vapr normals, GLOBAL "
                            "(no per-country file exists)",
                 f"{WC_BASE}/wc2.1_30s_vapr.zip", FETCH))

    for var in WC_HIST_VARS:
        for lo, hi in WC_HIST_BLOCKS:
            rows.append(("climate-monthly",
                         f"WorldClim hist monthly {var} {lo}-{hi}",
                         f"{WC_HIST}/wc2.1_cruts4.09_10m_{var}_{lo}-{hi}.zip",
                         FETCH))
        rows.append(("climate-monthly", f"WorldClim 10m {var} normals",
                     f"{WC_BASE}/wc2.1_10m_{var}.zip", FETCH))

    rows += [
        ("terrain", "GEDTM30 v1.2 DTM",
         f"{GEDTM30_BASE}/v1.2/gedtm_rf_m_30m_s_20060101_20151231_go_epsg."
         "4326.3855_v1.2.tif", WINDOW),
        ("terrain", "GEDTM30 v1.2 uncertainty",
         f"{GEDTM30_BASE}/v1.2/gedtm_rf_std_30m_s_20060101_20151231_go_epsg."
         "4326.3855_v1.2.tif", WINDOW),
        ("terrain", "GEDTM30 v1.2.0 published slope (cross-check)",
         f"{GEDTM30_BASE}/v1.2.0/slope.in.degree_gedtm_m_30m_s_20060101_"
         "20151231_go_epsg.4326_v1.2.0.tif", WINDOW),
        ("terrain", "GEDTM30 COG index",
         "https://codeberg.org/openlandmap/GEDTM30/raw/branch/main/metadata/"
         "cog_list.csv", FETCH),
    ]

    # SoilGrids 2.0 at native 250 m, as VRTs over tiled COGs. Depth intervals
    # 0-5/5-15/15-30 compose into our 0-30cm slab, 30-60 is direct.
    for prop in ("phh2o", "sand", "silt", "clay", "bdod", "soc", "ocd",
                 "nitrogen", "cec", "cfvo"):
        for depth in ("0-5cm", "5-15cm", "15-30cm", "30-60cm"):
            rows.append(("soil", f"SoilGrids 250m {prop} {depth}",
                         f"{SG_250M}/{prop}/{prop}_{depth}_mean.vrt", WINDOW))

    # OpenLandMap-soildb: the primary source in the existing soil block, native
    # EPSG:4326 at 120 m, so it beats SoilGrids for the properties it covers.
    olm = {
        "ph_h2o": ("global_soil_props_v20250204_mosaics",
                   "ph.h2o_iso.10390.2021.index", "v20250204"),
        "soc": ("global_soil_props_v20250204_mosaics",
                "oc_iso.10694.1995.wpml", "v20250204"),
        "socd": ("global_soil_props_v20250204_mosaics",
                 "oc_iso.10694.1995.mg.cm3", "v20250204"),
        "bdod": ("global_soil_props_v20250204_mosaics",
                 "bd.core_iso.11272.2017.g.cm3", "v20250204"),
        "clay": ("global_soil_props_v20250523",
                 "clay.tot_iso.11277.2020.wpct", "v20250523"),
        "sand": ("global_soil_props_v20250523",
                 "sand.tot_iso.11277.2020.wpct", "v20250523"),
        "silt": ("global_soil_props_v20250523",
                 "silt.tot_iso.11277.2020.wpct", "v20250523"),
    }
    for name, (folder, stem, version) in olm.items():
        for depth in ("b0cm..30cm", "b30cm..60cm"):
            rows.append(("soil", f"OpenLandMap 120m {name} {depth}",
                         f"{OLM_S3}/{folder}/{stem}_m_120m_{depth}_"
                         f"20200101_20221231_g_epsg.4326_{version}.tif",
                         WINDOW))

    rows += [
        ("satellite", "ESA CCI Land Cover PFT 2020 (last year published)",
         f"{PFT_BASE}/ESACCI-LC-L4-PFT-Map-300m-P1Y-2020-v2.0.81.nc", WINDOW),
        ("satellite", "PKU GIMMS NDVI v1.2 2011-2022 (Zenodo)",
         NDVI_ZIP, WINDOW),
        ("satellite", "ESA CCI soil moisture, one daily file",
         f"{SM_BASE}/2020/ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-"
         "20200115000000-fv09.2.nc", FETCH),
    ]

    rows += [
        ("alternative", "CHELSA V2.1 bio1 30s",
         f"{CHELSA_BASE}/1981-2010/bio/CHELSA_bio1_1981-2010_V.2.1.tif",
         WINDOW),
        ("alternative", "CHELSA V2.1 bio12 30s",
         f"{CHELSA_BASE}/1981-2010/bio/CHELSA_bio12_1981-2010_V.2.1.tif",
         WINDOW),
        ("alternative", "CHELSA V2.1 Penman PET, January",
         f"{CHELSA_BASE}/1981-2010/pet/CHELSA_pet_penman_01_1981-2010_"
         "V.2.1.tif", WINDOW),
        ("alternative", "WorldClim 30s bio, GLOBAL zip (the route NOT taken)",
         f"{WC_BASE}/wc2.1_30s_bio.zip", FETCH),
    ]
    return rows


def country_specific(iso):
    """(group, label, url, mode) for the per-country 30 arcsec WorldClim files."""
    iso = iso.upper()
    return [("country-climate", f"WorldClim 30s {iso} {var}",
             f"{WC_ISO}/{iso}_wc2.1_30s_{var}.tif", FETCH)
            for var in WC_ISO_VARS]


# ─────────────────────────────────────────────────────
# CHECKING
# ─────────────────────────────────────────────────────
def check(row):
    """HEAD one URL, falling back to a tiny ranged GET where HEAD is refused."""
    group, label, url, mode = row
    last = None
    for _ in range(RETRIES):
        try:
            r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
            if r.status_code in FALLBACK_STATUSES:
                r = requests.get(url, stream=True, timeout=TIMEOUT,
                                 headers={"Range": "bytes=0-1"})
                r.close()

            size = r.headers.get("content-length")
            if r.status_code == 206:
                size = r.headers.get("Content-Range", "/").rsplit("/", 1)[-1]
            try:
                size = int(size)
            except (TypeError, ValueError):
                size = None

            return {"group": group, "label": label, "url": url, "mode": mode,
                    "status": r.status_code, "size": size,
                    "ranges": r.headers.get("accept-ranges", "").lower(),
                    "auth": "www-authenticate" in r.headers}
        except requests.exceptions.RequestException as exc:
            last = f"{type(exc).__name__}"

    return {"group": group, "label": label, "url": url, "mode": mode,
            "status": last, "size": None, "ranges": "", "auth": False}


def run(rows, threads=THREADS):
    with futures.ThreadPoolExecutor(threads) as pool:
        return list(pool.map(check, rows))


def is_live(res):
    return res["status"] in (200, 206)


def human(size):
    if size is None:
        return "-"
    if size >= 1e9:
        return f"{size / 1e9:.2f} GB"
    if size >= 1e6:
        return f"{size / 1e6:.1f} MB"
    return f"{size / 1e3:.0f} kB"


# ─────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────
def report(results, quiet=False):
    by_group = collections.OrderedDict()
    for res in results:
        by_group.setdefault(res["group"], []).append(res)

    for group, rows in by_group.items():
        live = [r for r in rows if is_live(r)]
        fetch = sum(r["size"] or 0 for r in live if r["mode"] == FETCH)
        window = sum(r["size"] or 0 for r in live if r["mode"] == WINDOW)

        totals = []
        if fetch:
            totals.append(f"{human(fetch)} to fetch")
        if window:
            totals.append(f"{human(window)} of remote files read through a "
                          "window")
        print(f"\n{group.upper()}  --  {len(live)}/{len(rows)} live"
              + (", " + ", ".join(totals) if totals else ""))

        if quiet:
            continue
        for r in rows:
            flag = "ok " if is_live(r) else "FAIL"
            ranges = "R" if r["ranges"].startswith("bytes") else " "
            mode = "win" if r["mode"] == WINDOW else "   "
            auth = "  AUTH-REQUIRED" if r["auth"] else ""
            print(f"  {flag} {str(r['status']):>4} {human(r['size']):>9} "
                  f"{ranges} {mode}  {r['label']}{auth}")
            if not is_live(r):
                print(f"         {r['url']}")

    return by_group


def main():
    parser = argparse.ArgumentParser(
        description="HEAD-check every data source before committing to a "
                    "download.")
    parser.add_argument("--iso", default=None,
                        help="also check the per-country 30 arcsec files")
    parser.add_argument("--group", action="append", default=None,
                        help="only check these groups (repeatable)")
    parser.add_argument("--quiet", action="store_true",
                        help="per-group totals only")
    args = parser.parse_args()

    rows = country_independent()
    if args.iso:
        rows = country_specific(args.iso) + rows
    if args.group:
        wanted = set(args.group)
        rows = [r for r in rows if r[0] in wanted]

    print("=" * 70)
    print("SOURCE URL LIVENESS CHECK")
    print("=" * 70)
    print(f"checking {len(rows)} URLs"
          + (f" for {args.iso.upper()}" if args.iso else "")
          + f", {THREADS} at a time")
    print("  legend: R = server advertises byte ranges;  "
          "win = read through a window, never downloaded whole")

    results = run(rows)
    report(results, quiet=args.quiet)

    live = [r for r in results if is_live(r)]
    dead = [r for r in results if not is_live(r)]
    walled = [r for r in results if r["auth"]]
    fetch = sum(r["size"] or 0 for r in live if r["mode"] == FETCH)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  live            {len(live)}/{len(results)}")
    print(f"  bytes to fetch  {human(fetch)}  "
          "(windowed sources excluded -- only their country window travels)")
    print(f"  credentialled   {len(walled)}  "
          "(any non-zero number is a problem for this project)")
    if dead:
        print(f"  NOT LIVE        {len(dead)}:")
        for r in dead:
            print(f"      [{r['status']}] {r['label']}")
            print(f"              {r['url']}")
    else:
        print("  NOT LIVE        0 -- every source is reachable without "
              "credentials")

    return 1 if dead else 0


if __name__ == "__main__":
    raise SystemExit(main())
