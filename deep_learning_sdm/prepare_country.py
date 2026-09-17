"""
One entry point that prepares a whole country: give it an ISO code and it
produces the same tree as country_data/FRA/ and species_occurrences/FR/.

WHY THIS EXISTS
    Every acquisition script in this project was already country-parameterised
    -- country_clip.py, country_climate_rasters.py, country_soil_rasters.py,
    country_topography_rasters.py and country_satellite_download.py all take an
    ISO code, and download_species.py and friends all take --country. What did
    not exist was the ORDER, the EXTENT DECISIONS and the two hand edits that
    turned those pieces into country_data/FRA/. France was assembled by hand
    and the recipe lived in a shell history. This file is that recipe.

    The output has to be structurally identical to France's, not merely
    equivalent, because deep_sdm_training.py resolves predictor layers by
    explicit filename per profile. A renamed layer is not an error there; it is
    silently dropped. So the stages below reproduce France's filenames exactly,
    including the two post-processing moves nobody would guess:

        soil_30s/source_flag_30s.tif  ->  provenance/source_flag_30s.tif
        aligned_10m/source_flag_10m.tif   deleted

    Both are soil PROVENANCE -- which source won in each cell -- not soil
    properties. Left in place they would be counted as predictors.

THE EXTENT DECISION IS THE WHOLE PROBLEM
    Everything else generalises for free. The extent does not, and it cannot be
    derived, because the question "which ground is this country for modelling
    purposes?" is not a question about polygons.

    Two mechanisms get it wrong in opposite directions:

    WorldClim's per-country 30 arcsec file, which pins the grid, covers a
    country's whole SOVEREIGN span. For USA that is -179.5 to 180 E and 18.5 to
    73 N: 43,140 x 6,540 = 282 million cells reaching from Guam across the
    antimeridian to Maine, overwhelmingly ocean. For ESP it reaches down to the
    Canaries at 27.5 N.

    country_clip.py's distance filter, which drops outlying possessions, is
    tuned to keep real archipelagos whole and therefore cannot cut either of
    these: Alaska's bounding box comes within 5.5 deg of the lower 48's, and
    the Canaries within 7.1 deg of mainland Spain's. Both are inside the 10 deg
    threshold that keeps Indonesia and the Philippines intact. The
    country_clip.py docstring already says this, and offers --bbox as the
    escape hatch. PROFILES below is where that hatch is documented per country.

    Each profile therefore carries an explicit bbox and the reason for it.
    A country with no profile falls back to WorldClim's own extent, which is
    right for any compact country -- it is what France uses.

STAGE ORDER, AND WHY grid COMES FIRST
    country_soil_rasters.py and country_topography_rasters.py both adopt
    country_data/<ISO>/country_grid.json, and country_climate_rasters.py writes
    it -- at the END of a run that begins with a multi-gigabyte download. Taken
    literally that serialises the whole pipeline behind that download.

    So the `grid` stage pins country_grid.json up front by reading only the
    HEADER of WorldClim's file over HTTP, which costs one range request and no
    download at all. After it, every other stage can run in any order and in
    parallel. --jobs does exactly that.

    The grid the `grid` stage writes and the grid country_climate_rasters.py
    derives from the downloaded file are the same object computed the same way
    from the same inputs, so climate overwrites it with an identical file.

RATE LIMITS
    The raster sources are plain file servers and take parallel connections
    happily; `worldclim` opens one per variable, which measured ~2.6x a single
    stream. GBIF is different: it is a shared community service that slows
    under sustained querying, so the species stages stay single-threaded and
    checkpointed, exactly as download_species.py already has them, and are the
    last thing this file runs.

USAGE
    python3 prepare_country.py --list                  # profiles and extents
    python3 prepare_country.py USA --plan              # what it would do
    python3 prepare_country.py USA                     # everything, resumable
    python3 prepare_country.py ESP --jobs 4            # raster blocks parallel
    python3 prepare_country.py USA --stage terrain --stage verify
    python3 prepare_country.py USA --skip species      # rasters only
    python3 prepare_country.py PRT --bbox=-9.6,36.9,-6.1,42.2   # ad hoc country

Every stage is resumable: a stage whose output already exists is skipped
unless --force. Nothing here writes to country_data/FRA/, species_occurrences/
FR/, the France sampling_effort/ outputs, or any global block.
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time

import rasterio
from tqdm import tqdm

import country_climate_rasters as ccr
from country_clip import RES_10M, RES_30S, country_grid, snap_bounds

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
# This file lives in deep_learning_sdm/; the data tree it writes lives at the
# repo root. Stage scripts are therefore located under HERE and run with their
# working directory set to DATA_ROOT, which is what every data path resolves
# against.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(HERE)

OUT_ROOT = "country_data"
LOG_DIR = "logs"
STATUS_FILE = "STATUS_COUNTRIES.md"
SEP = "=" * 70

# SoilGrids rung. France was built at 1000m, not the 250m default, and the
# country blocks have to be comparable with each other before they are
# comparable with anything else.
SG_RESOLUTION = "1000m"

# NDVI years. France holds 2021 and 2022; the source ends in 2022.
NDVI_YEARS = ("2021", "2022")

# Grid the sampling-effort surface and background points are built on.
# "30s" is the country's own 1 km grid, the same one the predictors use.
# "10m" is the global 18.5 km template, which is what France's first run used
# and what the fra_10m training profile still reads.
EFFORT_RESOLUTION = "30s"

# Exponent on the effort weight, per country. 1.0 is strict target-group
# sampling and is right wherever recording effort and tree records are
# concentrated to a similar degree. It is NOT universal: measured against
# presence cells on 1 km built-up fraction, strict weighting overshoots badly
# in the USA, where GBIF plant recording is far more city-concentrated than
# tree recording is (the top 1% of cells hold 61.3% of all plant records,
# against 54.3% in France and 35.3% in Spain).
#
#   country   power 0.50   power 0.75   power 1.00     (gap, presence - bg)
#   FR            +3.48        +2.77        +2.47
#   ES            +5.69        +5.25        +4.92
#   US            +4.13        -0.07        -4.37
#
# So the USA is drawn at 0.75, which lands its background on the presences'
# built-up fraction instead of 4.4 points past them. See the caveat in
# STATUS_COUNTRIES.md: that figure is calibrated on the 49 species downloaded
# so far, which are the best-recorded ones and skew urban, and it should be
# re-measured when the download finishes.
EFFORT_POWER = {"USA": 0.75}
DEFAULT_EFFORT_POWER = 1.0

# Satellite sub-stages not fetched. ESA CCI surface soil moisture is skipped
# because country_satellite_download.fetch_soil_moisture() calls
# satellite_rasters.download_file(..., desc=...) and that function takes no
# desc, so the stage raises TypeError and takes the manifest down with it.
# France hit the same bug and country_data/FRA/satellite_download/ has no soil
# moisture in it either, so skipping is what reproduces France rather than a
# silent loss. It is also the one layer here that is NOT a vegetation proxy,
# so it is the only one whose absence costs a possible future predictor --
# noted rather than papered over.
SATELLITE_SKIP = ("soilmoisture",)

# Soil provenance layers. Real outputs, but they record WHICH SOURCE won in
# each cell, not a soil property, so they are moved out of the predictor blocks
# rather than left to be picked up by name. See the module docstring.
PROVENANCE_MOVES = (("soil_30s/source_flag_30s.tif", "provenance/source_flag_30s.tif"),)
PROVENANCE_DROPS = ("aligned_10m/source_flag_10m.tif",)

# The 10 arcmin blocks, clipped straight out of the global ones as windows.
# Names are the global directory -> the country subdirectory France uses.
COARSE_BLOCKS = (("climate_recent", "climate_recent"),
                 ("climate_current", "climate_current"),
                 ("soil_data/aligned_10m", "aligned_10m"))


# ─────────────────────────────────────────────────────
# COUNTRY PROFILES  --  the extent decisions
# ─────────────────────────────────────────────────────
# bbox is (west, south, east, north) in degrees, or None to accept WorldClim's
# own per-country extent. `why` is not decoration: an extent decision that is
# not written down is a decision someone will silently reverse.
PROFILES = {
    "FRA": {
        "iso2": "FR",
        "bbox": (-5.5, 41.0, 10.0, 51.5),
        "why": "Metropolitan France including Corsica. WorldClim's FRA file is "
               "already metropolitan-only, so this bbox is a no-op that records "
               "the decision; the overseas departments are filed by GBIF under "
               "their own ISO codes and would otherwise put Amazonian and "
               "Indian Ocean climate in a temperate-Europe model.",
    },
    "USA": {
        "iso2": "US",
        "bbox": (-125.0, 24.0, -66.5, 49.5),
        "why": "The CONTIGUOUS LOWER 48 only, the same decision as metropolitan "
               "France. WorldClim's USA file spans -179.5 to 180 E and 18.5 to "
               "73 N -- Guam and the Aleutians across the antimeridian to Maine "
               "-- which is 282 M cells, 13x the lower 48, almost all of it "
               "ocean, and these grids do not wrap so the box would be "
               "meaningless anyway. Alaska, Hawaii, Puerto Rico and the Pacific "
               "territories are excluded: each is a separate climate with its "
               "own tree flora, and none of them shares a climate envelope with "
               "the lower 48, so nothing is lost by modelling them separately "
               "later. Edges are on the 0.5 deg lattice WorldClim snaps to and "
               "clear every extreme point of the lower 48 (Cape Alava -124.73, "
               "West Quoddy Head -66.95, Key West 24.54, Lake of the Woods "
               "49.38).",
    },
    "ESP": {
        "iso2": "ES",
        "bbox": (-9.5, 35.5, 4.5, 44.0),
        "why": "MAINLAND SPAIN PLUS THE BALEARICS, excluding the Canary "
               "Islands. The Canaries are Atlantic and subtropical, 1,000 km "
               "off Africa, with a laurisilva and Canary pine flora that shares "
               "almost nothing with the peninsula -- the same call as keeping "
               "Corsica but dropping French Guiana. WorldClim's ESP file "
               "reaches 27.5 N to include them; the south edge at 35.5 cuts "
               "them out. Ceuta sits at 35.89 N and so falls inside the window: "
               "unavoidable, since any rectangle containing Gibraltar contains "
               "northern Morocco, and harmless, because the country polygon "
               "still decides what counts as Spain.",
    },
}

DEFAULT_PROFILE = {"iso2": None, "bbox": None,
                   "why": "no profile; WorldClim's own per-country extent is "
                          "adopted, which is correct for a compact country"}


def profile_for(iso, bbox_override=None, iso2_override=None):
    """The extent decision for one country, with CLI overrides applied."""
    entry = dict(PROFILES.get(iso.upper(), DEFAULT_PROFILE))
    if bbox_override is not None:
        entry["bbox"] = bbox_override
        entry["why"] = "extent given on the command line"
    if iso2_override is not None:
        entry["iso2"] = iso2_override
    if entry["iso2"] is None:
        entry["iso2"] = iso.upper()[:2]
        entry["iso2_guessed"] = True
    return entry


# ─────────────────────────────────────────────────────
# STATUS LOG
# ─────────────────────────────────────────────────────
def status(line, path=STATUS_FILE):
    """Append one timestamped line to the run log. Append-only, on purpose."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"- `{stamp}` | {line}\n")


# ─────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────
def remote_worldclim_header(iso, var="tmin"):
    """Extent of WorldClim's per-country file, read over HTTP without fetching it.

    GDAL's /vsicurl driver range-requests the GeoTIFF header only, so this
    costs a few kB against a file that can be 1.25 GB. That is what lets the
    grid be pinned before the download rather than after it.
    """
    url = "/vsicurl/" + ccr.wc_iso_url(iso, var)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open(url) as src:
            return src.transform, src.width, src.height


def pin_grid(iso, bbox, force=False):
    """Write country_data/<ISO>/country_grid.json before anything is downloaded.

    Computed exactly as country_climate_rasters.grid_from_worldclim() does, so
    the climate stage later rewrites an identical file rather than a conflicting
    one. Returns the grid dict.
    """
    out_path = os.path.join(OUT_ROOT, iso, "country_grid.json")
    cached = os.path.join(ccr.CACHE_ROOT, iso, f"{iso}_wc2.1_30s_tmin.tif")

    if os.path.exists(cached) and os.path.getsize(cached) > 1000:
        grid = ccr.grid_from_worldclim(cached, iso, 0.0, 1.0, 10.0, bbox)
    else:
        transform, width, height = remote_worldclim_header(iso)
        west, north = transform.c, transform.f
        east = west + width * RES_30S
        south = north - height * RES_30S
        if bbox is not None:
            want = snap_bounds(bbox, margin_deg=0.0, snap_res=RES_10M)
            west, south = max(west, want[0]), max(south, want[1])
            east, north = min(east, want[2]), min(north, want[3])
            if east <= west or north <= south:
                raise SystemExit(f"--bbox does not intersect WorldClim's "
                                 f"extent for {iso}")
        grid = country_grid(iso, res_deg=RES_30S, margin_deg=0.0,
                            bbox=(west, south, east, north))
        grid.update(width=int(round((east - west) / RES_30S)),
                    height=int(round((north - south) / RES_30S)),
                    bounds=(west, south, east, north),
                    grid_source=f"{iso}_wc2.1_30s_tmin.tif header (remote)")

    for value in grid["bounds"]:
        if abs(value / RES_10M - round(value / RES_10M)) > 1e-6:
            raise SystemExit(f"{iso}: edge {value} is not on the 10 arcmin "
                             "lattice, so the two grids would not nest 20x20")

    if os.path.exists(out_path) and not force:
        return grid

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"country": iso, "res_deg": RES_30S, "crs": "EPSG:4326",
                   "transform": list(grid["transform"])[:6],
                   "width": grid["width"], "height": grid["height"],
                   "bounds": list(grid["bounds"]),
                   "margin_deg": 0.0, "area_frac": 1.0, "max_gap_deg": 10.0,
                   "bbox": list(bbox) if bbox else None,
                   "defined_by": grid["grid_source"]}, handle, indent=2)
    return grid


# ─────────────────────────────────────────────────────
# WORLDCLIM PREFETCH
# ─────────────────────────────────────────────────────
def prefetch_worldclim(iso, workers=6):
    """Fetch the per-country 30 arcsec files, one connection per variable.

    country_climate_rasters.py fetches them one after another, which is fine
    for a 25 MB country and not fine for the USA's 1.25 GB per variable. A
    single stream measured ~1.0 MB/s off geodata.ucdavis.edu and six parallel
    streams ~2.6 MB/s aggregate, so this is ~2.5x on the one stage that is
    purely network-bound. It writes into the same cache directory with the same
    names, so the climate stage then reports every file as "[have]".

    This is a file server, not GBIF. The species stages stay serial.
    """
    cache_dir = os.path.join(ccr.CACHE_ROOT, iso)
    os.makedirs(cache_dir, exist_ok=True)

    todo = []
    for var in ccr.WC_ISO_VARS:
        path = os.path.join(cache_dir, f"{iso}_wc2.1_30s_{var}.tif")
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            continue
        todo.append((var, path))

    if not todo:
        print("  all per-country files already cached")
        return True

    print(f"  fetching {len(todo)} variables on {min(workers, len(todo))} "
          "connections")
    ok = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, iso, var, path): var
                   for var, path in todo}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="WorldClim", unit="var"):
            var = futures[future]
            ok[var] = future.result()

    for var in sorted(ok):
        size = os.path.join(cache_dir, f"{iso}_wc2.1_30s_{var}.tif")
        note = (f"{os.path.getsize(size) / 1e6:.0f} MB" if ok[var]
                else "FAILED (optional unless tmin/tmax/prec)")
        print(f"    {var:<6} {note}")

    return all(ok[v] for v in ccr.WC_ISO_REQUIRED if v in ok)


def _fetch_one(iso, var, path):
    """One variable, streamed to <path>.part then renamed. Quiet; the caller bars.

    Resumes a partial .part with a Range request. At USA size -- 1.25 GB for a
    single variable, twenty minutes on a good connection -- restarting from
    zero after an interruption is the difference between a retry and an hour,
    and geodata.ucdavis.edu answers Range with 206 Partial Content.
    """
    import requests

    url = ccr.wc_iso_url(iso, var)
    tmp = path + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    try:
        headers = {"Range": f"bytes={have}-"} if have else {}
        response = requests.get(url, stream=True, timeout=300, headers=headers)

        if have and response.status_code == 416:
            # Already complete; the server has nothing past `have` to send.
            os.replace(tmp, path)
            return True
        if have and response.status_code != 206:
            # No range support on this response: start over rather than
            # concatenate a second copy of the file onto the first.
            have, headers = 0, {}
            response = requests.get(url, stream=True, timeout=300)
        if response.status_code not in (200, 206):
            return False

        with open(tmp, "ab" if have else "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)

        if os.path.getsize(tmp) < 1000:
            os.remove(tmp)
            return False
        os.replace(tmp, path)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────
# POST-PROCESSING  --  the two hand edits France needed
# ─────────────────────────────────────────────────────
def move_provenance(iso):
    """Get the soil source flags out of the predictor blocks.

    They are genuine outputs and worth keeping -- they say whether OpenLandMap
    or SoilGrids won in each cell -- but they describe the data, they are not
    a soil property, and a layer sitting in soil_30s/ is a layer a name-based
    loader will pick up. France holds the 30 arcsec flag under provenance/ and
    does not keep the 10 arcmin one at all, since that is just a window of
    soil_data/aligned_10m/source_flag_10m.tif, which is untouched globally.
    """
    root = os.path.join(OUT_ROOT, iso)
    moved = []
    for src_rel, dst_rel in PROVENANCE_MOVES:
        src, dst = os.path.join(root, src_rel), os.path.join(root, dst_rel)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved.append(f"{src_rel} -> {dst_rel}")
    for rel in PROVENANCE_DROPS:
        path = os.path.join(root, rel)
        if os.path.exists(path):
            os.remove(path)
            moved.append(f"{rel} removed (window of an untouched global layer)")
    for line in moved:
        print(f"  {line}")
    return moved


# ─────────────────────────────────────────────────────
# STAGES
# ─────────────────────────────────────────────────────
def stage_table(iso, iso2, bbox):
    """name -> (argv, output that proves it finished, one-line description).

    argv of None means the stage is a python function, dispatched in run_stage.
    """
    root = os.path.join(OUT_ROOT, iso)
    box = None if bbox is None else ",".join(f"{v:g}" for v in bbox)
    box_arg = [] if box is None else [f"--bbox={box}"]

    return {
        "grid": (None, f"{root}/country_grid.json",
                 "pin the 1 km grid from WorldClim's header, no download"),
        "worldclim": (None,
                      os.path.join(ccr.CACHE_ROOT, iso,
                                   f"{iso}_wc2.1_30s_prec.tif"),
                      "fetch the per-country 30 arcsec files in parallel"),
        "climate": (["country_climate_rasters.py", iso] + box_arg,
                    f"{root}/climate_30s/MANIFEST.json",
                    "29 climate layers at 1 km, delta-downscaled to 2015-2024"),
        "clip10m": (["country_clip.py", iso] + box_arg
                    + ["--margin", "0"]
                    + [a for src, _ in COARSE_BLOCKS for a in ("--block", src)],
                    f"{root}/climate_recent/coverage_mask.tif",
                    "window the 10 arcmin global blocks: 19+19+1 climate, "
                    "22 soil"),
        "soil": (["country_soil_rasters.py", iso,
                  "--sg-resolution", SG_RESOLUTION],
                 f"{root}/soil_30s/MANIFEST.json",
                 "22 soil layers at 1 km from native SoilGrids/OpenLandMap"),
        "terrain": (["country_topography_rasters.py", iso],
                    f"{root}/topography_30s/MANIFEST.json",
                    "12 terrain layers at 1 km from GEDTM30, plus 6 at "
                    "10 arcmin"),
        "satellite": (["country_satellite_download.py", iso] + box_arg
                      + ["--ndvi-years"] + list(NDVI_YEARS)
                      + ["--skip"] + list(SATELLITE_SKIP),
                      f"{root}/satellite_download/MANIFEST.json",
                      "NDVI and land cover at native res -- FILTERS ONLY, "
                      "never predictors"),
        "provenance": (None, f"{root}/provenance/source_flag_30s.tif",
                       "move the soil source flags out of the predictor blocks"),
        # The species stages are spelled out rather than delegated to
        # prepare_species_data.py --all, because two of them write CANONICAL
        # top-level files by default and a second country must not touch them:
        #
        #   clean_species.py     -> species_occurrences/gbif_trees_clean.csv
        #                           species_occurrences/gbif_trees_thinned.csv
        #   sampling_effort.py   -> sampling_effort/background_points.csv
        #
        # Those three are what sdm_features.py reads, so writing them from a
        # second country does not merely litter -- it silently repoints a
        # France training run at Spanish or American data. --output-prefix
        # suppresses the first two (and resolves to the same per-country names
        # anyway) and --no-canonical, added here, suppresses the third.
        "species_plan": (["download_species.py", "--plan", "--country", iso2],
                         f"species_occurrences/{iso2}/species_plan.json",
                         "resolve the curated list to taxonKeys and size the job"),
        "species_download": (["download_species.py", "--download",
                              "--country", iso2],
                             f"species_occurrences/{iso2}/"
                             f"gbif_trees_{iso2}_raw.csv",
                             "fetch occurrences by taxonKey, checkpointed per "
                             "species"),
        "species_clean": (["clean_species.py", "--country", iso2,
                           "--output-prefix",
                           f"species_occurrences/{iso2}/gbif_trees_{iso2}"],
                          f"species_occurrences/{iso2}/"
                          f"gbif_trees_{iso2}_thinned.csv",
                          "drop bad records and thin to the 18.5 km grid"),
        # --resolution 30s builds the effort surface on the country's OWN 1 km
        # grid rather than the global 18.5 km template. A target-group
        # correction computed 400x coarser than the predictors only cancels
        # bias between 18.5 km blocks; within one, the surveyors still walked
        # the roads, and the model can still learn "near a town" as habitat.
        "species_effort": (["sampling_effort.py", "--surface", "--background",
                            "--country", iso2, "--no-canonical",
                            "--resolution", EFFORT_RESOLUTION,
                            "--effort-power",
                            str(EFFORT_POWER.get(iso, DEFAULT_EFFORT_POWER))],
                           f"sampling_effort/background_{iso2}_"
                           f"{EFFORT_RESOLUTION}.csv",
                           "effort surface and target-group background points "
                           "on the country's 1 km grid"),
        "verify": (["verify_country_alignment.py", iso], None,
                   "prove one grid per resolution, 20x20 nesting, clips are "
                   "windows"),
    }


# Raster work that may run concurrently once `grid` has pinned the grid. Each
# entry is a CHAIN run in order by one worker; the chains run against each
# other. They touch disjoint output directories and disjoint remote sources.
#
# worldclim is chained ahead of climate rather than run before everything,
# because it is the only stage climate depends on and it is the slowest thing
# here for a large country -- 5.5 GB for the USA. Blocking soil, terrain and
# satellite behind it, as an ordinary serial ordering would, adds an hour of
# wall clock for no reason: none of them reads a WorldClim file. They read
# country_grid.json, which the `grid` stage has already pinned from an HTTP
# header.
PARALLEL_CHAINS = (("worldclim", "climate"),
                   ("clip10m",),
                   ("soil",),
                   ("terrain",),
                   ("satellite",))
PARALLEL_STAGES = tuple(s for chain in PARALLEL_CHAINS for s in chain)

SPECIES_STAGES = ("species_plan", "species_download", "species_clean",
                  "species_effort")

STAGE_ORDER = ["grid", "worldclim", "climate", "clip10m", "soil", "terrain",
               "satellite", "provenance", *SPECIES_STAGES, "verify"]


def run_stage(name, iso, iso2, bbox, force=False, quiet=False):
    """Run one stage. Returns True on success or a justified skip."""
    argv, output, description = stage_table(iso, iso2, bbox)[name]

    if output and os.path.exists(output) and not force:
        print(f"\n### {name}: SKIPPED -- {output} exists (--force to rebuild)")
        return True

    print(f"\n{SEP}\n### {name}: {description}\n{SEP}")
    start = time.monotonic()
    status(f"{iso} | {name} | started -- {description}")

    if argv is None:
        ok = _run_builtin(name, iso, bbox)
    else:
        ok = _run_script(name, iso, argv, quiet)

    elapsed = (time.monotonic() - start) / 60.0
    verdict = "done" if ok else "FAILED"
    print(f"\n### {name}: {verdict} in {elapsed:.1f} min")
    status(f"{iso} | {name} | {verdict} in {elapsed:.1f} min")

    if ok and output and not os.path.exists(output):
        print(f"### warning: expected {output} but it is not there")
    return ok


def _run_builtin(name, iso, bbox):
    if name == "grid":
        grid = pin_grid(iso, bbox)
        print(f"  {grid['width']} x {grid['height']} at 30 arcsec, bounds "
              + " ".join(f"{v:g}" for v in grid["bounds"]))
        print(f"  {grid['width'] // 20} x {grid['height'] // 20} at 10 arcmin")
        return True
    if name == "worldclim":
        return prefetch_worldclim(iso)
    if name == "provenance":
        move_provenance(iso)
        return True
    raise ValueError(f"no builtin for stage {name}")


def _run_script(name, iso, argv, quiet):
    """Run one acquisition script, tee'ing to logs/<iso>_<stage>.log.

    cwd is pinned to DATA_ROOT so the child resolves country_data/ and the rest
    of the data tree the same way this file does, whatever directory the user
    launched from. Pinning it also keeps a stray /tmp/enum.py from shadowing the
    stdlib, which has happened before. The script itself is addressed absolutely
    under HERE, because it is no longer in the working directory.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{iso.lower()}_{name}.log")
    command = [sys.executable, os.path.join(HERE, argv[0])] + list(argv[1:])
    print(f"$ {' '.join(command)}\n  log: {log_path}\n")

    with open(log_path, "w", encoding="utf-8") as log:
        if quiet:
            result = subprocess.run(command, cwd=DATA_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT)
        else:
            result = subprocess.run(command, cwd=DATA_ROOT,
                                    stderr=subprocess.STDOUT,
                                    stdout=subprocess.PIPE, text=True)
            log.write(result.stdout or "")
            tail = (result.stdout or "").splitlines()
            for line in tail[-25:]:
                print("  " + line)
    return result.returncode == 0


# ─────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────
def describe(iso, entry, bbox):
    lines = [f"country      {iso} / {entry['iso2']}"]
    if entry.get("iso2_guessed"):
        lines.append("             (ISO2 guessed from ISO3; set --iso2 if wrong)")
    if bbox is None:
        lines.append("extent       WorldClim's own per-country extent")
    else:
        lines.append("extent       " + " ".join(f"{v:g}" for v in bbox))
    lines.append("")
    for chunk in entry["why"].split(". "):
        if chunk.strip():
            lines.append(f"  {chunk.strip().rstrip('.')}.")
    return "\n".join(lines)


def summarise(iso):
    """Layer counts and disk size per block, for the end of a run."""
    root = os.path.join(OUT_ROOT, iso)
    if not os.path.isdir(root):
        return
    print(f"\n{SEP}\nWHAT IS ON DISK  --  {root}/\n{SEP}")
    total = 0
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        tifs, size = 0, 0
        for dirpath, _, files in os.walk(path):
            for fname in files:
                size += os.path.getsize(os.path.join(dirpath, fname))
                tifs += fname.endswith(".tif")
        total += size
        print(f"  {name:<20} {tifs:>4} tif  {size / 1e6:>9.1f} MB")
    print(f"  {'TOTAL':<20} {'':>8}  {total / 1e6:>9.1f} MB")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Prepare one country's full data tree, mirroring "
                    "country_data/FRA/ exactly.")
    parser.add_argument("iso", nargs="?", help="3-letter country code, e.g. USA")
    parser.add_argument("--iso2", default=None,
                        help="2-letter GBIF country code (default: from the "
                             "profile, else the first two letters)")
    parser.add_argument("--bbox", default=None, metavar="W,S,E,N",
                        help="override the profile's extent, in degrees")
    parser.add_argument("--stage", action="append", choices=STAGE_ORDER,
                        default=[], help="run specific stages, repeatable")
    parser.add_argument("--skip", action="append",
                        choices=STAGE_ORDER + ["species"],
                        default=[], help="skip specific stages, repeatable; "
                                         "'species' skips all four of them")
    parser.add_argument("--jobs", type=int, default=1,
                        help="run the independent raster blocks this many at a "
                             "time (climate, clip10m, soil, terrain, satellite)")
    parser.add_argument("--force", action="store_true",
                        help="rerun stages whose output already exists")
    parser.add_argument("--plan", action="store_true",
                        help="print the extent decision and the stages, then "
                             "stop")
    parser.add_argument("--list", action="store_true",
                        help="list the country profiles and exit")
    parser.add_argument("--quiet", action="store_true",
                        help="send stage output only to logs/")
    args = parser.parse_args()

    if args.list:
        for iso, entry in sorted(PROFILES.items()):
            box = entry["bbox"]
            print(f"\n{iso} / {entry['iso2']}   "
                  + ("WorldClim extent" if box is None
                     else " ".join(f"{v:g}" for v in box)))
            print("   " + entry["why"])
        return

    if not args.iso:
        parser.error("give a 3-letter country code, or --list")

    iso = args.iso.upper()
    bbox = None
    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))
        if len(bbox) != 4:
            parser.error("--bbox needs exactly four numbers: W,S,E,N")

    entry = profile_for(iso, bbox, args.iso2)
    bbox, iso2 = entry["bbox"], entry["iso2"]

    skip = set(args.skip)
    if "species" in skip:
        skip.update(SPECIES_STAGES)
    wanted = args.stage or [s for s in STAGE_ORDER if s not in skip]
    wanted = [s for s in STAGE_ORDER if s in wanted]

    print(SEP)
    print("PREPARE COUNTRY")
    print(SEP)
    print(describe(iso, entry, bbox))
    print(f"\nstages       {' '.join(wanted)}")
    print(f"parallel     {args.jobs} raster block(s) at a time")

    if args.plan:
        table = stage_table(iso, iso2, bbox)
        print()
        for name in wanted:
            argv, output, description = table[name]
            call = "built in" if argv is None else " ".join(argv)
            print(f"  {name:<11} {description}")
            print(f"  {'':<11} $ {call}")
        return

    status(f"{iso} | run start | stages: {' '.join(wanted)}; extent "
           + ("WorldClim" if bbox is None
              else " ".join(f"{v:g}" for v in bbox)))

    failed = []
    serial = [s for s in wanted if s not in PARALLEL_STAGES or args.jobs < 2]
    parallel = [s for s in wanted if s in PARALLEL_STAGES and args.jobs >= 2]

    for name in serial:
        if name in parallel:
            continue
        if not run_stage(name, iso, iso2, bbox, args.force, args.quiet):
            failed.append(name)
            # grid and worldclim are what everything else stands on.
            if name in ("grid", "worldclim"):
                break

    if parallel and not failed:
        chains = [[s for s in chain if s in parallel]
                  for chain in PARALLEL_CHAINS]
        chains = [c for c in chains if c]
        print(f"\n{SEP}\nRUNNING {len(chains)} RASTER CHAINS IN PARALLEL\n{SEP}")
        for chain in chains:
            print("  " + " -> ".join(chain))

        def run_chain(names):
            for name in names:
                if not run_stage(name, iso, iso2, bbox, args.force, True):
                    return names[:names.index(name) + 1][-1:]
            return []

        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            for bad in pool.map(run_chain, chains):
                failed.extend(bad)
        # provenance and verify have to follow the blocks they inspect.
        for name in ("provenance", "verify"):
            if name in wanted:
                run_stage(name, iso, iso2, bbox, args.force, args.quiet)

    summarise(iso)

    print(f"\n{SEP}")
    if failed:
        print(f"{len(failed)} stage(s) FAILED: {', '.join(sorted(set(failed)))}")
        print(f"see {LOG_DIR}/{iso.lower()}_<stage>.log")
        status(f"{iso} | run end | FAILED: {', '.join(sorted(set(failed)))}")
        sys.exit(1)
    print(f"{iso} prepared. Verify with: "
          f"python3 verify_country_alignment.py {iso}")
    status(f"{iso} | run end | all requested stages completed")


if __name__ == "__main__":
    main()
