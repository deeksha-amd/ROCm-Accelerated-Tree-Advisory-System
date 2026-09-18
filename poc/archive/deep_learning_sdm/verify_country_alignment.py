"""
Prove that a country-scoped data tree is internally consistent and faithful to
the global blocks it came from.

WHAT IT CHECKS
    1. ONE GRID PER RESOLUTION. Every layer at a given pixel size must share a
       bit-identical CRS, transform, width and height. This is the property the
       whole project rests on -- it is what lets a predictor stack be indexed by
       (row, col) instead of by coordinate lookup -- so it is checked first and
       reported per resolution group.

    2. NESTING. Where both a 10-arcmin and a 30-arcsec block exist, the fine
       grid must be an exact integer refinement of the coarse one, covering
       identical ground. If that holds, either can be block-reduced to the
       other and the two are directly comparable.

    3. FAITHFULNESS. A clip must be a window, not a warp. For any layer tagged
       CLIPPED_FROM by country_clip.py, the corresponding window is re-read
       from the global source and compared cell by cell. Anything other than
       an exact match means something resampled, and a resampled predictor
       silently shifts every value by up to half a cell.

    4. COVERAGE. What share of the country polygon actually carries data in
       each layer, so a block that is quietly empty over half the country
       cannot reach a model unnoticed.

    Checks 1 and 4 run on any directory tree. Checks 2 and 3 need the metadata
    country_clip.py writes, and are skipped with a note where it is absent.

USAGE
    python3 verify_country_alignment.py IND
    python3 verify_country_alignment.py IND --root country_data/IND
    python3 verify_country_alignment.py IND --no-faithfulness   # skip re-reads
"""

import argparse
import json
import os
import sys

import numpy as np
import rasterio
from affine import Affine

from country_clip import RES_10M, RES_30S, country_grid, country_mask

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
DEFAULT_ROOT = "country_data"
SEP = "=" * 70

# Directories excluded from the one-grid check, and why.
#
# satellite_download/ is NOT a predictor block. It holds NDVI and land cover as
# output filters and diagnostics, deliberately left at each source's native
# resolution so they cannot be stacked with the predictors by accident. Pulling
# them into this check would report a grid mismatch that is the intended
# design, and worse, "fixing" that failure would mean putting circular
# vegetation layers onto the predictor grid.
#
# provenance/ holds the soil source flag, which records whether each pixel came
# from OpenLandMap or the SoilGrids backfill. It is on the predictor grid, but
# it is metadata about the data rather than an environmental variable, so it is
# kept out of the block that feeds the model.
EXCLUDE_DIRS = ("satellite_download", "provenance")
SUB = "-" * 70

# Pixel sizes we expect to meet, largest first, for grouping the report.
KNOWN_RES = [(RES_10M, "10 arcmin (~18.5 km)"), (RES_30S, "30 arcsec (~1 km)")]
RES_TOL = 1e-9

# A layer covering less of the country than this is called out.
LOW_COVERAGE_WARN = 0.90


# ─────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────
def find_layers(root):
    """Every predictor GeoTIFF under `root`, excluding EXCLUDE_DIRS."""
    out, excluded = [], {}
    for dirpath, dirnames, files in os.walk(root):
        skipped = [d for d in dirnames if d in EXCLUDE_DIRS]
        for name in skipped:
            dirnames.remove(name)
            count = sum(len([f for f in fs if f.endswith((".tif", ".tiff"))])
                        for _, _, fs in os.walk(os.path.join(dirpath, name)))
            excluded[name] = count
        for name in sorted(files):
            if name.endswith((".tif", ".tiff")):
                out.append(os.path.join(dirpath, name))
    return sorted(out), excluded


def describe_res(res):
    for value, label in KNOWN_RES:
        if abs(res - value) < RES_TOL:
            return label
    return f"{res:.8f} deg ({res * 3600:.1f} arcsec)"


def nodata_mask(arr, nodata):
    """Which cells carry no data, under this project's conventions.

    Three things all mean "absent" here: NaN, the file's declared nodata, and
    anything below -1e30, which is the sentinel test DATA_OVERVIEW.md documents
    for the WorldClim-derived blocks (their nodata is -3.4e38).
    """
    if np.issubdtype(arr.dtype, np.floating):
        out = ~np.isfinite(arr) | (arr <= -1e30)
        if nodata is not None and np.isfinite(nodata):
            out |= arr == nodata
        return out
    return (np.zeros(arr.shape, bool) if nodata is None
            else arr == nodata)


def layer_info(path):
    with rasterio.open(path) as s:
        return {
            "path": path,
            "crs": s.crs.to_string() if s.crs else "NONE",
            "transform": tuple(s.transform)[:6],
            "width": s.width, "height": s.height,
            "res": s.transform.a, "count": s.count,
            "dtype": s.dtypes[0], "nodata": s.nodata,
            "tags": s.tags(),
        }


# ─────────────────────────────────────────────────────
# CHECK 1: one grid per resolution
# ─────────────────────────────────────────────────────
def check_one_grid(infos):
    """Group layers by pixel size and require an identical grid within a group."""
    print("\n" + SEP)
    print("CHECK 1  ONE GRID PER RESOLUTION")
    print(SEP)

    groups = {}
    for info in infos:
        key = round(info["res"], 12)
        groups.setdefault(key, []).append(info)

    all_ok = True
    grids = {}
    for res in sorted(groups, reverse=True):
        members = groups[res]
        ref = members[0]
        signature = (ref["crs"], ref["transform"], ref["width"], ref["height"])
        mismatched = [m for m in members
                      if (m["crs"], m["transform"], m["width"], m["height"])
                      != signature]

        print(f"\n{describe_res(res)}  --  {len(members)} layers")
        print(f"  crs        {ref['crs']}")
        print(f"  size       {ref['width']} x {ref['height']}")
        print(f"  origin     ({ref['transform'][2]:.6f}, "
              f"{ref['transform'][5]:.6f})")
        print(f"  pixel      {ref['transform'][0]:.10f} x "
              f"{-ref['transform'][4]:.10f} deg")
        bands = sorted({m["count"] for m in members})
        dtypes = sorted({m["dtype"] for m in members})
        print(f"  bands      {bands}")
        print(f"  dtypes     {dtypes}")

        if mismatched:
            all_ok = False
            print(f"  FAIL       {len(mismatched)} layers differ from the group:")
            for m in mismatched[:10]:
                print(f"               {m['path']}")
                print(f"               {m['width']}x{m['height']} "
                      f"origin ({m['transform'][2]:.6f}, "
                      f"{m['transform'][5]:.6f}) crs {m['crs']}")
        else:
            print(f"  PASS       all {len(members)} layers share one grid "
                  "exactly")
            grids[res] = ref

    return all_ok, grids


# ─────────────────────────────────────────────────────
# CHECK 2: the fine grid nests inside the coarse one
# ─────────────────────────────────────────────────────
def check_nesting(grids):
    """The 30-arcsec grid must be an exact integer refinement of the 10-arcmin one."""
    print("\n" + SEP)
    print("CHECK 2  FINE GRID NESTS INSIDE COARSE GRID")
    print(SEP)

    coarse = next((g for r, g in grids.items() if abs(r - RES_10M) < RES_TOL),
                  None)
    fine = next((g for r, g in grids.items() if abs(r - RES_30S) < RES_TOL),
                None)
    if not (coarse and fine):
        have = ", ".join(describe_res(r) for r in sorted(grids, reverse=True))
        print(f"  SKIP       only one resolution present ({have or 'none'})")
        return True

    ratio = coarse["res"] / fine["res"]
    ratio_ok = abs(ratio - round(ratio)) < 1e-6
    n = int(round(ratio))

    ct, ft = coarse["transform"], fine["transform"]
    origin_ok = abs(ct[2] - ft[2]) < 1e-9 and abs(ct[5] - ft[5]) < 1e-9
    size_ok = (fine["width"] == coarse["width"] * n
               and fine["height"] == coarse["height"] * n)

    print(f"  ratio      {ratio:.6f}  -> {n} x {n} fine cells per coarse cell "
          f"[{'ok' if ratio_ok else 'NOT AN INTEGER'}]")
    print(f"  origin     coarse ({ct[2]:.6f}, {ct[5]:.6f})  "
          f"fine ({ft[2]:.6f}, {ft[5]:.6f})  "
          f"[{'identical' if origin_ok else 'DIFFERENT'}]")
    print(f"  size       coarse {coarse['width']}x{coarse['height']}  "
          f"fine {fine['width']}x{fine['height']}  "
          f"expected {coarse['width'] * n}x{coarse['height'] * n}  "
          f"[{'ok' if size_ok else 'MISMATCH'}]")

    ok = ratio_ok and origin_ok and size_ok
    print(f"  {'PASS' if ok else 'FAIL'}       "
          + ("the two blocks cover identical ground; either can be "
             "block-reduced to the other"
             if ok else "the blocks do NOT cover identical ground"))
    return ok


# ─────────────────────────────────────────────────────
# CHECK 3: clips are windows, not warps
# ─────────────────────────────────────────────────────
def find_source(name, search_dirs):
    """Resolve a CLIPPED_FROM tag to a file on disk.

    country_clip.py records a relative path, which is used directly. The
    basename search is only a fallback for older files, and it is deliberately
    refused when the name is ambiguous: wc2.1_10m_bio_1.tif exists in
    climate_current/, climate_recent/ and climate_2000_2015/ alike, and
    silently picking the first would compare a clip against the wrong block.
    """
    if os.path.isfile(name):
        return name

    base = os.path.basename(name)
    hits = []
    for d in search_dirs:
        for dirpath, _, files in os.walk(d):
            if base in files:
                hits.append(os.path.join(dirpath, base))
    if len(hits) == 1:
        return hits[0]
    return None


def check_faithfulness(infos, search_dirs):
    """Re-read each clip's window from its global source and compare exactly."""
    print("\n" + SEP)
    print("CHECK 3  CLIPS ARE WINDOWS, NOT WARPS")
    print(SEP)

    clipped = [i for i in infos if i["tags"].get("CLIPPED_FROM")]
    if not clipped:
        print("  SKIP       no layer carries a CLIPPED_FROM tag; nothing here "
              "was produced by country_clip.py")
        return True

    print(f"  comparing {len(clipped)} clipped layers against their global "
          "sources, cell by cell\n")

    exact, missing, failed = 0, [], []
    for info in clipped:
        name = info["tags"]["CLIPPED_FROM"]
        src_path = find_source(name, search_dirs)
        if not src_path:
            missing.append((info["path"], name))
            continue

        with rasterio.open(src_path) as src, rasterio.open(info["path"]) as dst:
            ta, tb = src.transform, dst.transform
            col = int(round((tb.c - ta.c) / ta.a))
            row = int(round((tb.f - ta.f) / ta.e))

            # Only the overlap is comparable: a clip may be padded with nodata
            # where the country window overhangs the global source.
            c0, r0 = max(col, 0), max(row, 0)
            c1 = min(col + dst.width, src.width)
            r1 = min(row + dst.height, src.height)
            if c1 <= c0 or r1 <= r0:
                failed.append((info["path"], "window does not overlap source"))
                continue

            win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
            a = src.read(1, window=win)
            b = dst.read(1, window=rasterio.windows.Window(
                c0 - col, r0 - row, c1 - c0, r1 - r0))

            a_nd = nodata_mask(a, src.nodata)
            b_nd = nodata_mask(b, dst.nodata)
            masked = info["tags"].get("CLIP_MASKED") == "True"
            both = ~a_nd & ~b_nd
            same_values = bool(np.array_equal(a[both].astype("float64"),
                                              b[both].astype("float64")))
            # Masking may add nodata, but a clip must never invent DATA where
            # the source had none.
            invented = bool((a_nd & ~b_nd).any())
            lost = bool((~a_nd & b_nd).any())

            if same_values and not invented and (not lost or masked):
                exact += 1
            else:
                reason = []
                if not same_values:
                    diff = np.abs(a[both].astype("float64")
                                  - b[both].astype("float64"))
                    reason.append(f"values differ, max |diff| {diff.max():.6g}")
                if invented:
                    reason.append("data where the source had nodata")
                if lost and not masked:
                    reason.append("nodata where the source had data, "
                                  "but the layer is not tagged as masked")
                failed.append((info["path"], "; ".join(reason)))

    print(f"  exact      {exact} of {len(clipped)} layers are bit-identical "
          "to their source window")
    if missing:
        print(f"  no source  {len(missing)} layers name a source that is not "
              "on disk (not a failure):")
        for path, name in missing[:5]:
            print(f"               {os.path.basename(path)} <- {name}")
    if failed:
        print(f"  FAIL       {len(failed)} layers do not match their source:")
        for path, reason in failed[:10]:
            print(f"               {path}: {reason}")
        return False

    print("  PASS       every clip is a pure window of its global source")
    return True


# ─────────────────────────────────────────────────────
# CHECK 4: coverage inside the country
# ─────────────────────────────────────────────────────
def polygon_grid(iso, res, root, margin_deg, area_frac, max_gap_deg):
    """The country polygon on the grid the layers are actually written on.

    country_grid.json is the authoritative extent, written by
    country_climate_rasters.py from WorldClim's own per-country footprint. It
    is preferred over re-deriving the box here, because re-deriving it with
    different margin settings would burn the polygon at the wrong offset and
    make coverage look catastrophic for no reason.
    """
    grid = country_grid(iso, res_deg=res, margin_deg=margin_deg,
                        area_frac=area_frac, max_gap_deg=max_gap_deg)

    path = os.path.join(root, "country_grid.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        scale = saved["res_deg"] / res
        if abs(scale - round(scale)) < 1e-6:
            n = int(round(scale))
            west, south, east, north = saved["bounds"]
            grid.update(transform=Affine(res, 0.0, west, 0.0, -res, north),
                        width=saved["width"] * n, height=saved["height"] * n,
                        bounds=(west, south, east, north), res_deg=res)
    return grid


def check_coverage(infos, iso, root, margin_deg, area_frac, max_gap_deg):
    """How much of the country polygon each layer actually populates."""
    print("\n" + SEP)
    print("CHECK 4  DATA COVERAGE INSIDE THE COUNTRY POLYGON")
    print(SEP)

    masks = {}
    rows, thin = [], []
    for info in infos:
        res = round(info["res"], 12)
        if res not in masks:
            grid = polygon_grid(iso, info["res"], root, margin_deg, area_frac,
                                max_gap_deg)
            if (grid["width"], grid["height"]) != (info["width"],
                                                   info["height"]):
                print(f"  note       {describe_res(res)} layers are "
                      f"{info['width']}x{info['height']} but the country "
                      f"grid resolves to {grid['width']}x{grid['height']}; "
                      "coverage for this group is skipped rather than "
                      "measured against a misplaced polygon")
                masks[res] = None
            else:
                masks[res] = country_mask(grid).astype(bool)
        inside = masks[res]
        if inside is None:
            continue

        with rasterio.open(info["path"]) as s:
            band = s.read(1)
            nd = s.nodata
        valid = ~nodata_mask(band, nd)
        share = float((valid & inside).sum()) / max(int(inside.sum()), 1)
        rows.append((info["path"], share))
        if share < LOW_COVERAGE_WARN:
            thin.append((info["path"], share))

    if not rows:
        print("  SKIP       no layer could be matched to a country polygon")
        return True

    shares = np.array([r[1] for r in rows])
    print(f"  measured   {len(rows)} layers")
    print(f"  coverage   min {shares.min() * 100:.1f}%   "
          f"median {np.median(shares) * 100:.1f}%   "
          f"max {shares.max() * 100:.1f}%  of in-country cells")

    if thin:
        print(f"\n  {len(thin)} layers below {LOW_COVERAGE_WARN * 100:.0f}% "
              "coverage (not necessarily wrong -- soil has no ocean, soil "
              "moisture has canopy holes -- but worth knowing):")
        for path, share in sorted(thin, key=lambda t: t[1])[:15]:
            print(f"      {share * 100:5.1f}%  {path}")
    else:
        print(f"  PASS       every layer covers at least "
              f"{LOW_COVERAGE_WARN * 100:.0f}% of the country")
    return True


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verify a country-scoped data tree shares one grid and is "
                    "faithful to the global blocks.")
    parser.add_argument("iso", help="3-letter country code")
    parser.add_argument("--root", default=None,
                        help=f"tree to check (default {DEFAULT_ROOT}/<ISO>)")
    parser.add_argument("--source-dir", action="append", default=None,
                        help="where to look for global sources (repeatable)")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--area-frac", type=float, default=1.0)
    parser.add_argument("--max-gap", type=float, default=10.0)
    parser.add_argument("--no-faithfulness", action="store_true",
                        help="skip the source re-read comparison")
    args = parser.parse_args()

    root = args.root or os.path.join(DEFAULT_ROOT, args.iso.upper())
    search_dirs = args.source_dir or ["climate_current", "climate_recent",
                                      "climate_2000_2015",
                                      "climate_future_2050", "soil_data",
                                      "topography", "satellite"]

    print(SEP)
    print("COUNTRY DATA ALIGNMENT VERIFICATION")
    print(SEP)
    print(f"country    {args.iso.upper()}")
    print(f"root       {root}/")

    if not os.path.isdir(root):
        print(f"\nERROR: {root} does not exist. Run the country acquisition "
              "scripts first.")
        sys.exit(2)

    paths, excluded = find_layers(root)
    if not paths:
        print(f"\nERROR: no GeoTIFFs under {root}/")
        sys.exit(2)

    print(f"layers     {len(paths)} predictor layers")
    for name, count in sorted(excluded.items()):
        print(f"excluded   {name}/ ({count} files) -- not a predictor block; "
              "held at native\n           resolution on purpose so it cannot "
              "join the predictor stack")
    infos = [layer_info(p) for p in paths]

    results = {}
    results["one grid per resolution"], grids = check_one_grid(infos)
    results["fine nests in coarse"] = check_nesting(grids)
    if args.no_faithfulness:
        print("\n" + SEP)
        print("CHECK 3  CLIPS ARE WINDOWS, NOT WARPS")
        print(SEP)
        print("  SKIP       --no-faithfulness")
    else:
        results["clips are windows"] = check_faithfulness(infos, search_dirs)
    results["coverage"] = check_coverage(infos, args.iso.upper(), root,
                                         args.margin, args.area_frac,
                                         args.max_gap)

    print("\n" + SEP)
    print("SUMMARY")
    print(SEP)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    if not all(results.values()):
        print("\nAt least one check FAILED. Do not train on this tree until "
              "it is resolved.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
