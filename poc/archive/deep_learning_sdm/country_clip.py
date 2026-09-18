"""
Clip any of the project's global rasters to one country, on an EXACTLY aligned
subset of the source grid.

WHY THIS EXISTS
    The project is moving from a global model to a single-country proof of
    concept. Every global block (climate_*/, soil_data/aligned_10m/,
    topography/, satellite/) already sits on one 2160 x 1080 grid, and that
    property is what makes the blocks stackable. A country subset must not
    throw it away: if the clip resamples, or snaps to the country's bounding
    box instead of to the grid, the layers stop being cell-for-cell comparable
    and every later join becomes a nearest-neighbour lookup.

    So the rule here is: a clip is a WINDOW, never a warp. The output transform
    is the source transform translated by an INTEGER number of pixels, the
    pixel size is bit-identical, and the pixel values are copied untouched.
    verify_alignment() asserts exactly that, and the CLI runs it on every file.

THE TWO GRIDS
    Both grids used by this project are anchored at (-180, 90) and have a
    pixel size that divides a degree, so they nest exactly:

        10 arcmin  = 1/6 deg     2160 x 1080    the existing global blocks
        30 arcsec  = 1/120 deg   43200 x 21600  the country-scale target

    1/6 / (1/120) = 20, so one 10-arcmin cell is exactly 20 x 20 cells of the
    30-arcsec grid. country_grid() snaps every window to the COARSER grid by
    default, which means the fine country grid and the coarse country grid
    cover precisely the same ground and either can be derived from the other by
    a 20x block reduction. Nothing here interpolates between the two.

BOUNDARIES
    Natural Earth admin-0, public domain, no credentials, ~1-5 MB. Taken as
    GeoJSON from the natural-earth-vector repository rather than as a
    shapefile, because fiona/geopandas are not installed in this environment
    and GeoJSON needs nothing but the json module. Polygon masking then goes
    through rasterio.features.rasterize, which accepts GeoJSON geometry dicts
    directly.

    ISO_A3 in Natural Earth is "-99" for a number of territories (and, in
    some releases, for France and Norway, whose sovereign geometry is split).
    ISO_KEYS therefore tries several code fields in order of preference, and
    where several features answer to one code (SOV_A3 "BRA" matches both
    Brazil and the uninhabited Brazilian Island) the largest one wins.

OUTLYING TERRITORIES
    A bounding box is a bad summary of a country with remote possessions.
    Chile's box reaches Easter Island 3,500 km offshore, so only 4.5% of it is
    Chilean ground; France's spans French Guiana to Reunion. Downloading and
    training on that is mostly downloading and training on ocean.

    MAX_GAP_DEG drops parts more than that far from the main landmass cluster,
    which is the filter that works, because the problem is distance and not
    size. See select_parts() for why an area threshold alone cannot do it.

    Chaining through intermediate islands is deliberate -- it is what keeps
    Indonesia and the Philippines whole -- but it is not free: Portugal chains
    mainland to Madeira to the Azores, and there is no distance rule that
    both keeps Indonesia intact and cuts the Azores. So --bbox is the escape
    hatch, and describe_grid() always prints what share of the window is
    really inside the polygon and warns when it is low, so a pathological box
    cannot pass unnoticed. A country whose parts straddle the antimeridian
    (USA, Russia, Fiji, New Zealand) is flagged for the same reason: these
    grids do not wrap, so the box would otherwise silently span the Pacific.

USAGE
    python3 country_clip.py --list-like braz              # find the ISO code
    python3 country_clip.py IND --info                    # grid it would make
    python3 country_clip.py CHL --info                    # drops Easter Island
    python3 country_clip.py PRT --info --bbox -9.6,36.9,-6.1,42.2
    python3 country_clip.py IND --block climate_recent    # clip a whole block
    python3 country_clip.py IND --block soil_data/aligned_10m --mask
    python3 country_clip.py IND climate_recent/wc2.1_10m_bio_1.tif

    as a library:
        from country_clip import country_grid, clip_raster, country_mask
        grid = country_grid("IND", res_deg=1/120)
"""

import argparse
import json
import os
import sys

import numpy as np
import rasterio
import requests
from affine import Affine
from rasterio.features import rasterize
from rasterio.windows import Window
from tqdm import tqdm

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
CACHE_DIR = "country_src_cache"        # new dir; the existing blocks are read-only
BOUNDARY_DIR = os.path.join(CACHE_DIR, "natural_earth")

# Natural Earth admin-0, public domain. 10m is the detailed one; 50m is a fifth
# of the size and is plenty for a bounding box, so it is the fallback.
NE_VERSION = "v5.1.2"
NE_BASE = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
           f"/{NE_VERSION}/geojson")
NE_SCALES = {
    "10m": ("ne_10m_admin_0_countries.geojson", 4.7),    # MB
    "50m": ("ne_50m_admin_0_countries.geojson", 1.0),
}
NE_DEFAULT_SCALE = "10m"

# Property fields searched, in order, for a 3-letter country code. ADM0_A3 is
# the one Natural Earth always populates; ISO_A3 is the one people expect.
ISO_KEYS = ("ISO_A3", "ISO_A3_EH", "ADM0_A3", "SOV_A3", "GU_A3", "SU_A3")
NAME_KEYS = ("NAME_LONG", "NAME_EN", "NAME", "ADMIN", "SOVEREIGNT")

# Grid anchors. Both are exact divisors of a degree and share the origin, so
# the fine grid nests 20 x 20 inside the coarse one.
GRID_ORIGIN = (-180.0, 90.0)
RES_10M = 1.0 / 6.0            # 10 arcmin, the existing global blocks
RES_30S = 1.0 / 120.0          # 30 arcsec, the country-scale target
SNAP_RES = RES_10M             # windows snap to this grid, whatever res_deg is

# How much ground to keep outside the border. A species' climate envelope is
# fitted from occurrences that do not stop at the border, and a convolutional
# or patch-based model needs real data in the patch around an edge cell, not
# padding. 0.5 deg is ~55 km, comfortably wider than any patch we would use.
DEFAULT_MARGIN_DEG = 0.5

# Fraction of a country's area that must be kept when discarding outlying
# parts. 1.0 keeps every islet; 0.99 is enough to drop Easter Island from
# Chile. See the OUTLYING TERRITORIES note in the module docstring.
DEFAULT_AREA_FRAC = 1.0

# A polygon part further than this from the main cluster's growing bounding box
# is treated as an outlying possession and dropped. 10 deg (~1,100 km) is a
# no-op for every compact country and for real archipelagos, but it does cut
# Easter Island off Chile, the Azores off Portugal, French Guiana off France
# and Alaska + Hawaii off the USA. Set to None to keep everything.
DEFAULT_MAX_GAP_DEG = 10.0

# describe_grid() warns below this in-polygon share of the window.
SPARSE_WINDOW_WARN = 0.25


# ─────────────────────────────────────────────────────
# DOWNLOAD HELPERS
# ─────────────────────────────────────────────────────
def download_file(url, output_path, desc="Downloading"):
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
                  desc=desc[:44]) as pbar:
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                pbar.update(len(chunk))
    os.replace(tmp_path, output_path)
    return True


def ensure_boundaries(scale=NE_DEFAULT_SCALE):
    """Local path to the Natural Earth admin-0 GeoJSON, downloading if needed."""
    if scale not in NE_SCALES:
        raise ValueError(f"scale must be one of {list(NE_SCALES)}")
    fname, approx_mb = NE_SCALES[scale]
    path = os.path.join(BOUNDARY_DIR, fname)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    os.makedirs(BOUNDARY_DIR, exist_ok=True)
    print(f"fetching Natural Earth admin-0 {scale} (~{approx_mb:.1f} MB, "
          "public domain)")
    if not download_file(f"{NE_BASE}/{fname}", path, desc=fname):
        raise RuntimeError(f"could not download {fname}")
    return path


# ─────────────────────────────────────────────────────
# COUNTRY LOOKUP
# ─────────────────────────────────────────────────────
def _first(props, keys, default=""):
    for k in keys:
        v = props.get(k)
        if v not in (None, "", "-99", -99):
            return v
    return default


def _parts(geom):
    """A GeoJSON Polygon / MultiPolygon as a list of Polygon dicts."""
    if geom["type"] == "Polygon":
        return [geom]
    return [{"type": "Polygon", "coordinates": poly}
            for poly in geom["coordinates"]]


def _geometry_bbox(geom):
    """Bounding box of a GeoJSON Polygon / MultiPolygon, ignoring holes."""
    xs, ys = [], []
    rings = (geom["coordinates"] if geom["type"] == "Polygon"
             else [r for poly in geom["coordinates"] for r in poly])
    for ring in rings:
        arr = np.asarray(ring, dtype=float)
        xs.extend((arr[:, 0].min(), arr[:, 0].max()))
        ys.extend((arr[:, 1].min(), arr[:, 1].max()))
    return min(xs), min(ys), max(xs), max(ys)


def _ring_area(ring):
    """Shoelace area of one ring in deg^2, scaled by cos(mean latitude).

    Only ever used to RANK parts against each other, so a cheap cylindrical
    correction is enough -- it just has to stop a high-latitude islet from
    out-ranking a tropical mainland.
    """
    arr = np.asarray(ring, dtype=float)
    x, y = arr[:, 0], arr[:, 1]
    planar = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return planar * np.cos(np.deg2rad(np.clip(y.mean(), -89.0, 89.0)))


def _part_area(part):
    """Outer-ring area of a Polygon part, minus its holes."""
    rings = part["coordinates"]
    return _ring_area(rings[0]) - sum(_ring_area(r) for r in rings[1:])


def geometry_area(geom):
    """Total ranking area of a Polygon / MultiPolygon."""
    return sum(_part_area(p) for p in _parts(geom))


def _bbox_gap(a, b):
    """Shortest distance in degrees between two bounding boxes (0 if they meet)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(np.hypot(dx, dy))


def select_parts(geom, area_frac=DEFAULT_AREA_FRAC,
                 max_gap_deg=DEFAULT_MAX_GAP_DEG):
    """Discard outlying polygon parts. Returns (geometry, kept, dropped).

    Two filters, applied in that order:

    area_frac   keeps the largest parts summing to that share of the area.
                Cheap, but on its own it is the WRONG tool: Chile's mainland
                is 163 separate fjord polygons, so Easter Island is larger
                than most of them and survives any area threshold that keeps
                the mainland intact.

    max_gap_deg is the filter that actually works, because the problem is
                spatial, not one of size. Starting from the largest part, a
                part joins the cluster if it comes within max_gap_deg of the
                cluster's current bounding box, and the box then grows to
                include it. Chaining like this keeps a genuine archipelago
                together (island-to-island gaps in Indonesia, the Philippines
                or Japan are a degree or two) while cutting a possession
                thousands of km offshore.

    Both default to no-ops on a compact country. Whatever is dropped is
    reported by describe_grid(), never silently.
    """
    parts = _parts(geom)
    if len(parts) == 1:
        return geom, 1, 0

    areas = np.array([_part_area(p) for p in parts])
    order = list(np.argsort(areas)[::-1])

    if area_frac < 1.0:
        cumulative = np.cumsum(areas[order])
        n_keep = min(int(np.searchsorted(cumulative,
                                         area_frac * areas.sum()) + 1),
                     len(order))
        order = order[:n_keep]

    if max_gap_deg is not None:
        boxes = {i: _geometry_bbox(parts[i]) for i in order}
        cluster, pending = [order[0]], list(order[1:])
        box = list(boxes[order[0]])
        grew = True
        while grew:
            grew = False
            for i in list(pending):
                if _bbox_gap(box, boxes[i]) <= max_gap_deg:
                    cluster.append(i)
                    pending.remove(i)
                    b = boxes[i]
                    box = [min(box[0], b[0]), min(box[1], b[1]),
                           max(box[2], b[2]), max(box[3], b[3])]
                    grew = True
        order = cluster

    if len(order) == len(parts):
        return geom, len(parts), 0
    return ({"type": "MultiPolygon",
             "coordinates": [parts[i]["coordinates"] for i in order]},
            len(order), len(parts) - len(order))


def load_country(iso_a3, scale=NE_DEFAULT_SCALE):
    """Look one country up by 3-letter code.

    Several features can answer to one code once the sovereignty fields are
    searched, so candidates are ranked by how authoritative the matching field
    is and then by area. That is what keeps "BRA" resolving to Brazil rather
    than to Brazilian Island, whose SOV_A3 is also BRA.

    Returns {"iso", "name", "geometry", "bbox"}. bbox is the raw geometry
    extent in degrees, before any margin, part selection or grid snapping.
    """
    iso = iso_a3.upper()
    with open(ensure_boundaries(scale), encoding="utf-8") as f:
        fc = json.load(f)

    candidates = []
    for feat in fc["features"]:
        props = feat["properties"]
        rank = next((i for i, k in enumerate(ISO_KEYS)
                     if str(props.get(k, "")).upper() == iso), None)
        if rank is not None:
            candidates.append((rank, -geometry_area(feat["geometry"]), feat))

    if not candidates:
        raise KeyError(
            f"no Natural Earth {scale} country matches ISO code {iso!r}; "
            "try --list-like <part of the name>")

    _, _, feat = min(candidates, key=lambda c: (c[0], c[1]))
    return {"iso": iso,
            "name": _first(feat["properties"], NAME_KEYS, iso),
            "geometry": feat["geometry"],
            "bbox": _geometry_bbox(feat["geometry"])}


def list_like(fragment, scale=NE_DEFAULT_SCALE):
    """Every country whose name or code contains `fragment`, for the CLI."""
    frag = fragment.lower()
    with open(ensure_boundaries(scale), encoding="utf-8") as f:
        fc = json.load(f)

    hits = []
    for feat in fc["features"]:
        props = feat["properties"]
        name = _first(props, NAME_KEYS, "?")
        iso = _first(props, ISO_KEYS, "?")
        if frag in name.lower() or frag in str(iso).lower():
            w, s, e, n = _geometry_bbox(feat["geometry"])
            hits.append((iso, name, (w, s, e, n)))
    return sorted(hits)


# ─────────────────────────────────────────────────────
# GRID CONSTRUCTION
# ─────────────────────────────────────────────────────
def global_grid(res_deg):
    """The full global grid at a given pixel size, anchored at (-180, 90)."""
    width = int(round(360.0 / res_deg))
    height = int(round(180.0 / res_deg))
    transform = Affine(res_deg, 0.0, GRID_ORIGIN[0],
                       0.0, -res_deg, GRID_ORIGIN[1])
    return {"crs": rasterio.crs.CRS.from_epsg(4326), "transform": transform,
            "width": width, "height": height}


def snap_bounds(bbox, margin_deg=DEFAULT_MARGIN_DEG, snap_res=SNAP_RES):
    """Grow a bbox by `margin_deg`, then round OUTWARD to `snap_res` lines.

    Snapping outward rather than to the nearest line guarantees the requested
    ground is fully contained. Snapping to the coarse grid (not to res_deg) is
    what makes the 10-arcmin and 30-arcsec country grids cover identical
    ground.
    """
    w, s, e, n = bbox
    w, s, e, n = w - margin_deg, s - margin_deg, e + margin_deg, n + margin_deg

    ox, oy = GRID_ORIGIN
    w = ox + np.floor((w - ox) / snap_res) * snap_res
    e = ox + np.ceil((e - ox) / snap_res) * snap_res
    n = oy - np.floor((oy - n) / snap_res) * snap_res
    s = oy - np.ceil((oy - s) / snap_res) * snap_res

    # Never run off the ends of the world.
    return (max(w, -180.0), max(s, -90.0), min(e, 180.0), min(n, 90.0))


def grid_from_bounds(bounds, res_deg):
    """A grid definition covering `bounds` at `res_deg`, on the global anchor."""
    w, s, e, n = bounds
    transform = Affine(res_deg, 0.0, w, 0.0, -res_deg, n)
    return {"crs": rasterio.crs.CRS.from_epsg(4326), "transform": transform,
            "width": int(round((e - w) / res_deg)),
            "height": int(round((n - s) / res_deg))}


def country_grid(iso_a3, res_deg=RES_30S, margin_deg=DEFAULT_MARGIN_DEG,
                 scale=NE_DEFAULT_SCALE, snap_res=SNAP_RES,
                 area_frac=DEFAULT_AREA_FRAC,
                 max_gap_deg=DEFAULT_MAX_GAP_DEG, bbox=None):
    """The target grid for one country: snapped bounds at the chosen pixel size.

    Because the bounds are snapped to `snap_res` and res_deg divides it, the
    result is a strict subset of the global grid at res_deg -- the transform
    origin lands on an integer pixel index of that global grid.

    `bbox` overrides the geometry-derived extent as (west, south, east, north).
    The polygon is still kept for masking; only the window changes.
    """
    country = load_country(iso_a3, scale)
    geometry, kept, dropped = select_parts(country["geometry"], area_frac,
                                           max_gap_deg)
    derived = _geometry_bbox(geometry)
    bbox = tuple(bbox) if bbox is not None else derived
    bounds = snap_bounds(bbox, margin_deg, snap_res)

    grid = grid_from_bounds(bounds, res_deg)
    grid.update(iso=country["iso"], name=country["name"], geometry=geometry,
                bbox=bbox, full_bbox=country["bbox"], bounds=bounds,
                margin_deg=margin_deg, res_deg=res_deg, area_frac=area_frac,
                max_gap_deg=max_gap_deg, parts_kept=kept, parts_dropped=dropped,
                bbox_override=bbox != derived,
                antimeridian=(_geometry_bbox(country["geometry"])[2]
                              - _geometry_bbox(country["geometry"])[0]) > 180.0)
    return grid


def window_for_bounds(src_transform, src_width, src_height, bounds):
    """The integer pixel window of a source grid covering `bounds`.

    Offsets are floor()ed and sizes ceil()ed on the source pixel lattice, so
    the window edges fall on source pixel edges and the clip needs no
    resampling. A fractional offset here would mean the target bounds are not
    on the source lattice at all, which snap_bounds() exists to prevent.
    """
    res_x, res_y = src_transform.a, -src_transform.e
    w, s, e, n = bounds

    col_off = int(np.floor((w - src_transform.c) / res_x + 1e-9))
    row_off = int(np.floor((src_transform.f - n) / res_y + 1e-9))
    col_end = int(np.ceil((e - src_transform.c) / res_x - 1e-9))
    row_end = int(np.ceil((src_transform.f - s) / res_y - 1e-9))

    col_off, row_off = max(col_off, 0), max(row_off, 0)
    col_end, row_end = min(col_end, src_width), min(row_end, src_height)
    if col_end <= col_off or row_end <= row_off:
        raise ValueError(f"bounds {bounds} do not intersect the source grid")

    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


# ─────────────────────────────────────────────────────
# MASKING
# ─────────────────────────────────────────────────────
def country_mask(grid, all_touched=True):
    """Burn the country polygon onto `grid`. 1 = inside, 0 = outside.

    all_touched=True keeps any cell the border passes through. At 18.5 km that
    matters a great deal -- a strict centroid test drops most of a narrow
    coastal country -- and at 1 km it still keeps the coastline intact.
    """
    return rasterize([(grid["geometry"], 1)],
                     out_shape=(grid["height"], grid["width"]),
                     transform=grid["transform"], fill=0,
                     all_touched=all_touched, dtype="uint8")


# ─────────────────────────────────────────────────────
# CLIPPING
# ─────────────────────────────────────────────────────
def verify_alignment(src_path, out_path, atol=1e-12):
    """Prove a clip is a pure window of its source.

    Checks the pixel size is bit-identical, the CRS matches, and the origin
    shift is a whole number of pixels. Returns (ok, message).
    """
    with rasterio.open(src_path) as a, rasterio.open(out_path) as b:
        ta, tb = a.transform, b.transform
        same_res = abs(ta.a - tb.a) < atol and abs(ta.e - tb.e) < atol
        same_crs = a.crs == b.crs
        dcol = (tb.c - ta.c) / ta.a
        drow = (tb.f - ta.f) / ta.e
        int_off = (abs(dcol - round(dcol)) < 1e-6
                   and abs(drow - round(drow)) < 1e-6)
        inside = (round(dcol) >= 0 and round(drow) >= 0
                  and round(dcol) + b.width <= a.width
                  and round(drow) + b.height <= a.height)

    ok = same_res and same_crs and int_off and inside
    msg = (f"res={'ok' if same_res else 'DIFFERS'} "
           f"crs={'ok' if same_crs else 'DIFFERS'} "
           f"offset=({round(dcol)},{round(drow)})px"
           f"{'' if int_off else ' NON-INTEGER'}"
           f"{'' if inside else ' OVERHANGS-SOURCE'}")
    return ok, msg


def clip_raster(src_path, out_path, grid, mask=False, dtype=None):
    """Window one global raster down to a country grid, preserving alignment.

    All bands are copied. No resampling happens: the output is the source
    pixels, and the output transform is the source transform translated by the
    window offset. If the country grid extends past the source (an island
    group clipped at the antimeridian, say) the output is padded with nodata
    rather than silently shrunk, so every clipped block keeps one shape.

    Values are never rescaled or re-typed and the source's nodata sentinel is
    carried through untouched. That matters: climate_recent/ flags nodata with
    -3.4e38 while soil_data/ uses NaN, and helpfully "normalising" one into the
    other would both break the project's documented `value > -1e30` test and
    make the clip something other than a copy.

    mask=True additionally sets everything outside the country polygon to the
    source's own nodata value. A source with no nodata at all has nothing to
    write there, so the output is promoted to float32 with NaN.
    """
    with rasterio.open(src_path) as src:
        if abs(src.transform.a - grid["res_deg"]) > 1e-12:
            raise ValueError(
                f"{src_path} has pixel size {src.transform.a} but the target "
                f"grid is {grid['res_deg']}; clip_raster only windows, it does "
                "not resample. Use a grid at the source resolution.")

        win = window_for_bounds(src.transform, src.width, src.height,
                                grid["bounds"])
        data = src.read(window=win)
        profile = src.profile.copy()
        src_nodata = src.nodata

        # Masking needs somewhere to put "outside the country". Reuse the
        # source's nodata if it has one; only promote to float when it does not.
        promote = mask and src_nodata is None
        out_dtype = dtype or ("float32" if promote else src.dtypes[0])
        out_nodata = np.float32(np.nan) if promote else src_nodata

        # Where the target grid overhangs the source (an island group at the
        # antimeridian, say), the read block is placed at the right offset
        # inside a full-size nodata canvas, so every clipped block keeps one
        # shape.
        pad = out_nodata
        if pad is None:
            pad = np.nan if out_dtype.startswith("float") else 0
        out = np.full((data.shape[0], grid["height"], grid["width"]),
                      pad, dtype=out_dtype)

        tw = grid["transform"]
        c0 = int(round((src.transform.c + win.col_off * src.transform.a - tw.c)
                       / tw.a))
        r0 = int(round((src.transform.f + win.row_off * src.transform.e - tw.f)
                       / tw.e))
        out[:, r0:r0 + data.shape[1], c0:c0 + data.shape[2]] = data

        if mask:
            outside = ~country_mask(grid).astype(bool)
            out[:, outside] = out_nodata

        profile.update(width=grid["width"], height=grid["height"],
                       transform=tw, dtype=out_dtype, crs=grid["crs"],
                       nodata=out_nodata,
                       compress="deflate", predictor=2, tiled=True,
                       blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        tags = dict(src.tags())
        descriptions = src.descriptions

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out)
        # The full relative path, not just the basename: climate_current/,
        # climate_recent/ and climate_2000_2015/ all hold a file called
        # wc2.1_10m_bio_1.tif, so a basename cannot identify the source block.
        tags.update(CLIPPED_FROM=os.path.relpath(src_path),
                    CLIP_COUNTRY=grid["iso"],
                    CLIP_MARGIN_DEG=f"{grid['margin_deg']:g}",
                    CLIP_MASKED=str(bool(mask)),
                    CLIP_BOUNDS=",".join(f"{v:g}" for v in grid["bounds"]),
                    CLIPPED_BY="country_clip.py")
        dst.update_tags(**tags)
        for i, desc in enumerate(descriptions, start=1):
            if desc:
                dst.set_band_description(i, desc)
    return out_path


def clip_block(block_dir, out_dir, grid, mask=False, pattern=".tif"):
    """Clip every GeoTIFF in a directory, reporting the alignment check."""
    names = sorted(f for f in os.listdir(block_dir) if f.endswith(pattern))
    if not names:
        raise FileNotFoundError(f"no {pattern} files in {block_dir}")

    os.makedirs(out_dir, exist_ok=True)
    written, failures = [], []
    for name in tqdm(names, desc=os.path.basename(block_dir.rstrip("/")),
                     unit="layer"):
        src = os.path.join(block_dir, name)
        out = os.path.join(out_dir, name)
        clip_raster(src, out, grid, mask=mask)
        ok, msg = verify_alignment(src, out)
        (written if ok else failures).append((name, msg))
    return written, failures


# ─────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────
def describe_grid(grid):
    """Human-readable summary of a country grid, including its land fraction."""
    w, s, e, n = grid["bounds"]
    res = grid["res_deg"]
    km = res * 111.32
    cells = grid["width"] * grid["height"]
    inside = int(country_mask(grid).sum())

    share = inside / cells
    lines = [
        f"country      {grid['name']} ({grid['iso']})",
        f"bbox         {grid['bbox'][0]:.3f} {grid['bbox'][1]:.3f} "
        f"{grid['bbox'][2]:.3f} {grid['bbox'][3]:.3f}",
    ]
    if grid.get("parts_dropped"):
        fb = grid["full_bbox"]
        lines.append(
            f"parts        kept {grid['parts_kept']}, dropped "
            f"{grid['parts_dropped']} outlying "
            f"(max-gap {grid['max_gap_deg']}, area-frac "
            f"{grid['area_frac']:g}); full bbox was "
            f"{fb[0]:.2f} {fb[1]:.2f} {fb[2]:.2f} {fb[3]:.2f}")
    lines += [
        f"margin       {grid['margin_deg']:g} deg "
        f"(~{grid['margin_deg'] * 111.32:.0f} km)",
        f"snapped      {w:.4f} {s:.4f} {e:.4f} {n:.4f}",
        f"resolution   {res:.8f} deg  ({res * 3600:.0f} arcsec, ~{km:.2f} km)",
        f"size         {grid['width']} x {grid['height']} = {cells:,} cells",
        f"in polygon   {inside:,} cells ({100.0 * share:.1f}% of the window)",
        f"per layer    {cells * 4 / 1e6:.1f} MB uncompressed float32",
    ]
    if grid.get("bbox_override"):
        lines.append("bbox         (manually overridden)")
    if share < SPARSE_WINDOW_WARN:
        lines.append(
            f"WARNING      only {100.0 * share:.1f}% of this window is inside "
            "the country, so most of what gets stored is sea or "
            "foreign land. Genuine for an archipelago; if not, tighten it "
            "with --max-gap or --bbox.")
    if grid.get("antimeridian"):
        lines.append(
            "WARNING      this country has parts on both sides of the "
            "antimeridian. These grids do not wrap, so the window covers "
            "whichever side the main cluster is on and the rest is dropped. "
            "Use --bbox to be explicit.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Clip global project rasters to one country, "
                    "preserving exact grid alignment.")
    parser.add_argument("iso", nargs="?", help="3-letter country code, e.g. IND")
    parser.add_argument("sources", nargs="*",
                        help="individual GeoTIFFs to clip")
    parser.add_argument("--block", action="append", default=[],
                        help="clip every .tif in this directory "
                             "(repeatable), e.g. climate_recent")
    parser.add_argument("--out-root", default=None,
                        help="output root (default country_data/<ISO>)")
    parser.add_argument("--res", type=float, default=None,
                        help="pixel size in degrees (default 1/6, the existing "
                             "global grid; 1/120 = 30 arcsec)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_DEG,
                        help=f"margin in degrees (default {DEFAULT_MARGIN_DEG})")
    parser.add_argument("--area-frac", type=float, default=DEFAULT_AREA_FRAC,
                        help="keep only the largest polygon parts covering "
                             "this fraction of the country's area, e.g. 0.99 "
                             f"(default {DEFAULT_AREA_FRAC:g}, keep all)")
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP_DEG,
                        help="drop polygon parts further than this many "
                             "degrees from the main landmass cluster "
                             f"(default {DEFAULT_MAX_GAP_DEG:g}; 0 disables "
                             "chaining, use a huge value to keep all)")
    parser.add_argument("--bbox", default=None, metavar="W,S,E,N",
                        help="override the geometry-derived extent, in degrees")
    parser.add_argument("--mask", action="store_true",
                        help="set cells outside the country polygon to nodata")
    parser.add_argument("--scale", default=NE_DEFAULT_SCALE,
                        choices=sorted(NE_SCALES))
    parser.add_argument("--info", action="store_true",
                        help="describe the grid and exit without clipping")
    parser.add_argument("--list-like", metavar="TEXT",
                        help="search Natural Earth for a country code")
    args = parser.parse_args()

    if args.list_like:
        hits = list_like(args.list_like, args.scale)
        if not hits:
            print(f"nothing matches {args.list_like!r}")
            return
        for iso, name, bbox in hits:
            print(f"  {iso:<5} {name:<40} bbox "
                  f"{bbox[0]:8.2f} {bbox[1]:7.2f} {bbox[2]:8.2f} {bbox[3]:7.2f}")
        return

    if not args.iso:
        parser.error("give a country code, or --list-like TEXT")

    bbox = None
    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))
        if len(bbox) != 4:
            parser.error("--bbox needs exactly four numbers: W,S,E,N")

    grid = country_grid(args.iso, res_deg=args.res or RES_10M,
                        margin_deg=args.margin, scale=args.scale,
                        area_frac=args.area_frac, max_gap_deg=args.max_gap,
                        bbox=bbox)

    print("=" * 62)
    print("COUNTRY CLIP")
    print("=" * 62)
    print(describe_grid(grid))

    if args.info or (not args.sources and not args.block):
        return

    out_root = args.out_root or os.path.join("country_data", grid["iso"])
    print(f"\noutput root  {out_root}/")
    print(f"masking      {'yes' if args.mask else 'no (bounding box only)'}")

    total_ok, total_bad = 0, []
    for block in args.block:
        out_dir = os.path.join(out_root, os.path.basename(block.rstrip("/")))
        written, failures = clip_block(block, out_dir, grid, mask=args.mask)
        total_ok += len(written)
        total_bad += failures
        print(f"  {block} -> {out_dir}/  ({len(written)} layers aligned)")

    for src in args.sources:
        out = os.path.join(out_root, os.path.basename(src))
        clip_raster(src, out, grid, mask=args.mask)
        ok, msg = verify_alignment(src, out)
        print(f"  {src} -> {out}  [{msg}]")
        total_ok += int(ok)
        if not ok:
            total_bad.append((src, msg))

    print("\n" + "=" * 62)
    print(f"{total_ok} layers clipped and verified grid-aligned")
    if total_bad:
        print(f"{len(total_bad)} FAILED the alignment check:")
        for name, msg in total_bad:
            print(f"  {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
