"""
Minimal GBIF REST client — shared by download_species.py and sampling_effort.py

Why this file exists instead of `pygbif`: pygbif is not installed and cannot be
installed here (system pip is blocked by PEP 668). Every GBIF endpoint we need
is anonymous, keyless JSON over HTTPS, so `requests` is sufficient.

Three things live here:

1. `GbifClient` — one pooled session with a descriptive User-Agent, a rate
   limit and retry-with-backoff. GBIF asks for a contactable User-Agent and
   throttles hard on 429/503; a bare `requests.get` loop gets you blocked.

2. Taxonomy helpers built on `/species/match`, which is the *taxonomic* lookup.
   The old script used `/species/name_lookup` (full-text search), which is why
   `gbif_500_species.csv` contains warblers, gall wasps and a bacterium.
   See `match_genus` for the full explanation.

3. `density_tiles` — a from-scratch reader for GBIF's Mapbox Vector Tile
   occurrence-density service. It returns per-pixel record counts for an entire
   taxon in a handful of requests, which is how the sampling-effort surface in
   sampling_effort.py is built without downloading 500 million records.
   MVT is protobuf; there is no protobuf library here either, so the ~60 lines
   of varint/zigzag decoding below are hand-rolled.

Measured behaviour of the endpoints (probed 2026-09-16, keyless):

    /v1/occurrence/count      REJECTS hasCoordinate, occurrenceStatus and
                              decimalLatitude. It only accepts taxonKey,
                              country, year, basisOfRecord, isGeoreferenced
                              and a few others, so it is useless for planning
                              against our real filter set.
    /v1/occurrence/search     with limit=0 returns the same `count` field in
                              ~0.2 s and DOES accept every filter, including
                              decimalLatitude/decimalLongitude ranges. This is
                              the counting endpoint we actually use.
    /v1/occurrence/search     paging is hard-capped: offset + limit <= 100001,
                              limit is silently clamped to 300.
    facet=speciesKey          returns at most 1200 distinct values.
    /v2/map/.../{z}/{x}/{y}.mvt
                              EPSG:4326 has 2*2^z columns by 2^z rows; layer
                              extent is 512 px; tiles carry a 64 px buffer, so
                              pixels outside [0,512) are copies of neighbouring
                              tiles and MUST be dropped or records near a tile
                              edge count twice. After clipping, the global sum
                              of the `total` attribute reproduces the search
                              count to within 0.4% at species and genus level
                              and 0.2% against the fully filtered phylum count,
                              with zero duplicated pixels. The service applies
                              hasGeospatialIssue=false and occurrenceStatus=
                              PRESENT itself; it honours taxonKey, year and
                              basisOfRecord, and SILENTLY IGNORES any parameter
                              it does not know (including misspelled ones).
"""

import struct
import time

import numpy as np
import requests

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
V1 = "https://api.gbif.org/v1"
MAP_V2 = "https://api.gbif.org/v2/map/occurrence/density"

# GBIF's terms of use ask for a User-Agent that identifies the client well
# enough to be contacted if it misbehaves.
USER_AGENT = ("ROCm-Accelerated-Tree-Advisory-System/1.0 "
              "(species distribution model data prep; "
              "https://github.com/deeksha-amd/ROCm-Accelerated-Tree-Advisory-System)")

MIN_INTERVAL = 0.12      # seconds between requests (~8/s); GBIF tolerates this
MAX_ATTEMPTS = 6
BACKOFF_BASE = 1.8       # seconds; attempt n waits BACKOFF_BASE**n
# A normal page answers in 0.5-0.9 s, so 45 s is ample headroom. It was 180,
# which let a single wedged request stall the whole download for three minutes
# before even the first retry — and with retries, for a quarter of an hour.
# Failing fast and retrying is strictly better when the work is checkpointed.
TIMEOUT = 45

PAGE_SIZE = 300          # API maximum
MAX_OFFSET = 100_000     # API maximum; offset + limit must be <= 100001
FACET_LIMIT = 1200       # API maximum number of distinct facet values

BACKBONE_DATASET = "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c"   # GBIF backbone
TRACHEOPHYTA_KEY = 7707728     # phylum, the "target group" for a tree SDM
PLANTAE_KEY = 6                # kingdom; a match falling back to this is a bug

MAP_ZOOM = 3             # EPSG:4326 zoom for the effort surface. Pixel size is
                         # 180/(2^z * 512) deg: z=2 -> 0.088, z=3 -> 0.044,
                         # z=4 -> 0.022. The target grid is 0.1667 deg, so z=3
                         # puts ~14 tile pixels in every output cell — enough
                         # that the non-integer grid ratio does not alias.
                         # Cost scales as 4^z: z=3 is 128 tiles / ~52 MB.

# The quality filters that define "a usable natural occurrence of a tree".
# Applied server-side so the API never sends us records we would throw away.
OCCURRENCE_FILTERS = {
    "hasCoordinate": "true",        # must have decimalLatitude/Longitude
    "hasGeospatialIssue": "false",  # drops GBIF's own coordinate-validity flags
    "occurrenceStatus": "PRESENT",  # drops surveyed-and-absent records
}

# basisOfRecord values kept. The two exclusions matter enormously for trees:
#   FOSSIL_SPECIMEN  — a Miocene leaf impression says nothing about today's
#                      climate envelope, which is what we are modelling.
#   LIVING_SPECIMEN  — botanic-garden and arboretum accessions. A Eucalyptus in
#                      Kew Gardens is a horticultural fact, not a natural
#                      occurrence, and these plantings sit overwhelmingly in
#                      rich temperate cities, so they import survey bias and
#                      climatic nonsense at the same time.
BASIS_OF_RECORD = [
    "HUMAN_OBSERVATION",
    "PRESERVED_SPECIMEN",
    "OBSERVATION",
    "MACHINE_OBSERVATION",
    "MATERIAL_SAMPLE",
    "MATERIAL_CITATION",
    "OCCURRENCE",
]


class GbifRequestError(RuntimeError):
    """A GBIF request failed after every retry."""


# ─────────────────────────────────────────────────────
# HTTP CLIENT
# ─────────────────────────────────────────────────────
class GbifClient:
    """Rate-limited, retrying GBIF session.

    Deliberately synchronous and single-connection. The whole job is a few
    thousand small requests; hammering GBIF with a thread pool to save twenty
    minutes is not a trade worth making against a free public service.
    """

    def __init__(self, min_interval=MIN_INTERVAL, user_agent=USER_AGENT):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent,
                                     "Accept-Encoding": "gzip, deflate"})
        self.min_interval = min_interval
        self._last = 0.0
        self.n_requests = 0
        self.n_retries = 0
        self.bytes_received = 0

    def _wait(self):
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def get(self, url, params=None, raw=False, allow_404=False):
        """GET with retries. Returns parsed JSON, or bytes when raw=True."""
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            self._wait()
            try:
                response = self.session.get(url, params=params, timeout=TIMEOUT)
            except requests.exceptions.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                self.n_requests += 1
                self.bytes_received += len(response.content)
                if response.status_code == 200:
                    return response.content if raw else response.json()
                # The map service answers 204 No Content for a tile holding no
                # records at all, which is normal over sea and outside the
                # country filter, not a failure.
                if response.status_code == 204:
                    return b"" if raw else None
                if response.status_code == 404 and allow_404:
                    return None
                # 429 = throttled, 5xx = transient. 4xx otherwise is our bug
                # and retrying cannot help, so fail loudly and immediately.
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise GbifRequestError(
                        f"{response.status_code} from {url} "
                        f"params={params} body={response.text[:200]}")
                last_error = f"HTTP {response.status_code}"

            self.n_retries += 1
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_BASE ** attempt)

        raise GbifRequestError(f"{url} failed after {MAX_ATTEMPTS} attempts "
                               f"({last_error})")

    # ── occurrence search ───────────────────────────
    def occurrence_search(self, limit=PAGE_SIZE, offset=0, facet=None,
                          facet_limit=FACET_LIMIT, basis_of_record=None,
                          **filters):
        """/occurrence/search. Repeated params (basisOfRecord) need a list."""
        params = [("limit", limit), ("offset", offset)]
        params += [(k, v) for k, v in filters.items() if v is not None]
        for basis in (basis_of_record or []):
            params.append(("basisOfRecord", basis))
        if facet:
            params += [("facet", facet), ("facetLimit", facet_limit)]
        return self.get(f"{V1}/occurrence/search", params)

    def count(self, basis_of_record=None, **filters):
        """Exact record count under the full filter set.

        Uses search with limit=0 rather than /occurrence/count, which rejects
        hasCoordinate and occurrenceStatus outright (see module docstring).
        """
        return int(self.occurrence_search(
            limit=0, basis_of_record=basis_of_record, **filters)["count"])

    def facet_counts(self, facet, basis_of_record=None, **filters):
        """Return [(value, count), ...] for one facet, biggest first."""
        payload = self.occurrence_search(
            limit=0, facet=facet, basis_of_record=basis_of_record, **filters)
        facets = payload.get("facets") or []
        if not facets:
            return []
        return [(c["name"], int(c["count"])) for c in facets[0]["counts"]]

    # ── taxonomy ────────────────────────────────────
    def species(self, key):
        """/species/{key} — full backbone record for one usage key."""
        return self.get(f"{V1}/species/{int(key)}", allow_404=True)

    def match(self, name, rank=None, kingdom=None, phylum=None):
        """/species/match — fuzzy-but-taxonomic name resolution."""
        params = {"name": name, "strict": "false"}
        if rank:
            params["rank"] = rank
        if kingdom:
            params["kingdom"] = kingdom
        if phylum:
            params["phylum"] = phylum
        return self.get(f"{V1}/species/match", params)

    def species_search(self, query, rank=None, limit=30):
        """/species/search restricted to the GBIF backbone taxonomy."""
        params = {"q": query, "datasetKey": BACKBONE_DATASET, "limit": limit}
        if rank:
            params["rank"] = rank
        return self.get(f"{V1}/species/search", params)

    # ── density tiles ───────────────────────────────
    def density_tile(self, z, x, y, **filters):
        """One MVT density tile as raw bytes, or None if empty."""
        params = {"srs": "EPSG:4326"}
        params.update({k: v for k, v in filters.items() if v is not None})
        blob = self.get(f"{MAP_V2}/{z}/{x}/{y}.mvt", params, raw=True)
        return blob or None


# ─────────────────────────────────────────────────────
# TAXONOMY: the fix for the bird/wasp/fungus contamination
# ─────────────────────────────────────────────────────
def _search_genus_fallback(client, genus):
    """Resolve a genus that /species/match refuses, via the backbone index.

    `match` gives up and returns `matchType=HIGHERRANK` when a name is a
    homonym across several families. Cedrus is the live example: the backbone
    holds five plant genera called Cedrus (accepted in Pinaceae, a synonym in
    Meliaceae, doubtful in Cupressaceae and two more), so `match` returns the
    phylum Tracheophyta instead — 523 million records — and the rank guard in
    `match_genus` is the only thing standing between that and the download.

    /species/search returns the candidates rather than collapsing them, so the
    single ACCEPTED vascular-plant genus with exactly this name can be picked
    out. If there is more than one, this fails and says so; pin the key by
    hand rather than letting the script guess.
    """
    payload = client.species_search(genus, rank="GENUS")
    exact = [r for r in payload.get("results", [])
             if (r.get("canonicalName") or "").lower() == genus.lower()
             and r.get("kingdom") == "Plantae"
             and r.get("phylum") == "Tracheophyta"
             and r.get("taxonomicStatus") == "ACCEPTED"
             and r.get("rank") == "GENUS"]

    if not exact:
        return None, "no accepted vascular-plant genus of that name"
    if len(exact) > 1:
        families = sorted(str(r.get("family")) for r in exact)
        return None, (f"{len(exact)} accepted plant genera named {genus} "
                      f"({', '.join(families)}); pin the key in "
                      f"GENUS_KEY_OVERRIDES")

    record = exact[0]
    return {"genus": genus,
            "accepted_name": record.get("canonicalName"),
            "key": int(record["key"]),
            "family": record.get("family"),
            "order": record.get("order"),
            "class": record.get("class"),
            "status": record.get("taxonomicStatus"),
            "confidence": None,
            "resolved_via": "species/search"}, None


def match_genus(client, genus):
    """Resolve a genus name to a verified plant backbone key.

    THIS IS THE CORE FIX. The old download_species.py called
    `name_lookup(q="Quercus")`, which is Elasticsearch full-text search over
    every name string in GBIF. It matches "Quercus" anywhere in any field, so
    it happily returned `Paenibacillus quercus` (a bacterium), `Andricus
    quercuscalicis` (a gall wasp) and `Boletellus betula` (a fungus), and the
    per-species keys it handed on were those organisms' keys. 9.5% of the
    resulting file is not a plant.

    `/species/match` is a different service: it resolves a name against the
    GBIF backbone taxonomy and returns one `usageKey`. Querying occurrences by
    `taxonKey=<that key>` restricts results to that taxon *and its
    descendants*, because GBIF indexes every occurrence against its full
    classification chain. A warbler is not a descendant of Quercus, so it
    cannot come back — the contamination is excluded by the shape of the query
    rather than by a filter we have to remember to apply.

    Two traps this function guards against, both observed live:

      * Without hints, `match` returns usageKey=None for ambiguous names —
        Ficus, Cordia, Corymbia, Senegalia and Polylepis all fail silently.
        So we pass kingdom=Plantae and rank=GENUS.
      * With those hints, an unmatchable name falls back to the nearest
        ancestor instead of failing. `match(name="Dendroica", kingdom="Plantae")`
        returns key=6, rank=KINGDOM — the entire plant kingdom, and
        match(name="Cedrus") returns key=7707728, rank=PHYLUM. Downloading
        either would be catastrophic. So we require rank == GENUS, kingdom ==
        Plantae, phylum == Tracheophyta, and that the returned canonical name
        is the name we asked for. Genuine genera that trip the rank guard
        because they are homonyms are recovered by `_search_genus_fallback`.
    """
    payload = client.match(genus, rank="GENUS", kingdom="Plantae")
    key = payload.get("usageKey")
    returned = (payload.get("canonicalName") or payload.get("genus") or "")

    # A HIGHERRANK match means "I could not pin this name down, here is an
    # ancestor". Never usable, but often recoverable from the backbone index.
    if not key or payload.get("rank") != "GENUS" \
            or returned.lower() != genus.lower():
        info, reason = _search_genus_fallback(client, genus)
        if info is not None:
            return info, None
        if not key:
            return None, f"no match; {reason}"
        return None, (f"resolved to rank {payload.get('rank')} "
                      f"({returned!r}), not a genus; {reason}")

    if payload.get("kingdom") != "Plantae":
        return None, f"kingdom {payload.get('kingdom')}, not Plantae"
    if payload.get("phylum") != "Tracheophyta":
        return None, f"phylum {payload.get('phylum')}, not Tracheophyta"

    # A synonym genus still has occurrences indexed under it, but the accepted
    # key covers strictly more, so prefer it when GBIF offers one.
    if payload.get("status") == "SYNONYM" and payload.get("acceptedUsageKey"):
        key = payload["acceptedUsageKey"]
        accepted = client.species(key) or {}
        if accepted.get("rank") != "GENUS" or accepted.get("kingdom") != "Plantae":
            return None, "synonym resolved to a non-genus plant taxon"
        returned = accepted.get("canonicalName", returned)

    return {"genus": genus,
            "accepted_name": returned,
            "key": int(key),
            "family": payload.get("family"),
            "order": payload.get("order"),
            "class": payload.get("class"),
            "status": payload.get("status"),
            "confidence": payload.get("confidence"),
            "resolved_via": "species/match"}, None


def match_species(client, name):
    """Resolve a binomial to a verified vascular-plant species key.

    Same mechanism and the same guards as `match_genus`, at species rank. The
    guards matter just as much here: `match` falls back to an ancestor rather
    than failing, so without the rank check a misspelling in the curated CSV
    would silently resolve to a genus — or to the plant kingdom — and the
    download would pull every species in it.

    Synonyms are followed to the accepted name, since that is the key GBIF
    indexes occurrences under. The returned `species` is the accepted name, so
    the output CSV stays consistent even where the curated list used an older
    name (Sophora japonica -> Styphnolobium japonicum, for instance).
    """
    payload = client.match(name, rank="SPECIES", kingdom="Plantae")
    key = payload.get("acceptedUsageKey") or payload.get("usageKey")

    if not key:
        return None, "no match"
    if payload.get("rank") != "SPECIES":
        return None, f"resolved to rank {payload.get('rank')}, not a species"
    if payload.get("kingdom") != "Plantae":
        return None, f"kingdom {payload.get('kingdom')}, not Plantae"
    if payload.get("phylum") != "Tracheophyta":
        return None, f"phylum {payload.get('phylum')}, not Tracheophyta"

    # matchType FUZZY means GBIF guessed at a misspelling. Accept it, but only
    # when it stayed inside the same genus, so "Quercus robr" is repaired while
    # a name that drifts to a different genus is rejected as a list error.
    canonical = payload.get("canonicalName") or name
    if canonical.split()[0].lower() != name.split()[0].lower():
        return None, (f"matched {canonical!r}, a different genus "
                      f"(matchType={payload.get('matchType')})")

    return {"key": int(key),
            "accepted_name": canonical,
            "family": payload.get("family"),
            "order": payload.get("order"),
            "genus": payload.get("genus"),
            "status": payload.get("status"),
            "match_type": payload.get("matchType"),
            "confidence": payload.get("confidence")}, None


def is_vascular_plant(record):
    """Defensive check on a /species/{key} payload."""
    return bool(record) and record.get("kingdom") == "Plantae" \
        and record.get("phylum") == "Tracheophyta"


# ─────────────────────────────────────────────────────
# MVT DECODER  (Mapbox Vector Tile = protobuf, hand-rolled)
# ─────────────────────────────────────────────────────
# Wire format, only the parts we need:
#   Tile   field 3  = repeated Layer
#   Layer  field 1  = name, 2 = repeated Feature, 3 = repeated key (string),
#                     4 = repeated Value, 5 = extent (varint)
#   Value  field 1  = string, 2 = float, 3 = double, 4/5 = (u)int64,
#                     6 = sint64 (zigzag), 7 = bool
#   Feature field 2 = packed tag varints (key index, value index, ...),
#                     3 = geometry type, 4 = packed geometry commands
def _varint(buf, i):
    shift = value = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


def _zigzag(value):
    return (value >> 1) ^ -(value & 1)


def _fields(buf, start, end):
    """Yield (field_number, value) for one protobuf message."""
    i = start
    while i < end:
        tag, i = _varint(buf, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, i = _varint(buf, i)
            yield field, value
        elif wire == 2:                      # length-delimited: (start, end)
            length, i = _varint(buf, i)
            yield field, (i, i + length)
            i += length
        elif wire == 5:
            yield field, struct.unpack_from("<f", buf, i)[0]
            i += 4
        elif wire == 1:
            yield field, struct.unpack_from("<d", buf, i)[0]
            i += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def decode_density_tile(blob):
    """Decode one GBIF density tile to (x, y, total, extent).

    x/y are integer pixel coordinates within the tile, INCLUDING the 64 px
    buffer, so callers must clip to [0, extent) before using `total`.
    """
    xs, ys, totals, extent = [], [], [], 4096

    for field, span in _fields(blob, 0, len(blob)):
        if field != 3:                       # not a Layer
            continue
        keys, values, features = [], [], []
        for lfield, lspan in _fields(blob, *span):
            if lfield == 3:
                keys.append(blob[lspan[0]:lspan[1]].decode("utf-8", "replace"))
            elif lfield == 4:
                for vfield, value in _fields(blob, *lspan):
                    values.append(blob[value[0]:value[1]].decode("utf-8", "replace")
                                  if vfield == 1 else
                                  _zigzag(value) if vfield == 6 else value)
            elif lfield == 5:
                extent = lspan                # varint, so already an int
            elif lfield == 2:
                features.append(lspan)

        if "total" not in keys:
            continue
        total_index = keys.index("total")

        for fspan in features:
            tags, geometry = [], []
            for ffield, fspan_inner in _fields(blob, *fspan):
                if ffield == 2:
                    i, end = fspan_inner
                    while i < end:
                        tag, i = _varint(blob, i)
                        tags.append(tag)
                elif ffield == 4:
                    i, end = fspan_inner
                    while i < end:
                        tag, i = _varint(blob, i)
                        geometry.append(tag)
            if not geometry:
                continue

            total = None
            for k in range(0, len(tags) - 1, 2):
                if tags[k] == total_index:
                    total = values[tags[k + 1]]
            if total is None:
                continue

            # Density tiles are MULTIPOINT: one MoveTo command carrying n
            # zigzag-delta point pairs. Every observed feature had n == 1, but
            # handle n > 1 rather than silently dropping points.
            n_points = geometry[0] >> 3
            cursor_x = cursor_y = 0
            i = 1
            for _ in range(n_points):
                cursor_x += _zigzag(geometry[i])
                cursor_y += _zigzag(geometry[i + 1])
                i += 2
                xs.append(cursor_x)
                ys.append(cursor_y)
                totals.append(total / n_points)

    return (np.asarray(xs, dtype="int64"), np.asarray(ys, dtype="int64"),
            np.asarray(totals, dtype="float64"), extent)


def tile_grid(zoom):
    """(columns, rows) of the EPSG:4326 tile pyramid at this zoom."""
    return 2 * 2 ** zoom, 2 ** zoom


def tile_pixel_bounds(zoom, tx, ty, extent):
    """Degrees per tile pixel and the tile's north-west corner."""
    n_cols, n_rows = tile_grid(zoom)
    tile_lon = 360.0 / n_cols
    tile_lat = 180.0 / n_rows
    return (-180.0 + tx * tile_lon, 90.0 - ty * tile_lat,
            tile_lon / extent, tile_lat / extent)


def tiles_covering(zoom, bounds):
    """Tile (x, y) ranges covering a (lat_min, lat_max, lon_min, lon_max) box.

    Restricting to one country's box is what makes a high zoom affordable:
    the whole world at zoom 6 is 8,192 tiles, while France is 24.
    """
    n_cols, n_rows = tile_grid(zoom)
    lat_min, lat_max, lon_min, lon_max = bounds
    tile_lon, tile_lat = 360.0 / n_cols, 180.0 / n_rows
    tx0 = max(0, int((lon_min + 180.0) / tile_lon))
    tx1 = min(n_cols - 1, int((lon_max + 180.0) / tile_lon))
    ty0 = max(0, int((90.0 - lat_max) / tile_lat))
    ty1 = min(n_rows - 1, int((90.0 - lat_min) / tile_lat))
    return range(tx0, tx1 + 1), range(ty0, ty1 + 1)


def density_tiles(client, zoom, progress=None, bounds=None, **filters):
    """Iterate tiles at `zoom`, yielding clipped (lon, lat, count) arrays.

    Covers the whole world unless `bounds` restricts it. Pixel centres are
    returned, so the caller can bin them onto any grid.
    """
    if bounds is None:
        n_cols, n_rows = tile_grid(zoom)
        xs, ys = range(n_cols), range(n_rows)
    else:
        xs, ys = tiles_covering(zoom, bounds)

    for tx in xs:
        for ty in ys:
            blob = client.density_tile(zoom, tx, ty, **filters)
            if progress is not None:
                progress.update(1)
            if not blob:
                continue
            x, y, total, extent = decode_density_tile(blob)
            if not len(x):
                continue

            # Drop the 64 px buffer, or every record near a tile edge is
            # counted twice. With the buffer in, the global Tracheophyta sum
            # came to 1.76e9 at zoom 1; clipped it is 5.227e8, against a
            # search count of 5.235e8 under the same filters. Zero pixel
            # positions repeat across tiles once clipped, which is the check
            # that the tile scheme below is the right one.
            keep = (x >= 0) & (x < extent) & (y >= 0) & (y < extent)
            if not keep.any():
                continue
            x, y, total = x[keep], y[keep], total[keep]

            west, north, dlon, dlat = tile_pixel_bounds(zoom, tx, ty, extent)
            lon = west + (x + 0.5) * dlon
            lat = north - (y + 0.5) * dlat
            yield lon, lat, total


# ─────────────────────────────────────────────────────
# SELF TEST  (tiny probes only — no bulk download)
# ─────────────────────────────────────────────────────
def selftest(client=None):
    """Prove the endpoints still behave as documented. ~12 small requests."""
    client = client or GbifClient()
    checks = []

    def check(label, ok, detail=""):
        checks.append((label, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:44s} {detail}")

    print("GBIF endpoint self-test")

    info, err = match_genus(client, "Quercus")
    check("species/match Quercus -> plant genus", info is not None,
          f"key={info['key'] if info else err}")

    info, err = match_genus(client, "Ficus")
    check("species/match Ficus needs kingdom hint", info is not None,
          f"key={info['key'] if info else err}")

    info, err = match_genus(client, "Cedrus")
    check("homonym genus recovered from the backbone",
          info is not None and info["family"] == "Pinaceae",
          f"key={info['key']} via {info['resolved_via']}" if info else err)

    bad, err = match_genus(client, "Dendroica")
    check("species/match rejects the warbler genus", bad is None, err or "")

    bad, err = match_genus(client, "Betulapion")
    check("species/match rejects the weevil genus", bad is None, err or "")

    info, err = match_species(client, "Quercus robur")
    check("species/match resolves a binomial",
          info is not None and info["key"] == 2878688,
          f"key={info['key']} {info['family']}" if info else err)

    bad, err = match_species(client, "Quercus")
    check("a bare genus is rejected at species rank", bad is None, err or "")

    n = client.count(taxonKey=2878688, basis_of_record=BASIS_OF_RECORD,
                     **OCCURRENCE_FILTERS)
    check("search?limit=0 counts with full filters", n > 100_000, f"{n:,}")

    facets = client.facet_counts("speciesKey", taxonKey=2877951,
                                 basis_of_record=BASIS_OF_RECORD,
                                 **OCCURRENCE_FILTERS)
    check("facet=speciesKey enumerates a genus", len(facets) > 100,
          f"{len(facets)} Quercus species with data")

    page = client.occurrence_search(limit=2, taxonKey=2878688,
                                   basis_of_record=BASIS_OF_RECORD,
                                   **OCCURRENCE_FILTERS)
    fields = page["results"][0] if page.get("results") else {}
    needed = {"decimalLatitude", "decimalLongitude", "species",
              "basisOfRecord", "countryCode"}
    check("occurrence records carry our schema", needed <= set(fields),
          f"missing={sorted(needed - set(fields))}")

    blob = client.density_tile(1, 2, 0, taxonKey=TRACHEOPHYTA_KEY)
    x, y, total, extent = decode_density_tile(blob) if blob else ((), (), (), 0)
    check("density tile decodes", len(x) > 1000,
          f"{len(x)} pixels, extent={extent}, {len(blob or b''):,} bytes")

    if len(x):
        keep = (x >= 0) & (x < extent) & (y >= 0) & (y < extent)
        check("density tile carries a 64 px buffer", (~keep).any(),
              f"{(~keep).sum()} of {len(x)} pixels outside [0,{extent})")

        west, north, dlon, dlat = tile_pixel_bounds(1, 2, 0, extent)
        lon = west + (x[keep] + 0.5) * dlon
        lat = north - (y[keep] + 0.5) * dlat
        box = (lat >= 45) & (lat < 55) & (lon >= 0) & (lon < 20)
        api = client.count(taxonKey=TRACHEOPHYTA_KEY,
                           decimalLatitude="45,55", decimalLongitude="0,20",
                           **OCCURRENCE_FILTERS)
        ratio = total[keep][box].sum() / max(api, 1)
        check("tile counts agree with search counts", 0.95 <= ratio <= 1.05,
              f"ratio={ratio:.3f} over lat 45-55N lon 0-20E "
              f"(tiles={total[keep][box].sum():,.0f} search={api:,})")

    ok = all(c[1] for c in checks)
    print(f"\n{sum(c[1] for c in checks)}/{len(checks)} checks passed; "
          f"{client.n_requests} requests, {client.bytes_received/1e6:.1f} MB")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
