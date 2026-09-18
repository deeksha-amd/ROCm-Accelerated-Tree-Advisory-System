# What data each country has

Plain-language companion to `../DATA_OVERVIEW.md`, which covers the global
blocks. This one covers the per-country data the model actually trains on.

---

## Status

| Country | Rasters | Tree species | Records (thinned) | Model trained |
|---|---|---|---|---|
| **France** (metropolitan) | 274 MB, complete | **96** | 78,340 | **Yes** |
| **Spain** (mainland + Balearics) | 186 MB | 8 so far, of 148 | 3,829 | No |
| **USA** (lower 48) | 2.9 GB, in progress | downloading, of 284 | — | No |

Only France is served by the advisory today. Ask about a Spanish or American
location and you get "no model covers this location", which is the honest
answer until those finish.

---

## What "prepared" means

Every country gets the same tree of files, on two grids that nest exactly:

- **1 km** (30 arcsec) — what the model trains on
- **18.5 km** (10 arcmin) — the original global grid, kept for comparison

Per country, on both grids:

| Block | What it is | Layers |
|---|---|---|
| Climate | temperature, rainfall, seasonality, plus sun, wind, humidity and evaporation | 29 |
| Soil | 11 properties at two depths (0–30 cm and 30–60 cm) | 22 |
| Terrain | elevation, slope, which way the land faces | 12 |
| Satellite | NDVI and land cover — **downloaded but not used by the model** | separate |

Satellite sits in its own folder with a `DO_NOT_TRAIN_ON_THIS.md` marker.
See `../DATA_OVERVIEW.md` for why.

---

## What the cleaning actually removes

France, as the worked example: **384k records downloaded → 244k kept → 78k
used.**

| Removed | Count | Why |
|---|---|---|
| Duplicates | 91,419 | same species, same exact coordinates |
| "Magnet" coordinates | 2,369 | 1,180 spots where 8+ species share one point — a museum, a herbarium, or a record pinned to a town centre rather than an actual tree |
| Range outliers | 561 | a lone record thousands of km from the rest of its species |
| In the sea | 89 | bad coordinates |

Then **spatial thinning**: keep one record per species per grid cell. That
drops 244k to 78k. It sounds wasteful, but 500 sightings in one park tell the
model no more than one does, and keeping them all would teach it that popular
parks are good habitat.

A visible sign it works: the median record year climbs from 2009 to 2021 as
cleaning proceeds. The junk skews old, because digitised historical specimens
carry the vaguest locations.

### Background points

Alongside the sightings, 20,000 "the model should not expect trees here"
points. These are **not** random — they are placed where botanists actually
searched and recorded something else.

This matters more than it sounds. With random points, the model scored 0.917
and was really just learning "near a road or town". With these, it scores
0.747, and that lower number is the honest one.

---

## Sizes

| | Size | Keep? |
|---|---|---|
| `country_data/` | 3.4 GB | **Yes** — the predictor maps |
| `species_occurrences/` | 108 MB | **Yes** — the sightings |
| `sampling_effort/` | 2.6 MB | **Yes** — background points |
| `deep_sdm_output/` | 55 MB | trained models and metrics |
| `country_src_cache/` | 8.3 GB | no — raw downloads, re-fetchable |
| `venv/` | 14 GB | no — PyTorch, rebuildable |

The data that matters is about **3.5 GB**. The rest is cache and toolchain,
both gitignored and safe to delete.

The US is 2.9 GB against France's 274 MB simply because it is about 15 times
the area at the same 1 km detail.

---

## Known gaps

- **Spain and the USA have no trained model yet** — data only.
- **The US species download is slow.** GBIF throttles after sustained use,
  from ~25 seconds per species to ~3 minutes. It resumes from checkpoints.
- **France stopped at 96 of 148 modellable species** for the same reason. The
  queue is ordered by record count, so what is there is the best-recorded
  subset, not an arbitrary slice.
- **Overseas territories are excluded on purpose.** French Guiana, Réunion and
  the Canaries would put rainforest and subtropical islands in the same model
  as temperate mainland, which is a worse model, not a broader one.
