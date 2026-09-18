# Species occurrence data — runbook

How to build the training-ready tree occurrence dataset, what each number
should look like, and what to do when it does not.

The pipeline is scoped to **one country at a time**. France is the default; the
country is a config value, not a hard-coded assumption, so the same commands
re-run anywhere.

---

## 1. Quick start

```bash
# run from the repository root: the scripts live in deep_learning_sdm/, the
# data trees they write live at the root
cd /path/to/ROCm-Accelerated-Tree-Advisory-System

python3 deep_learning_sdm/gbif_api.py                              # 13 endpoint checks, ~8 s
python3 deep_learning_sdm/prepare_species_data.py --plan           # size the job, no records
python3 deep_learning_sdm/prepare_species_data.py --all            # ~30 min end to end
python3 deep_learning_sdm/prepare_species_data.py --status         # what exists so far
```

Every stage is resumable. Re-running `--all` after an interruption picks up
from the last finished species, so a dropped connection costs minutes, not the
whole job.

To run a different country:

```bash
python3 deep_learning_sdm/prepare_species_data.py --countries      # measure the candidates
python3 deep_learning_sdm/prepare_species_data.py --all --country GB
```

Outputs are namespaced by country (`species_occurrences/GB/…`,
`sampling_effort/background_GB.csv`), so two countries never overwrite each
other.

---

## 1a. What the delivered run actually contains

France, built 2026-09-16. Full detail in
`species_occurrences/FR/MANIFEST_FR.json` and
`species_occurrences/FR/species_manifest_FR.csv`.

| | |
|---|---|
| Curated names | 265 |
| Modellable in France (≥60 records) | 148 |
| **Downloaded** | **96** |
| Partial species excluded | 0 |
| Raw records | 383,876 |
| After cleaning | 243,746 |
| After thinning | 78,340 |
| Distinct grid cells | 2,517 |
| Species usable (≥30 thinned) | 96 of 96 |

**Why 96 and not 148.** GBIF's occurrence-search latency rose from ~25 s per
species at the start of the run to ~280 s near the end, and the run was
time-boxed. The queue is ordered by record count, so the 96 species present are
the best-recorded ones, not an arbitrary subset. The 52 species that were
queued but not reached are listed in the manifest with status
`not_downloaded`; re-running `--download` resumes them from checkpoints and
adds them without refetching anything.

**Consistency.** Every species was sampled with the identical 12-stratum
scheme and the identical 4,000-record budget, and every one returned its full
budget — no species was cut short mid-strata. A checkpoint is written only
after a species finishes all its passes, and `merge_raw` additionally refuses
any checkpoint whose row count disagrees with its metadata, so a
half-downloaded species cannot enter the dataset.

---

## 2. What each stage does

| Stage | Command | Time | Output |
|---|---|---|---|
| countries | `download_species.py --countries` | ~2 min | `species_occurrences/country_comparison.csv` |
| plan | `download_species.py --plan` | ~2 min | `species_occurrences/FR/species_plan.json` |
| download | `download_species.py --download` | ~25 min | `species_occurrences/FR/gbif_trees_FR_raw.csv` |
| clean | `clean_species.py` | ~1 min | `…_clean.csv`, `…_thinned.csv`, `…_flags.csv` |
| effort | `sampling_effort.py --surface --background` | ~2 min | `sampling_effort/plant_effort_FR_10m.tif`, `background_FR.csv` |

Extra checks, not part of `--all`:

```bash
python3 deep_learning_sdm/download_species.py --suggest            # well-recorded species missing from the list
python3 deep_learning_sdm/sampling_effort.py --compare             # does the background actually remove survey bias
python3 deep_learning_sdm/clean_species.py --input gbif_500_species.csv \
        --output-prefix /tmp/old_file            # quantify the damage in the old file
```

---

## 3. The species list

`tree_species_list.csv` is the product decision, held in a plain CSV so it can
be edited without touching code:

```csv
species,common_name,use,region,notes
Quercus robur,English oak,native_woodland,europe,
Tectona grandis,Teak,timber,asia,
```

`species` is the only column the pipeline reads. The rest are for humans.

**Inclusion rule.** A species is downloaded when all three hold:

1. It appears in `tree_species_list.csv`.
2. `/species/match` resolves it to an accepted **Plantae / Tracheophyta**
   species — not a genus, not a family.
3. It has at least `MIN_RECORDS_PER_SPECIES` (60) usable records in the country.

The list itself was assembled by hand from the common native and naturalised
trees of temperate Europe plus the widely planted timber, fruit, nut,
agroforestry, street and park trees of each major region, so it still applies
when `COUNTRY` changes. It is deliberately *not* "the top N species by GBIF
record count", because in France that rule promotes hedgerow shrubs —
*Crataegus monogyna*, *Prunus spinosa*, *Cornus sanguinea*, *Sambucus nigra*
are among the most-recorded woody plants in the country and nobody asks a
planting advisory whether to plant blackthorn. That ranking is still computed,
by `--suggest`, as a check on what the curated list might be missing.

**To add a species:** append a row and re-run `--plan` and `--download`. Only
the new species is fetched; existing checkpoints are reused.

**To remove one:** delete the row and re-run `--plan`, then
`download_species.py --merge-only` to rebuild the merged CSV without it.

**If a name does not resolve**, the plan prints why. The usual cause is a
synonym whose accepted name sits in a different genus, which the resolver
rejects on purpose so a typo cannot silently pull in the wrong taxon. Both
cases found in this list were real taxonomic moves:

- `Sorbus aria` → `Aria edulis`
- `Prunus dulcis` → `Prunus amygdalus`

---

## 4. Why the taxonomy is fixed now

The file this replaces, `gbif_500_species.csv`, searched with
`name_lookup(q=genus)` — Elasticsearch **full-text search**, not taxonomy. So
9.5% of its 392,725 records are not trees:

| Searched for | Also returned | Records |
|---|---|---|
| Pinus | *Dendroica pinus* (pine warbler) | 3,000 |
| Betula | *Betulapion* (a weevil) | 2,992 |
| Acer | *Erigeron acer* (blue fleabane, a herb) | 2,999 |
| Quercus | *Andricus*, *Cynips* (gall wasps) | 2,000+ |
| Quercus | *Paenibacillus quercus* (a bacterium) | — |

Its very first row is *Quercusia quercus*, a butterfly.

This pipeline resolves every name through `/species/match` to a backbone key
and then requests occurrences by **`taxonKey`**, which GBIF evaluates against
each record's full classification chain. Only descendants of that plant taxon
can come back. A warbler is excluded by the shape of the query, not by a
filter someone has to remember to apply.

`clean_species.py` re-checks the kingdom and phylum of every species name
anyway. On this pipeline's output it should find nothing. The braces have
already failed once.

---

## 5. Why sampling is stratified

Two measured facts about the GBIF occurrence API drove the design:

**The default page order front-loads the newest records, hard.** *Quercus
robur* in France has 322,231 usable records, of which only 1,794 are 2023-2026
— yet the first 300 returned are all 2025-2026, from four datasets. A naive
"take the first 4,000" yields a pure recent-citizen-science sample, which is
the most spatially biased slice available: app users walk paths and parks. A
3-species smoke test of an unstratified version came back 100%
`HUMAN_OBSERVATION`, years 2025-2026, with zero herbarium specimens.

**Deep offsets do not work.** `offset=0` answers in 0.9 s; `offset=30000` did
not answer within 60 s. Filters, by contrast, all cost ~0.15 s.

So the fix is to cut each species' query into strata small enough to page
shallowly, and never to page deep into one big query. The strata are
6 year bands × 2 latitude bands = 12 per species:

```python
YEAR_BANDS = [(1700, 1979), (1980, 1999), (2000, 2009),
              (2010, 2016), (2017, 2021), (2022, 2026)]
LAT_BANDS = 2
```

Year bands break the recency front-loading and, because old records are
herbarium specimens and new ones are observations, they spread the
basis-of-record mix too. Latitude bands add geographic spread. A third
unstratified pass picks up records carrying no year at all, which no year band
can reach.

After stratification the same three species span 1979-2026 with a median year
of 2009, against 2025-2026 unstratified.

---

## 6. Quality filters applied at download

```python
OCCURRENCE_FILTERS = {"hasCoordinate": "true",
                      "hasGeospatialIssue": "false",
                      "occurrenceStatus": "PRESENT"}
```

`basisOfRecord` is restricted to 7 of the 11 values. The two that matter:

- **`FOSSIL_SPECIMEN` excluded** — a Pliocene pollen core says nothing about
  where the tree grows today.
- **`LIVING_SPECIMEN` excluded** — this is botanic-garden and arboretum stock.
  It matters most for exactly the ornamental and street trees a planting
  advisory cares about: without this filter a jacaranda in a heated glasshouse
  in Edinburgh becomes evidence that jacarandas tolerate Scotland.

Also excluded: `coordinateUncertaintyInMeters` above 10 km, since a record
coarser than a grid cell cannot say which cell the tree was in. Records with
**unknown** uncertainty are kept (`ACCEPT_UNKNOWN_UNCERTAINTY = True`) —
most herbarium sheets record none, and dropping them would delete the
historical half of the sample.

---

## 7. Cleaning and thinning

`clean_species.py` runs four groups of tests and writes every rejected row to
`…_flags.csv` with the reason, so each drop is auditable.

1. **Taxonomy** — kingdom/phylum re-check via `/species/match`, cached.
2. **Duplicates** — exact species + coordinate repeats. These were 23.1% of the
   old file: the same herbarium sheet republished by several aggregators.
3. **Coordinate errors** — null island, `|lat| == |lon|` transpositions,
   outside the country bounding box, "magnet" coordinates, ocean points, and
   per-species range outliers.
4. **Spatial thinning** — one record per species per 1/6° cell, keeping the
   newest, on exactly the grid every raster in this repo uses.

**Why thin at all.** Two records 3 km apart carry one cell's worth of
environmental information but two rows' worth of influence on the loss
function. In a presence-only model that is how road-side sampling bias turns
into a fitted climate preference.

**The magnet test** stands in for CoordinateCleaner's institution and centroid
checks without their reference tables: one coordinate carrying ≥8 distinct
species and ≥25 records is an address, not a habitat — a herbarium's own
lat/long stamped onto every sheet it holds, or a country centroid standing in
for "somewhere in France".

**What is deliberately not implemented**, and why:

| CoordinateCleaner test | Status |
|---|---|
| `cc_cen` country/province centroids | Needs the package's 10,000-row `countryref` table. Partly caught by the magnet test. |
| `cc_inst` institution coordinates | Same — a reference table of museum positions. Partly caught by the magnet test. |
| `cc_urb` urban areas | **Skipped on purpose.** For trees an urban point is often a real street tree, and dropping them would bias the sample against exactly the warm-edge city records a planting advisory needs. |
| `cc_sea` sea points | Implemented, using this repo's own land-fraction raster. |
| `cc_outl` range outliers | Implemented, but coarser: distance from the species' geographic median rather than the full pairwise distribution, since scipy is unavailable. Catches the single wildly misplaced record, which is the error that actually occurs. |

Whole-degree coordinates are **flagged but kept** — at 1/6° a whole-degree
coordinate still lands in a defensible cell. Use `--drop-whole-degree` to
change that.

---

## 8. The sampling-effort surface

The survey-bias problem, measured: with uniform background points, built-up
fraction was the strongest single separator of presence from background
(AUC 0.746, |AUC − 0.5| = 0.246). That is not ecology. It is the observation
that people record trees where people are, and a model trained that way learns
"near cities" as habitat and then recommends planting in car parks.

The fix is **target-group background**: draw pseudo-absences in proportion to
how hard botanists have looked, not uniformly over land. Then the bias is
present in both classes and cancels.

Effort is measured as the count of all vascular-plant (Tracheophyta) records
per cell, read from GBIF's own pre-aggregated map service
(`/v2/map/occurrence/density`, Mapbox Vector Tiles). Counting through the
occurrence API instead would need one query per cell.

Restricting the tile pyramid to one country makes a much finer zoom affordable
— the world at zoom 6 is 8,192 tiles, France is 24 — so pixels are ~600 m and
roughly 1,100 of them fall inside each 1/6° cell.

`--surface` cross-checks the raster against `/occurrence/search` counts in six
latitude bands across the country. **Ratios should be within a few percent of
1.000.** If they are not, the tile buffer clipping is wrong; see the notes in
`gbif_api.density_tiles`.

Two knobs:

- `EFFORT_FLOOR = 0.5` — a cell with zero plant records is unsurveyed, not
  uninhabitable. Pure proportional sampling would give it probability exactly
  zero, quietly telling the model that unbotanised ground is suitable.
- `EFFORT_POWER = 1.0` — strict target-group sampling. Lower it to flatten the
  surface towards uniform if background ends up too concentrated to cover the
  environmental space.

Verify it worked:

```bash
python3 deep_learning_sdm/sampling_effort.py --compare
```

Built-up separation should fall towards 0. If it does not move, the effort
surface is not capturing the bias and `EFFORT_POWER` is the first thing to try.

**Known limitation.** The country mask is the bounding box intersected with
cells that have country-attributed plant records, because there is no border
polygon here. In France that is close to free — 92 million plant records leave
almost no unrecorded cells. In a sparsely recorded country it would wrongly
exclude real land, and there a real border polygon would be needed.

---

## 9. What to check before training

```bash
python3 deep_learning_sdm/prepare_species_data.py --status
```

Then read `species_occurrences/FR/gbif_trees_FR_cleaning_report.json` and check:

- **Non-plant records dropped: should be 0.** Anything above zero means the
  `taxonKey` query was bypassed somewhere.
- **Total dropped: expect a few percent.** If it is tens of percent, look at
  `…_flags.csv` and find which test is firing.
- **Species below 30 thinned records** cannot support spatial block
  cross-validation. `xgboost_training.py` already skips them at 30; they are
  listed by name at the end of the cleaning run.
- **Records per species after thinning** — the median matters more than the
  total. A species is capped by how many grid cells it occupies, not by how
  many records were downloaded, so the thinning ratio is expected to be harsh
  for the best-recorded species.

Then train on `…_thinned.csv` with background from
`sampling_effort/background_FR.csv`.

---

## 10. Config reference

| Where | Value | Meaning |
|---|---|---|
| `download_species.py` | `COUNTRY = "FR"` | the one value to change to move country |
| | `SPECIES_LIST_FILE` | `tree_species_list.csv` |
| | `MIN_RECORDS_PER_SPECIES = 60` | below this, spatial block CV cannot work |
| | `MAX_RECORDS_PER_SPECIES = 4000` | the main size dial |
| | `YEAR_BANDS`, `LAT_BANDS` | sampling strata |
| | `COUNTRY_BOUNDS` | approximate bboxes; only place latitude band edges |
| `clean_species.py` | `MAX_COORDINATE_UNCERTAINTY_M = 10000` | one grid cell |
| | `MAGNET_MIN_SPECIES = 8` | institution/centroid detection |
| | `OUTLIER_FLOOR_KM = 400` | within one country; a global run needs ~2500 |
| `sampling_effort.py` | `ZOOM = 6` | ~600 m tile pixels; cost scales as 4^zoom |
| | `N_BACKGROUND = 20000` | capped at one point per eligible cell |
| | `EFFORT_FLOOR`, `EFFORT_POWER` | see §8 |
| `gbif_api.py` | `MIN_INTERVAL = 0.12` | rate limit, ~8 requests/s |

---

## 11. Being a good GBIF citizen

No credentials are needed for anything here — the REST API is open and
`pygbif` is unavailable anyway, as is pip. A free GBIF account would add the
`/occurrence/download` endpoint, which returns a whole filtered dataset as one
DwC-A zip and a citable DOI. That is the right tool for a published analysis
and would replace §5 entirely, since it has no offset limit and no page
ordering to work around. It is asynchronous — request, wait, poll, fetch — so
it is not a drop-in replacement for the interactive loop used here.

What this pipeline does to stay polite:

- **Rate limited** to ~8 requests/s (`MIN_INTERVAL = 0.12`), single-threaded.
- **Retries with exponential backoff** on 429, 502, 503 and 504.
- **Descriptive User-Agent** identifying the project, so GBIF can see who is
  calling and why.
- **Checkpointed per species**, so an interruption resumes instead of
  re-requesting everything.

Do not raise `MAX_RECORDS_PER_SPECIES` and re-run the whole list casually. A
full France run is ~2,000 requests; that is a reasonable ask, and repeating it
needlessly is not.

---

## 12. Files

**New:**

```
tree_species_list.csv        the curated list — edit this
gbif_api.py                  shared client, taxonomy resolution, MVT decoder
download_species.py          country selection, planning, download
clean_species.py             cleaning and spatial thinning
sampling_effort.py           effort surface and background points
prepare_species_data.py      one entry point for all stages
SPECIES_DATA_RUNBOOK.md      this file
```

**Not touched:** `gbif_500_species.csv` (kept because it cannot currently be
regenerated, and is the evidence for §4), `soil_data/`, `topography/`,
`satellite/`, `climate_*/`, `validation/`, `xgboost_training.py`.

`xgboost_training.py` still uses its own uniform `generate_pseudo_absences()`.
Switching it to `background_FR.csv` is the next change, and is what
`sampling_effort.py --compare` is there to justify.
