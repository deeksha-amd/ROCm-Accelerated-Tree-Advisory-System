# The Deep SDM model

A multi-species Species Distribution Model. One network, 96 outputs — not 96
separate models.

Files: `deep_sdm_training.py` (model, training, metrics),
`sdm_features.py` (data layer).

---

## Architecture

```
64 environmental features
      ↓
Linear 64 → 256  +  LayerNorm  +  ReLU  +  Dropout 0.30
      ↓
Linear 256 → 128 +  LayerNorm  +  ReLU  +  Dropout 0.30
      ↓                                      ← shared encoder ends here
Linear 128 → 96                              ← one row of weights per species
      ↓
96 suitability scores
```

**62,688 parameters** total:

| Part | Params |
|---|---|
| Encoder layer 1 (64→256) | 16,896 |
| Encoder layer 2 (256→128) | 33,152 |
| Species heads (128→96) | 12,384 |

Tiny by deep-learning standards. That is deliberate — see "capacity" below.

### Why this shape

The encoder learns, from every record of every species, what an environment
*is*. Each species then gets one row of 128 weights describing its niche in
that learned space.

The consequence that matters: a species with 116 records (cork oak) does not
have to learn what soil and climate mean from 116 points. The encoder was paid
for by the common species; cork oak only fits its own row. Per-species models
cannot do this.

The final `Linear(128, 96)` layer *is* a species embedding — row *s* is
species *s*'s vector. An explicit embedding table was not added because it
would only earn its cost if it conditioned the encoder, or if scores were
needed for species absent from training.

### Why not a CNN

At 1 km a 64×64 patch spans 1,200 km. A "neighbourhood" would be a
subcontinent, so there is no local spatial structure to convolve over. A
point-wise model fits the data that exists.

---

## Training setup

| | |
|---|---|
| Loss | masked BCE — presence cells are positives for the recorded species and **masked** for others (GBIF is presence-only, so "not recorded" ≠ "absent") |
| Background | 2 cells per presence cell, drawn from a survey-effort surface, negatives for all species |
| Optimiser | Adam, lr 1e-3, weight decay 1e-4 |
| Batch | 4096 |
| Epochs | 400 budget, early stopping on mean per-species AUC, patience 10 |
| Validation | 5-fold spatial-block CV over 10 geographic clusters |
| Normalisation | z-score, fitted on training folds only |
| Missing values | median impute + a missingness indicator per predictor group |
| Hardware | AMD Instinct MI210 via ROCm; 1 km run takes ~2.5 min |

Spatial-block CV is the part that stops the score being fiction. Random splits
would put a training point 200 m from a test point and report a near-perfect
model that had memorised locations.

---

## Inputs

61 rasters → 64 feature columns, all at 1 km:

- **28 climate** — 19 bioclim plus aridity, PET (two methods), solar
  radiation, vapour pressure, VPD, water deficit, wind
- **22 soil** — 11 properties × 2 depths (0–30 cm, 30–60 cm)
- **11 terrain** — elevation, slope, northness/eastness, and derivatives
- **3 missingness indicators**

**No vegetation layers.** NDVI and land cover are blocked by an assertion, not
convention — see `../DATA_OVERVIEW.md` for the measurement behind that, and
`satellite_experiment/results/` for the numbers themselves.

---

## Results (metropolitan France, 96 species)

Cross-validated over 10 spatial blocks and 5 folds. The served run is the first
column; the advisory prefers it by name rather than by best AUC, because the
three 1 km runs agree to three decimal places on AUC and an automatic
"highest AUC" rule would swing between them on numerical noise.

| | 1 km, **served** | 1 km | 18.5 km |
|---|---|---|---|
| run | `fra_30s_96sp_bg1km` | `fra_30s_96sp` | `fra_10m_96sp` |
| AUC | **0.747** | 0.747 | 0.734 |
| TSS | **0.334** | 0.326 | 0.318 |
| Boyce | **0.882** | 0.850 | 0.782 |

The served run differs only in its background draw: it is trained against a
target-group background built from the 1 km recording-effort surface, which
carries 42% less built-up separation from the presences than the 18.5 km one
and so less GBIF survey bias. AUC is unchanged, TSS and Boyce improve.

AUC ≈ 0.75 means: given one site where a species does grow and one where it
does not, the model picks correctly about three times in four.

---

## How long things take

On the AMD Instinct MI210, France, 96 species.

| Step | Time |
|---|---|
| `--smoke` (8 species, pipeline check) | **5 s** |
| Train 18.5 km (2,526 rows) | **14 s** |
| Train 1 km (252,167 rows) | **~2.5 min** |
| Same, inside the ROCm container | 121 s |
| Score the whole country + write GeoTIFFs | seconds, inside the above |
| Advisory startup (load model, open 61 rasters, build reference) | 8 s cold, ~1 s cached |
| One coordinate query | **4–54 ms** |
| 24 concurrent queries | 108 ms |

Training cost is dominated by reading rasters, not by the network — 62,688
parameters is nothing for this GPU. The 18-fold jump from 18.5 km to 1 km
tracks the row count (2.5k → 252k), not model size.

Convergence differs by resolution and is worth knowing: **1 km folds stop at
epochs 12–51, 18.5 km folds at 84–195.** The coarse run needs far more passes
because it has so little data. The old 60-epoch default was cutting it off
before it converged and making it look worse than it was — always pass
`--epochs 400`.

Data preparation is the slow part, not training:

| | Time |
|---|---|
| Rasters for one country (France) | tens of minutes |
| Rasters for the USA (~15× the area) | hours |
| GBIF species download | **the bottleneck** — ~25 s per species, degrading to ~3 min once GBIF starts throttling |

The species download is resumable from checkpoints, which is why France
stopped at 96 of 148 species rather than failing.

---

## Where it works well

- **Species with a range boundary inside the study area.** Kermes oak 0.90,
  Aleppo pine 0.89, cork oak 0.94, Swiss stone pine 0.84. The model finds the
  climatic edge.
- **Rare species.** Joint training means 81 records is enough. Nothing was
  unevaluable for lack of training data, against 134 species-folds in the
  earlier global run.
- **Speed.** 2.5 minutes to train, milliseconds to query.
- **Honest under spatial CV.** The score survives holding out whole regions.

## Where it does badly

- **Species that occupy the whole country.** Hawthorn 0.59, ash 0.60,
  sycamore 0.61. Not a model fault: with no gradient inside the extent there
  is nothing to learn. AUC correlates **−0.408** with range size — under 250
  cells averages 0.898, over 500 cells averages 0.712. **Unfixable without
  widening the study area.**
- **Cultivated species.** Apple 0.60, pear 0.64 — orchards reflect farming
  decisions, not climate.
- **Cropland.** Boyce falls from 0.921 overall to 0.675 on cropland, which is
  the land an afforestation advisory most needs to rank.
- **Thresholds do not transfer.** TSS 0.326 at the training threshold against
  0.398 re-optimised on held-out folds.
- **It answers the wrong question, slightly.** Trained on adult trees found
  growing, so it predicts where a species *occurs*, not whether a planted
  sapling survives its first two summers.
- **One country, one climate window.** No dispersal, no soil–climate
  interaction terms, no 2050 projection yet.

## What has been ruled out by measurement

- **More capacity.** 10× the parameters (1024×512) moved AUC by +0.002 and
  overfitted sooner. 256×128 stays.
- **Satellite data.** The +0.012 gain was two-thirds survey bias from built-up
  fraction, not ecology.
- **Soil moisture.** Exactly +0.000 across every stratum.

---

## Reading the metrics

**AUC** describes the model's skill for a species, not the site. 0.5 is a coin
flip.

**Boyce** is not comparable across species: it runs *opposite* to AUC on range
size (0.466 for restricted species, 0.891 for widespread ones), because for a
species present almost everywhere nearly all land genuinely is suitable. **Run
level Boyce means are inflated by widespread species** — every run prints this
warning with its own breakdown.

**TSS** depends on a threshold, and the threshold does not transfer well.
