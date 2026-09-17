# Do not train on anything in this directory

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
