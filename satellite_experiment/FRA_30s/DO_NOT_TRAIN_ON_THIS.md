# Do not train on anything in this directory

Same rule as `country_data/FRA/satellite_download/`, and for the same reason.
This directory only differs in being on the 1 km predictor grid, which makes it
*more* dangerous, not less: these files can be stacked with the predictors
without anything complaining about shape.

They exist for one controlled experiment, `deep_sdm_satellite_experiment.py`,
which measures what happens when satellite layers are used as predictors and in
particular whether any gain survives on land with no existing tree cover. The
production path (`deep_sdm_training.py` + `sdm_features.py`) must never read
them: `sdm_features.assert_no_vegetation_layers()` blocks this directory twice,
on the path token `satellite` and on this marker, and only the explicit
`ALLOW_VEGETATION_PREDICTORS_EXPERIMENT` opt-in lifts the block.

`landcover_tree_frac.tif` and `landcover_cropland_frac.tif` have a second,
legitimate job: they define the tree-cover strata the experiment evaluates on.
Used that way they are evaluation metadata, not inputs.
