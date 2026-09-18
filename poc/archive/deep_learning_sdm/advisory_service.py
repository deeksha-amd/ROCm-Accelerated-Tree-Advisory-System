"""
Serving layer for the deep SDM — coordinates in, ranked planting advice out

What this is
------------
The domain half of the web advisory. `advisory_app.py` is the HTTP shell; this
module owns everything that can be wrong about the answer. It imports
`sdm_features` and `deep_sdm_training` and edits neither: the feature catalog,
the nodata handling, the scaler and the network class all come from the
training code, so a served prediction is built the same way a cross-validated
one was.

Why the checkpoint is not enough on its own
-------------------------------------------
`deep_sdm.pt` will happily return 96 numbers for Tokyo. Four things stand
between the raw head outputs and something honest enough to show a
non-expert, and each is a separate failure mode:

1. **Coverage.** The French rasters are a *bounding box*, not the country.
   bio_1 is a perfectly finite 17.5 degC at Barcelona and 12.2 at London,
   because both sit inside (-5.5, 41.0, 10.0, 51.5). Testing "is the raster
   finite here" would therefore have silently advised on Spain, Italy, Belgium
   and southern England using a model trained on French occurrences alone. The
   test is instead a rasterised Natural Earth country polygon, taken from the
   copy `country_clip.py` already cached. Regions are discovered from
   `deep_sdm_output/`, so a checkpoint for a new country appears on its own.

2. **The exclusion mask is applied to output.** `apply_exclusions` is reused
   verbatim, so bits 1|2|4|8 blank the answer and bits 16|32 become notes.
   The mask is only published at 10 arcmin, so its verdict is inherited by
   every 1 km cell within roughly 18 km; `EXCLUSION_NOTES` words each message
   to that precision and never tighter, and `override_exclusion` exists
   because a coarse "built up" flag over a whole valley is a reason to warn,
   not a reason to refuse to answer.

3. **The widespread-species trap.** AUC correlates -0.41 with occupancy here.
   Hawthorn (0.605), hazel (0.591), sycamore (0.613) and ash (0.615) score
   near chance *because they grow throughout France*, so there is no gradient
   to learn — which is close to the opposite of the reason a species should be
   ranked last. A single list sorted by score therefore buries exactly the
   trees a non-expert is most likely to want under narrow Mediterranean
   specialists that the model can separate cleanly. `bucket_species()` splits
   the answer instead of sorting it: species the model cannot discriminate but
   which occupy most of the region go in their own group, labelled as reliable
   *because* they are everywhere, and species the model genuinely cannot speak
   to are shown apart from both.

4. **Scores are not probabilities.** Each head is trained with its own capped
   `pos_weight`, so 0.9 for one species and 0.9 for another are not the same
   statement, and neither is a survival chance. Every number shown to a user
   is therefore a **percentile against that species' own score distribution
   over the covered region** (`ReferenceScores`), which is comparable across
   species and is honestly relative. The raw index is still returned, labelled
   as an index. The max-TSS threshold is reported as a flag rather than a
   verdict: it transfers poorly, TSS 0.326 at the training threshold against
   0.398 re-optimised on the held-out fold.

What is deliberately not done
-----------------------------
No vegetation layer enters the predictors. The catalog comes from
`feat.build_catalog`, which hard-stops on them, `ALLOW_VEGETATION_PREDICTORS_EXPERIMENT`
is asserted off at load, and a checkpoint whose run summary admits to
vegetation inputs is refused outright — an experiment model must not be able
to end up behind a public endpoint by being copied into the output directory.
"""

import json
import os
import re
import threading
import warnings

import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.windows import Window

import sdm_features as feat
from deep_sdm_training import MultiSpeciesSDM

# ─────────────────────────────────────────────────────
# CONFIG — where the trained models and boundaries live
# ─────────────────────────────────────────────────────
# Curated species lists live beside the code in deep_learning_sdm/, so they are
# anchored to this module's directory. Everything else below is data and stays
# relative to the working directory, which is the repo root.
HERE = os.path.dirname(os.path.abspath(__file__))

OUTPUT_ROOT = "deep_sdm_output"
CHECKPOINT_NAME = "deep_sdm.pt"
SUMMARY_NAME = "run_summary.json"
SPECIES_METRICS_NAME = "cv_metrics_by_species.csv"

# The runs quoted in MODEL.md. Naming them beats picking by best
# AUC: fra_30s_96sp, _w512x256 and _w1024x512 agree to four decimal places, so
# an automatic "highest mean AUC" rule would swing between width experiments on
# numerical noise. Any profile not listed here still gets served — it just
# falls back to the discovery order below.
# fra_30s_96sp_bg1km leads: same AUC as fra_30s_96sp (0.747) with better TSS
# (0.334 vs 0.326) and Boyce (0.882 vs 0.850), trained against a background
# whose built-up separation from the presences is 42% lower, so it carries less
# GBIF survey bias.
PREFERRED_RUNS = ("fra_30s_96sp_bg1km", "fra_30s_96sp",
                  "esp_30s_96sp", "usa_30s_96sp")
# Never serve these: smoke tests and anything that smells like an experiment.
SKIP_RUNS = ("smoke",)

# Cached by country_clip.py. Read only; no download is attempted, because a
# serving process should not reach for the network to answer a request.
BOUNDARIES = "country_src_cache/natural_earth/ne_10m_admin_0_countries.geojson"
# ISO_A3 is "-99" for France in this release, which is exactly the trap
# country_clip.py documents, so the fallbacks are not optional.
ISO_KEYS = ("ISO_A3", "ISO_A3_EH", "ADM0_A3", "SOV_A3", "GU_A3", "SU_A3")
NAME_KEYS = ("NAME_LONG", "NAME_EN", "NAME", "ADMIN", "SOVEREIGNT")
# Profile template paths look like country_data/FRA/climate_30s/... .
COUNTRY_DIR_PATTERN = re.compile(r"country_data[/\\]([A-Za-z]{3})[/\\]")

# Common names and use categories. All of these belong to the species-data
# pipeline and are read defensively: a sibling process rewrites them, so a
# half-written file must degrade to "no common name", never to a stack trace.
NAME_SOURCES = ("species_occurrences/FR/species_plan.json",
                "species_occurrences/ES/species_plan.json",
                "species_occurrences/US/species_plan.json")
NAME_CSV_SOURCES = tuple(os.path.join(HERE, name) for name in
                         ("tree_species_list.csv", "tree_species_list_ES.csv",
                          "tree_species_list_US.csv"))
# GBIF renamed three of the 96 trained species after the plans were written —
# Sorbus torminalis is now Torminalis glaberrima, Sorbus domestica is Cormus
# domestica, Eriobotrya japonica is Rhaphiolepis bibas — so the model's species
# names do not appear in any list and those trees would show as bare Latin. The
# caches key the new name to a GBIF usageKey and the plans key the old name to
# the same number, which bridges them without a hand-written synonym table.
TAXONOMY_SOURCES = ("species_occurrences/FR/taxonomy_cache.json",
                    "species_occurrences/ES/taxonomy_cache.json",
                    "species_occurrences/US/taxonomy_cache.json")
USE_LABELS = {"native_woodland": "Native woodland", "timber": "Timber",
              "fruit_nut": "Fruit and nuts", "street_park": "Street and park",
              "ornamental": "Ornamental", "agroforestry": "Agroforestry"}

# The curated species list carries free-text notes, and five of them say a
# species is invasive — one in so many words: "highly invasive; modelled for
# control not planting". The SDM has no opinion about this and scores Tree of
# heaven and black locust as highly as anything else, so a page that only
# reported model output would tell a user to plant them. These are matched in
# the notes and moved out of the recommendation groups entirely.
DO_NOT_PLANT_TOKENS = ("invasive",)
CAUTION_TOKENS = ("dieback", "disease", "canker", "blight")

CACHE_DIR = "advisory_cache"

# ─────────────────────────────────────────────────────
# CONFIG — scoring and presentation
# ─────────────────────────────────────────────────────
# "auto" matches deep_sdm_training.py's convention, and `resolve_device` below
# resolves it the same way. It used to be a hardcoded "cpu", which the
# container health check then reported as the truth even on a machine with a
# working MI210. Single-point inference on a 64-wide MLP is not a GPU workload,
# so pass --device cpu explicitly when the card is busy training.
DEVICE = "auto"
# Cells drawn once at startup to give every species a reference distribution to
# be a percentile of. 20k of ~930k French land cells puts the standard error of
# a percentile at well under one point, and the whole matrix is 96 x 20000
# float32 = 7.7 MB, so the sorted scores are kept in full and percentiles are
# exact lookups rather than interpolations off a quantile grid.
REFERENCE_CELLS = 20000
REFERENCE_SEED = 42

# A species is "widely suited" when the model cannot separate good sites from
# bad for it *and* it occupies much of the region — the second half is what
# distinguishes "everywhere, so unlearnable" from "genuinely not modelled".
# 0.70 AUC is the point the project's own numbers break at: under 250 presence
# cells averages 0.898, over 500 averages 0.712.
RELIABLE_AUC = 0.70          # at or above this, ranking by score is meaningful
GOOD_AUC = 0.80              # at or above this, the model discriminates well
WIDESPREAD_PRESENCES = 400   # per-fold presence cells; the region-wide species

# ─────────────────────────────────────────────────────
# CONFIG — plain language
# ─────────────────────────────────────────────────────
# The two numbers this advisory computes are a percentile and an AUC, and both
# were being printed raw. They answer completely different questions and a
# reader has no way to know that from "p 75.3  AUC 0.641": the percentile is a
# property of the *location*, the AUC is a property of the *model's knowledge
# of that species* and does not change from one place to another. Worse, a
# percentile next to a decimal invites reading it as a survival chance, which
# it emphatically is not. So each is turned into a sentence here, and the
# figures move to a labelled block at the end for whoever wants them.

# How the location rates for a species. (floor, key, short phrase, sentence).
# Deliberately not "suitable / unsuitable": the max-TSS threshold moves from
# 0.326 to 0.398 TSS when re-optimised out of sample, so a hard cut would claim
# a precision the validation does not support.
BANDS = (
    (90, "excellent", "one of the best spots in the region for this tree",
     "Compared with everywhere else in the region, very few places suit this "
     "tree better than here."),
    (75, "good", "better here than in most of the region",
     "This spot suits this tree better than three quarters of the region."),
    (50, "fair", "about average for the region",
     "This spot is unremarkable for this tree — neither notably good nor "
     "notably bad."),
    (25, "marginal", "below average for the region",
     "Most of the region suits this tree better than this spot does."),
    (0, "poor", "poor here",
     "This is among the least suitable parts of the region for this tree."),
)

# How much the model actually knows about a species, from its held-out AUC.
# This is the line that stops a reader treating a low number as "bad tree".
CONFIDENCE = {
    "high": ("knows this tree well",
             "the model knows this tree well",
             "Tested on parts of the region it was never trained on, the "
             "model told good sites from bad for this tree reliably."),
    "moderate": ("a fair idea about this tree",
                 "the model has a fair idea about this tree",
                 "The model gets this tree right more often than not, so "
                 "treat its position in the list as a steer rather than a "
                 "verdict."),
    "widespread": ("grows region-wide, so no spot stands out — usually a safe bet",
                   "this tree grows almost everywhere in the region, so the "
                   "model cannot pick out good spots for it",
                   "The model scores near guesswork for this tree, and the "
                   "reason is encouraging rather than worrying: there is "
                   "nowhere in the region it fails, so there was no contrast "
                   "for the model to learn. It is usually a safe choice."),
    "low": ("struggles with this tree — rough guide only",
            "the model struggles with this tree, so treat it as a rough guide",
            "The model scores near guesswork for this tree and it is not "
            "region-wide either, so its position here means little."),
    "unknown": ("no reliability score for this tree",
                "there is no reliability score for this tree in this run",
                "This species was not scored during cross-validation, so "
                "there is no measure of how well the model knows it."),
}

# The group names were jargon too. "site-matched" and "widely-suited" are
# accurate and mean nothing to a reader, so the split is described by what it
# is for instead.
GROUP_LABELS = {
    "well_matched": "Especially well suited to this exact spot",
    "general": "Safe general choices — these do well across most of the region",
    "invasive": "Do not plant these — flagged as invasive",
    "unrankable": "The model cannot rank these here",
}
GROUP_BLURBS = {
    "well_matched":
        "The model can tell good sites from bad for these species, and it "
        "rates this spot highly for them.",
    "general":
        "These grow throughout the region, so the model cannot say this spot "
        "is special for them — but that is precisely why they are dependable "
        "choices rather than a reason against them.",
    "invasive":
        "The model scores these as well as anything else, because it knows "
        "nothing about invasiveness. The species list flags them, so they are "
        "kept out of the recommendations and shown here as a warning.",
    "unrankable":
        "The model scores near guesswork for these and they are not "
        "region-wide either, so there is no reason to trust any ordering.",
}

# One-line key for the figures block, so the numbers are explained where they
# appear rather than in a document nobody opens.
METRICS_KEY = (
    "rating = where this spot sits among all places in the region for that "
    "tree, 0-100 (100 = nowhere better). confidence = the model's held-out "
    "AUC for that species, 0.5 = guesswork, 1.0 = perfect; it describes the "
    "model, not the place. raw score = the network's own output, useful only "
    "for comparing the same species between places.")

# ─────────────────────────────────────────────────────
# CONFIG — how much to show
# ─────────────────────────────────────────────────────
# A whole answer has to fit on one screen. Five is the cap across the *whole*
# recommendation, not per group: the previous layout showed four general plus
# five well-matched plus an invasive block and read like a database dump.
TOP_RECOMMENDATIONS = 5
# Of those five, how many come from the well-matched group. The well-matched
# species are the interesting answer, so they get the majority; but handing a
# user five obscure Mediterranean specialists and no everyday tree is the
# failure this split exists to prevent, so two slots are reserved for the
# dependable region-wide species. Either share backfills from the other when
# one group is short.
WELL_MATCHED_SHARE = 3
# Invasives never consume a recommendation slot. They are only worth raising at
# all when they score well enough here that a user might otherwise have picked
# one, so a poor-scoring invasive is silently left out of the warning.
INVASIVE_WARNING_PERCENTILE = 75
NEARBY_SUGGESTIONS = 3        # alternative locations offered
NEARBY_SPECIES_SHOWN = 3      # species listed per alternative, kept short

# Wording for the mask bits. Every hard-exclusion message has to survive the
# fact that the flag came from an 18.5 km cell, so none of them says "here".
EXCLUSION_NOTES = {
    "ocean": ("Open sea",
              "The 18.5 km cell containing this point is classified as ocean."),
    "inland_water": ("Lake or river",
                     "The 18.5 km cell containing this point is classified as "
                     "inland water."),
    "permanent_snow_ice": ("Permanent snow or ice",
                           "The 18.5 km cell containing this point is "
                           "classified as permanent snow or ice."),
    "built_up": ("Built-up area",
                 "The 18.5 km cell containing this point is classified as "
                 "built up. At that resolution the flag covers everything "
                 "within roughly 18 km of the town it was raised for, so a "
                 "garden or verge nearby may well be plantable."),
}
ADVISORY_NOTES = {
    "closed_canopy_forest": (
        "Already closed-canopy forest",
        "Within roughly 18 km this is mapped as closed-canopy forest. That is "
        "not a reason against these trees — it is land that has already proven "
        "it grows them, and simply may not need planting."),
    "cropland": (
        "Farmland",
        "Within roughly 18 km this is mapped as cropland. It is plantable, but "
        "it is someone's field; hedgerows and field margins are the usual "
        "answer here."),
}

# ─────────────────────────────────────────────────────
# CONFIG — "nothing worth planting here" and the nearby search
# ─────────────────────────────────────────────────────
# When is a location not worth planting? Measured rather than guessed. Over
# 3,000 random covered cells, the best non-invasive species at a cell sits at
# the 97th percentile of its own range at the median cell, the 87th at the
# tenth-worst percentile of cells, and the 79th at the fifth. Nothing above the
# 60th percentile happens at only 1.0% of cells, so that is the bar: it fires
# on genuinely poor ground and stays silent on ordinary ground. A location that
# clears it is not "good", it simply has at least one species that is not
# unusual for it, which is as much as a relative score can honestly say.
WORTH_PLANTING_PERCENTILE = 60.0
# The percentile bar alone is not enough, and the nearby search is what exposed
# it. A percentile is relative to one species' own spread, so a Mediterranean
# specialist whose scores are near zero across most of France still has a 60th
# percentile somewhere in the north: olive at 48.28N, 1.63E rates p61 on a raw
# score of 0.118, and calling that "about average for the region" would be
# true of the ranking and nonsense as advice. So a location also has to have
# at least one species clearing its own trained cut-off. Combining the two
# moves the share of covered cells with nothing worth planting from 0.6% to
# 1.3% — still rare, and now rare for the right reason. The cut-off transfers
# poorly between regions (TSS 0.326 against 0.398 re-optimised), which is why
# it is used to route and to order rather than shown as a verdict.
REQUIRE_MODEL_CUTOFF = True

# Search geometry. Bounded hard at 25 km: far enough to step out of a lake, a
# city's 18.5 km mask cell or a stretch of coast, close enough that the
# suggestion is still the same place in any useful sense. Candidates are taken
# on a 3 km lattice rather than every 1 km cell, because adjacent cells share
# almost all their climate and soil and scoring 2,000 near-identical points to
# choose three would be wasted work.
NEARBY_MAX_KM = 25.0
NEARBY_STRIDE_KM = 3.0
# Accepted suggestions must be this far from each other, so the answer is three
# genuinely different places rather than three cells in one field.
NEARBY_MIN_SEPARATION_KM = 6.0
# Ceiling on how many candidates get scored, nearest first. One batched forward
# pass, so the whole search costs about one extra point query.
NEARBY_MAX_CANDIDATES = 48

COMPASS = ("north", "north-north-east", "north-east", "east-north-east",
           "east", "east-south-east", "south-east", "south-south-east",
           "south", "south-south-west", "south-west", "west-south-west",
           "west", "west-north-west", "north-west", "north-north-west")

# The nearby search can only ever say "our coarse filter does not exclude
# this", which is a much weaker claim than "you may plant here", and the filter
# is coarse enough that the difference matters. Attached to every suggestion.
NEARBY_CAVEAT = (
    "These are the nearest places the model covers that its land-use filter "
    "does not rule out. That filter works on an 18.5 km grid and flags only "
    "0.4% of France as built up, so 'not ruled out' is a long way from "
    "'available to plant on' — each of these could still be a garden, a road "
    "or private farmland. Check the actual place, and whose it is.")

# Shown on every answer. The model is fitted to adult-tree sightings, which is
# a different question from whether a sapling survives its first summer.
STANDING_CAVEATS = [
    "Scores are relative, not probabilities. They say how this site compares "
    "with the rest of the covered region for that species — not how likely a "
    "given tree is to survive.",
    "The model is fitted to records of adult trees found growing wild or "
    "planted. It predicts where a species is found, not whether an unwatered "
    "sapling makes it through its first two summers.",
    "The water, city and forest filter comes from an 18.5 km grid and misses "
    "far more than it catches: Paris, Lyon, Marseille and Toulouse are "
    "flagged as built up, Bordeaux, Nantes, Lille and Strasbourg are not. "
    "Where it does flag something the flag covers everything within roughly "
    "18 km, so it is a prompt to look at the actual spot, not a survey of it.",
]


def divider(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def resolve_device(requested=DEVICE):
    """Resolve "auto" the way deep_sdm_training.pick_device does.

    Reimplemented in three lines rather than imported because `pick_device`
    prints a banner, and a one-shot coordinate lookup should print the answer
    and nothing else. The resolution rule is identical, including the fact
    that on this ROCm build the AMD GPU lives behind the `cuda` namespace.
    """
    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# ─────────────────────────────────────────────────────
# GEOMETRY — distance and direction in words
# ─────────────────────────────────────────────────────
def distance_km(lat1, lon1, lat2, lon2):
    """Equirectangular distance. Exact enough well inside the 25 km search."""
    mean_lat = np.radians((np.asarray(lat1) + np.asarray(lat2)) / 2.0)
    east = (np.asarray(lon2) - np.asarray(lon1)) * np.cos(mean_lat) * 111.320
    north = (np.asarray(lat2) - np.asarray(lat1)) * 110.574
    return np.hypot(east, north)


def bearing_degrees(lat1, lon1, lat2, lon2):
    mean_lat = np.radians((lat1 + lat2) / 2.0)
    east = (lon2 - lon1) * np.cos(mean_lat)
    north = lat2 - lat1
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def compass_name(bearing):
    return COMPASS[int((bearing % 360.0) / 22.5 + 0.5) % 16]


def plain_direction(lat1, lon1, lat2, lon2):
    """"about 12 km north-east" — the form a person can act on."""
    km = float(distance_km(lat1, lon1, lat2, lon2))
    where = compass_name(bearing_degrees(lat1, lon1, lat2, lon2))
    if km < 1.5:
        return f"under 2 km {where}", km, where
    return f"about {km:.0f} km {where}", km, where


# ─────────────────────────────────────────────────────
# COMMON NAMES
# ─────────────────────────────────────────────────────
def load_species_names(sources=NAME_SOURCES, csv_sources=NAME_CSV_SOURCES,
                       taxonomy_sources=TAXONOMY_SOURCES):
    """Scientific name -> {common_name, use, notes}, from whatever is readable.

    Notes are **accumulated across sources, never overwritten**, and that is
    not a stylistic choice. The per-country plans carry country-local notes:
    the Spanish plan says Acacia dealbata is "invasive in Galicia", the US plan
    says Ailanthus is "invasive in the US", and the US plan lists Robinia
    pseudoacacia — a North American native — with no note at all. Letting the
    last source win therefore erased "nitrogen fixer; invasive in Europe" from
    black locust and put it back among the trees this page suggests planting in
    France. A species flagged anywhere stays flagged.

    Every source here is owned by the species-data pipeline and may be being
    rewritten right now, so each is tried independently and a failure costs at
    most the names from that one file.
    """
    names = {}

    def absorb(key, common_name, use, note):
        key = str(key).strip()
        if not key:
            return
        entry = names.setdefault(key, {"common_name": "", "use": "",
                                       "notes": [], "seen": set()})
        # First non-empty wins for the label fields: the curated European list
        # is read first and is the right voice for a French advisory.
        entry["common_name"] = entry["common_name"] or _clean(common_name)
        entry["use"] = entry["use"] or _clean(use)
        # Deduplicate at the clause level, not the whole-note level. Sources
        # already carry semicolon-joined notes and they overlap in part: the
        # European list says "accepted name for Prunus dulcis", the US list
        # says "orchard crop in California; GBIF moved Prunus dulcis here",
        # and a plan contributes "GBIF moved Prunus dulcis here" on its own.
        # Comparing whole strings finds no duplicate and almond ends up
        # telling you twice that GBIF moved it. Order is preserved, so the
        # curated European note still leads.
        for clause in str(_clean(note)).split(";"):
            clause = clause.strip()
            if clause and clause.lower() not in entry["seen"]:
                entry["seen"].add(clause.lower())
                entry["notes"].append(clause)

    for path in csv_sources:
        try:
            import pandas as pd
            # on_bad_lines="skip": these files are being rewritten by another
            # process, so a malformed row means "caught it mid-save", not
            # "give up on the whole file".
            frame = pd.read_csv(path, on_bad_lines="skip", engine="python")
            for record in frame.to_dict("records"):
                absorb(record.get("species", ""), record.get("common_name"),
                       record.get("use"), record.get("notes"))
        except Exception as error:                       # noqa: BLE001
            warnings.warn(f"common names: {path} unreadable ({error})")

    by_gbif_key = {}
    for path in sources:
        try:
            with open(path, encoding="utf-8") as handle:
                plan = json.load(handle)
            for record in plan.get("species", []):
                for key in (record.get("species"),
                            record.get("accepted_name")):
                    absorb(key or "", record.get("common_name"),
                           record.get("use"), record.get("notes"))
                if record.get("key"):
                    by_gbif_key.setdefault(int(record["key"]), record)
        except Exception as error:                       # noqa: BLE001
            warnings.warn(f"common names: {path} unreadable ({error})")

    # Bridge the renamings: a cached taxonomy entry gives the current accepted
    # binomial and its usageKey, and the plans give the old binomial under the
    # same key, so the pair identifies the same tree without anyone typing a
    # synonym list that would then go stale.
    for path in taxonomy_sources:
        try:
            with open(path, encoding="utf-8") as handle:
                cache = json.load(handle)
            for name, record in cache.items():
                plan_record = by_gbif_key.get(record.get("usageKey"))
                if plan_record:
                    absorb(name, plan_record.get("common_name"),
                           plan_record.get("use"), plan_record.get("notes"))
        except Exception as error:                       # noqa: BLE001
            warnings.warn(f"taxonomy bridge: {path} unreadable ({error})")

    for entry in names.values():
        entry["notes"] = "; ".join(entry["notes"])
        entry.pop("seen", None)
    return names


def _clean(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none") else text


# ─────────────────────────────────────────────────────
# COUNTRY POLYGONS
# ─────────────────────────────────────────────────────
def load_country_geometry(iso_a3, path=BOUNDARIES):
    """Natural Earth geometry for a 3-letter code, or None if unavailable.

    Mirrors `country_clip.load_country`'s key preference rather than importing
    it: that module belongs to the data pipeline and is under active edit, and
    a serving process should not fall over because a script it only needed
    twenty lines of is mid-save.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        collection = json.load(handle)

    wanted = iso_a3.upper()
    best = None
    for feature in collection.get("features", []):
        properties = feature.get("properties", {})
        rank = next((i for i, key in enumerate(ISO_KEYS)
                     if str(properties.get(key, "")).upper() == wanted), None)
        if rank is not None and (best is None or rank < best[0]):
            best = (rank, feature)
    if best is None:
        return None
    properties = best[1]["properties"]
    name = next((properties[k] for k in NAME_KEYS if properties.get(k)), wanted)
    return {"iso": wanted, "name": str(name), "geometry": best[1]["geometry"]}


def coverage_from_geometry(geometry, grid):
    """Rasterise a country polygon onto the profile grid.

    all_touched=True so that coastal and border cells count as covered; a
    strict centroid test loses most of a narrow coastline, and the cells it
    wrongly adds are caught downstream by the ocean bit or by missing
    predictors.
    """
    return rasterize([(geometry, 1)], out_shape=(grid["height"], grid["width"]),
                     transform=grid["transform"], fill=0, all_touched=True,
                     dtype="uint8").astype(bool)


class GlobalLandCover:
    """The exclusion mask on its own 18.5 km global grid.

    Separate from the per-region copy because it answers a different question:
    a point outside every trained region gets no species list either way, but
    "no model covers the middle of the Bay of Biscay" is a much worse answer
    than "that is open sea". The raster writes 255 for cells with no land-cover
    data, which over water is essentially all of them — reported as "sea or
    unmapped" rather than flatly as ocean, because 255 is an absence of
    evidence.
    """

    def __init__(self, path=None):
        path = path or feat.GRID_PROFILES["global_10m"]["exclusion_mask"]
        self.path = path
        self.available = os.path.exists(path)
        if not self.available:
            warnings.warn(f"{path} not found; out-of-coverage points cannot be "
                          f"identified as sea")
            return
        with rasterio.open(path) as src:
            self.band = src.read(1)
            self.transform = src.transform
            self.width, self.height = src.width, src.height
            self.nodata = src.nodata

    def at(self, lat, lon):
        """Raw mask byte, or None off the raster."""
        if not self.available:
            return None
        col = int(np.floor((lon - self.transform.c) / self.transform.a))
        row = int(np.floor((lat - self.transform.f) / self.transform.e))
        if not (0 <= row < self.height and 0 <= col < self.width):
            return None
        return int(self.band[row, col])

    def describe(self, lat, lon):
        """A one-line land-use reading for a point anywhere on Earth."""
        value = self.at(lat, lon)
        if value is None:
            return None
        if self.nodata is not None and value == int(self.nodata):
            return ("Mapped as sea, or with no land-cover data at all, on the "
                    "18.5 km global grid — over water that is the same thing.")
        named = [name for name, bit in feat.EXCLUSION_BITS.items()
                 if value & bit]
        if not named:
            return "Mapped as open, unbuilt land on the 18.5 km global grid."
        return ("Mapped on the 18.5 km global grid as: "
                + ", ".join(n.replace("_", " ") for n in named) + ".")


def coverage_from_template(grid):
    """Fallback coverage: wherever the profile's template raster has data.

    Only correct for a profile that is genuinely clipped to its subject, which
    the French block is not. Used for global profiles and flagged as loose.
    """
    with rasterio.open(grid["template"]) as src:
        return np.isfinite(feat.read_band(src))


# ─────────────────────────────────────────────────────
# REFERENCE DISTRIBUTION
# ─────────────────────────────────────────────────────
class ReferenceScores:
    """Per-species score distribution over the covered region.

    The reason this exists rather than showing the sigmoid directly: each head
    was trained with its own `pos_weight`, capped at POS_WEIGHT_CAP, so the
    heads are on different scales and a raw 0.8 means something different for
    a species with 150 presences than for one with 680. Ranking those raw
    numbers against each other is the kind of arithmetic that produces a
    confident wrong answer. A percentile against the species' own distribution
    is comparable across species and is, accurately, a relative statement.
    """

    def __init__(self, sorted_scores, n_cells):
        self.sorted_scores = sorted_scores        # [n_species, n_cells]
        self.n_cells = int(n_cells)

    def percentile(self, scores):
        """Where each species' score at one site falls in its own spread."""
        out = np.empty(len(scores), dtype="float32")
        for i, value in enumerate(scores):
            position = np.searchsorted(self.sorted_scores[i], value)
            out[i] = 100.0 * position / max(self.sorted_scores.shape[1], 1)
        return out

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, sorted_scores=self.sorted_scores,
                            n_cells=self.n_cells)

    @classmethod
    def load(cls, path, n_species):
        data = np.load(path)
        scores = data["sorted_scores"]
        if scores.shape[0] != n_species:
            raise ValueError(f"cached reference has {scores.shape[0]} species, "
                             f"model has {n_species}")
        return cls(scores, int(data["n_cells"]))


# ─────────────────────────────────────────────────────
# ONE SERVED REGION
# ─────────────────────────────────────────────────────
class RegionModel:
    """A profile, its trained checkpoint, its coverage and its mask.

    Everything expensive happens in `__init__` — raster handles are opened and
    kept, the mask and the coverage polygon are held as arrays, and the
    reference distribution is computed or read from cache — so a point query is
    61 one-pixel reads and a 64-wide forward pass.
    """

    def __init__(self, run_dir, device=DEVICE, reference_cells=REFERENCE_CELLS,
                 cache_dir=CACHE_DIR, verbose=True):
        self.run_dir = run_dir
        self.name = os.path.basename(run_dir.rstrip("/"))
        self.device = resolve_device(device)
        self.verbose = verbose
        self._lock = threading.Lock()

        self.summary = _read_json(os.path.join(run_dir, SUMMARY_NAME)) or {}
        # A model fitted with vegetation predictors is circular for this
        # problem and must never reach a user, even if someone copies it into
        # the output directory. Refuse rather than warn.
        if int(self.summary.get("vegetation_layers_used", 0)) > 0:
            raise ValueError(
                f"{run_dir} was fitted with "
                f"{self.summary['vegetation_layers_used']} vegetation "
                f"predictor(s); that model is an experiment and is circular "
                f"for planting advice, so it will not be served")

        checkpoint = torch.load(os.path.join(run_dir, CHECKPOINT_NAME),
                                map_location="cpu", weights_only=False)
        for key in ("state_dict", "species", "feature_names", "scaler"):
            if key not in checkpoint:
                raise ValueError(f"{run_dir}/{CHECKPOINT_NAME} has no {key!r}; "
                                 f"it is not a deep_sdm_training checkpoint")

        self.profile = self.summary.get("profile") or feat.DEFAULT_PROFILE
        if self.profile not in feat.GRID_PROFILES:
            raise ValueError(f"{run_dir} names profile {self.profile!r}, which "
                             f"sdm_features does not define")
        self.species = list(checkpoint["species"])
        self.climate_window = checkpoint.get("climate_window",
                                             feat.DEFAULT_CLIMATE_WINDOW)

        self.grid = feat.load_grid(self.profile)
        self.catalog = feat.build_catalog(self.profile,
                                          climate_window=self.climate_window)
        self.groups = feat.feature_groups(self.catalog)
        self.n_predictors = len(self.groups)
        expected = feat.feature_names(self.catalog) + \
            [f"missing_{g}" for g in sorted(set(self.groups))]
        # A silent column shift would misread every predictor by one position
        # and still return plausible numbers, so this is a hard stop.
        if expected != list(checkpoint["feature_names"]):
            raise ValueError(
                f"{run_dir}: the predictors on disk no longer match the ones "
                f"the checkpoint was trained on "
                f"({len(expected)} vs {len(checkpoint['feature_names'])} "
                f"columns). Retrain or restore the rasters.")

        self.scaler = feat.FeatureScaler()
        self.scaler.median = np.asarray(checkpoint["scaler"]["median"])
        self.scaler.mean = np.asarray(checkpoint["scaler"]["mean"])
        self.scaler.scale = np.asarray(checkpoint["scaler"]["scale"])

        self.model = MultiSpeciesSDM(
            len(expected), len(self.species),
            hidden=list(checkpoint.get("hidden_sizes", [256, 128])))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval().to(self.device)

        self.thresholds = np.asarray(
            checkpoint.get("thresholds", np.full(len(self.species), np.nan)),
            dtype="float32")
        self.threshold_strategy = checkpoint.get("threshold_strategy", "unknown")

        self.country = self._resolve_country()
        self.coverage, self.coverage_kind = self._build_coverage()
        self.mask, _ = feat.read_exclusion_mask(self.grid)
        self.metrics = self._load_species_metrics()

        # Persistent handles: 61 opens at startup instead of 61 per request.
        self.handles = [rasterio.open(layer["path"]) for layer in self.catalog]
        self.reference = self._build_reference(reference_cells, cache_dir)

        if verbose:
            north, south, west, east = self.bounds()
            print(f"region           {self.name}  profile {self.profile}  "
                  f"{len(self.species)} species")
            print(f"                 coverage {self.coverage_kind}: "
                  f"{int(self.coverage.sum()):,} cells, "
                  f"lat {south:.2f}..{north:.2f} lon {west:.2f}..{east:.2f}")

    # ---- setup helpers -------------------------------------------------
    def _resolve_country(self):
        match = COUNTRY_DIR_PATTERN.search(
            feat.GRID_PROFILES[self.profile]["template"].replace("\\", "/"))
        if not match:
            return None
        return load_country_geometry(match.group(1).upper())

    def _build_coverage(self):
        if self.country:
            covered = coverage_from_geometry(self.country["geometry"], self.grid)
            if covered.any():
                return covered, f"{self.country['name']} boundary"
            warnings.warn(f"{self.name}: country polygon does not intersect the "
                          f"profile grid; falling back to raster extent")
        return coverage_from_template(self.grid), "raster extent (loose)"

    def _load_species_metrics(self):
        """Per-species cross-validated AUC, TSS, Boyce and occupancy.

        Without this the app cannot tell "the model ranks this species badly
        here" from "the model cannot rank this species anywhere", which is the
        whole widespread-species problem. If the table is missing, every
        species is marked unknown-reliability rather than silently trusted.
        """
        path = os.path.join(self.run_dir, SPECIES_METRICS_NAME)
        metrics = {}
        try:
            import pandas as pd
            for record in pd.read_csv(path).to_dict("records"):
                metrics[str(record["species"]).strip()] = {
                    "auc": _finite(record.get("auc")),
                    "tss": _finite(record.get("tss")),
                    "boyce": _finite(record.get("boyce")),
                    "n_presence": _finite(record.get("n_presence")),
                }
        except Exception as error:                       # noqa: BLE001
            warnings.warn(f"{path} unreadable ({error}); per-species "
                          f"reliability will be reported as unknown")
        return metrics

    def _build_reference(self, n_cells, cache_dir):
        """Score a random sample of covered land so percentiles have a base."""
        stamp = int(os.path.getmtime(os.path.join(self.run_dir,
                                                  CHECKPOINT_NAME)))
        path = os.path.join(cache_dir,
                            f"{self.name}_ref_{n_cells}_{stamp}.npz")
        if os.path.exists(path):
            try:
                reference = ReferenceScores.load(path, len(self.species))
                if self.verbose:
                    print(f"                 reference distribution from "
                          f"cache ({reference.n_cells:,} cells)")
                return reference
            except Exception as error:                   # noqa: BLE001
                warnings.warn(f"reference cache {path} unusable ({error}); "
                              f"recomputing")

        rows, cols = np.nonzero(self.coverage)
        rng = np.random.default_rng(REFERENCE_SEED)
        take = min(n_cells, len(rows))
        pick = rng.choice(len(rows), size=take, replace=False)
        lat, lon = feat.cell_centres(rows[pick], cols[pick], self.grid)

        if self.verbose:
            print(f"                 building reference distribution from "
                  f"{take:,} covered cells (once; then cached)")
        features, inside = feat.sample_points(self.catalog, lat, lon, self.grid,
                                              progress=self.verbose,
                                              desc="reference sample")
        features = self._add_indicators(features)
        keep = inside & feat.drop_empty_rows(features[:, :self.n_predictors])
        if keep.sum() < 100:
            raise ValueError(f"{self.name}: only {int(keep.sum())} covered "
                             f"cells have usable predictors; the rasters for "
                             f"this profile look wrong")
        scores = self._forward(features[keep])
        reference = ReferenceScores(np.sort(scores, axis=0).T.copy(),
                                    int(keep.sum()))
        try:
            reference.save(path)
        except OSError as error:
            warnings.warn(f"could not cache reference scores: {error}")
        return reference

    # ---- prediction ----------------------------------------------------
    def _add_indicators(self, features):
        indicators, _ = feat.missingness_columns(features, self.groups)
        return np.concatenate([features, indicators], axis=1)

    def _forward(self, features):
        with torch.no_grad():
            tensor = torch.as_tensor(self.scaler.transform(features),
                                     dtype=torch.float32, device=self.device)
            return torch.sigmoid(self.model(tensor)).cpu().numpy()

    def bounds(self):
        """(north, south, west, east) of the covered cells, in degrees."""
        rows, cols = np.nonzero(self.coverage)
        north, west = feat.cell_centres(rows.min(), cols.min(), self.grid)
        south, east = feat.cell_centres(rows.max(), cols.max(), self.grid)
        return float(north), float(south), float(west), float(east)

    def covers(self, lat, lon):
        row, col, inside = feat.lonlat_to_rowcol([lat], [lon], self.grid)
        if not inside[0]:
            return False
        return bool(self.coverage[row[0], col[0]])

    def sample_cells(self, rows, cols):
        """Predictors at many cells, one read per layer over a covering window.

        The nearby search asks for a few dozen cells inside a 25 km box, which
        is at most a 50x50 window — so reading each raster once across the
        whole box and fancy-indexing out of it costs barely more than reading
        a single pixel, and 61 opens are still avoided because the handles were
        opened at startup.

        `feat.read_band` and `feat.expand_circular` are reused rather than
        reimplemented so that nodata sentinels and any circular layer are
        treated exactly as they were during training.
        """
        rows = np.atleast_1d(np.asarray(rows, dtype="int64"))
        cols = np.atleast_1d(np.asarray(cols, dtype="int64"))
        r0, c0 = int(rows.min()), int(cols.min())
        window = Window(c0, r0, int(cols.max()) - c0 + 1,
                        int(rows.max()) - r0 + 1)
        local_rows, local_cols = rows - r0, cols - c0

        parts = []
        with self._lock:
            for handle, layer in zip(self.handles, self.catalog):
                band = feat.read_band(handle, window=window)
                parts.append(feat.expand_circular(band[local_rows, local_cols],
                                                  layer["kind"]))
        return np.concatenate(parts, axis=1)

    def sample_cell(self, row, col):
        """Predictors at one cell. The single-point case of `sample_cells`."""
        return self.sample_cells([row], [col])

    def score_cells(self, rows, cols):
        """Scores for many cells at once. Returns (scores, usable) aligned.

        `usable` is False wherever the training code would have dropped the
        row, so a nearby suggestion is never built on a cell whose predictors
        are mostly absent.
        """
        raw = self.sample_cells(rows, cols)
        features = self._add_indicators(raw)
        usable = feat.drop_empty_rows(features[:, :self.n_predictors])
        scores = np.full((len(features), len(self.species)), np.nan,
                         dtype="float32")
        if usable.any():
            scores[usable] = self._forward(features[usable])
        return scores, usable

    def score_point(self, lat, lon):
        """Raw per-species scores at one coordinate, plus what was read there.

        Returns None for `scores` when more than MAX_MISSING_FRACTION of the
        predictors are absent — the training code drops exactly those rows, so
        serving them would be scoring a cell the model never saw the like of.
        """
        row, col, inside = feat.lonlat_to_rowcol([lat], [lon], self.grid)
        row, col = int(row[0]), int(col[0])
        if not inside[0]:
            return {"scores": None, "reason": "off_grid"}

        raw = self.sample_cell(row, col)
        features = self._add_indicators(raw)
        missing = float(np.isnan(raw[0]).mean())
        if not feat.drop_empty_rows(features[:, :self.n_predictors])[0]:
            # The mask is still read here: a cell with no predictors is
            # usually sea, and "that is the sea" is a far more useful answer
            # than "the rasters are empty".
            return {"scores": None, "reason": "missing_predictors",
                    "missing_fraction": missing, "row": row, "col": col,
                    "mask_value": int(self.mask[row, col])}

        scores = self._forward(features)[0]
        centre_lat, centre_lon = feat.cell_centres(row, col, self.grid)
        return {"scores": scores, "reason": None,
                "missing_fraction": missing, "row": row, "col": col,
                "mask_value": int(self.mask[row, col]),
                "cell_lat": float(centre_lat), "cell_lon": float(centre_lon)}

    def close(self):
        for handle in self.handles:
            try:
                handle.close()
            except Exception:                            # noqa: BLE001
                pass


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:                                    # noqa: BLE001
        return None


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


# ─────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────
def discover_runs(output_root=OUTPUT_ROOT):
    """Group runnable checkpoints by profile, best candidate first.

    Discovery rather than a hardcoded path so that when the sibling agent's
    Spain or USA data lands and a checkpoint is trained against a new
    `GRID_PROFILES` entry, this picks it up with no edit here.
    """
    by_profile = {}
    if not os.path.isdir(output_root):
        return by_profile
    for name in sorted(os.listdir(output_root)):
        run_dir = os.path.join(output_root, name)
        if name in SKIP_RUNS or not os.path.isdir(run_dir):
            continue
        if not os.path.exists(os.path.join(run_dir, CHECKPOINT_NAME)):
            continue
        summary = _read_json(os.path.join(run_dir, SUMMARY_NAME)) or {}
        profile = summary.get("profile")
        if profile not in feat.GRID_PROFILES:
            continue
        if not os.path.exists(feat.GRID_PROFILES[profile]["template"]):
            continue
        by_profile.setdefault(profile, []).append({
            "dir": run_dir, "name": name, "summary": summary,
            "preferred": name in PREFERRED_RUNS,
            "n_species": int(summary.get("n_species_total", 0)),
            "mtime": os.path.getmtime(os.path.join(run_dir, CHECKPOINT_NAME)),
        })
    for candidates in by_profile.values():
        candidates.sort(key=lambda c: (not c["preferred"], -c["n_species"],
                                       -c["mtime"]))
    return by_profile


def choose_profiles(by_profile):
    """One run per country, finest grid first.

    fra_30s and fra_10m are the same country at two resolutions; serving both
    would ask a user to pick a number they have no basis to pick. The 1 km run
    wins on every headline metric (AUC 0.747 against 0.734), so finest-first is
    also best-first.
    """
    best = {}
    for profile, candidates in by_profile.items():
        template = feat.GRID_PROFILES[profile]["template"].replace("\\", "/")
        match = COUNTRY_DIR_PATTERN.search(template)
        country = match.group(1).upper() if match else profile
        cell = _profile_cell(profile)
        if country not in best or cell < best[country][0]:
            best[country] = (cell, candidates[0])
    return [best[country][1] for country in sorted(best)]


def _profile_cell(profile):
    try:
        with rasterio.open(feat.GRID_PROFILES[profile]["template"]) as src:
            return abs(src.transform.a)
    except Exception:                                    # noqa: BLE001
        return 999.0


# ─────────────────────────────────────────────────────
# THE ADVISORY
# ─────────────────────────────────────────────────────
class Advisory:
    """Every loaded region, and the logic that turns a click into advice."""

    def __init__(self, device=DEVICE, reference_cells=REFERENCE_CELLS,
                 output_root=OUTPUT_ROOT, verbose=True):
        assert not feat.ALLOW_VEGETATION_PREDICTORS_EXPERIMENT, \
            "the vegetation guard is open; refusing to serve predictions"
        self.regions = []
        self.failures = []
        self.requested_device = device
        # Resolved once and reported, so a container health check states the
        # device actually in use rather than the string someone typed.
        self.device = str(resolve_device(device))
        self.names = load_species_names()
        self.land_cover = GlobalLandCover()

        candidates = choose_profiles(discover_runs(output_root))
        if verbose:
            divider("LOADING TRAINED MODELS")
            if not candidates:
                print(f"no usable run found under {output_root}/")
        for candidate in candidates:
            try:
                self.regions.append(RegionModel(candidate["dir"], device=device,
                                                reference_cells=reference_cells,
                                                verbose=verbose))
            except Exception as error:                   # noqa: BLE001
                # One unloadable checkpoint must not take the service down —
                # a second country should still be answerable.
                self.failures.append({"run": candidate["name"],
                                      "error": f"{type(error).__name__}: "
                                               f"{error}"})
                print(f"region           {candidate['name']} FAILED TO LOAD: "
                      f"{type(error).__name__}: {error}")
        if verbose:
            print(f"\nloaded           {len(self.regions)} region(s), "
                  f"{len(self.failures)} failure(s)")

    @property
    def ready(self):
        return bool(self.regions)

    def coverage_summary(self):
        out = []
        for region in self.regions:
            north, south, west, east = region.bounds()
            out.append({
                "region": region.country["name"] if region.country
                          else region.profile,
                "profile": region.profile,
                "run": region.name,
                "resolution": feat.GRID_PROFILES[region.profile]["label"],
                "n_species": len(region.species),
                "coverage_kind": region.coverage_kind,
                "bounds": {"north": round(north, 3), "south": round(south, 3),
                           "west": round(west, 3), "east": round(east, 3)},
                "mean_auc": _finite(
                    (region.summary.get("mean") or {}).get("auc")),
            })
        return out

    # ---- the request ---------------------------------------------------
    def recommend(self, lat, lon, top=TOP_RECOMMENDATIONS,
                  override_exclusion=False, nearby=True):
        """Coordinates -> a structured answer, or a plain reason there is none."""
        if not self.ready:
            return {"status": "unavailable",
                    "headline": "No trained model is loaded",
                    "detail": "The server started without a usable checkpoint. "
                              "See /api/health for what failed.",
                    "failures": self.failures}

        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            return {"status": "bad_request",
                    "headline": "Those coordinates are not numbers",
                    "detail": "Give latitude and longitude in decimal degrees, "
                              "for example 43.61 and 3.88."}
        if not (np.isfinite(lat) and np.isfinite(lon)
                and -90 <= lat <= 90 and -180 <= lon <= 180):
            return {"status": "bad_request",
                    "headline": "Those coordinates are out of range",
                    "detail": "Latitude must be between -90 and 90, longitude "
                              "between -180 and 180."}

        region = next((r for r in self.regions if r.covers(lat, lon)), None)
        if region is None:
            covered = ", ".join(
                r.country["name"] if r.country else r.profile
                for r in self.regions) or "nowhere"
            reading = self.land_cover.describe(lat, lon)
            at_sea = reading is not None and "sea" in reading
            # A point just off the coast is outside every coverage polygon but
            # may still be a few kilometres from land the model does cover, so
            # the search is worth running. It can only ever return cells inside
            # a coverage polygon, which is what keeps Spain out of the answer
            # for a point in the Gulf of Lion.
            block = self._nearby_block(
                lat, lon,
                "This spot is outside every region the model covers. These "
                "are the nearest places inside coverage.", top=top) \
                if nearby else None
            return {
                "status": "outside_coverage",
                "latitude": lat, "longitude": lon,
                "headline": "That point is open sea" if at_sea
                            else "No model covers this location",
                "land_cover_reading": reading,
                "detail": f"This advisory has only been trained and validated "
                          f"for {covered}. A model fitted on French "
                          f"occurrences has no basis for ranking trees "
                          f"anywhere else, so no list is shown rather than a "
                          f"list that looks right and is not.",
                "nearby": block,
                "coverage": self.coverage_summary(),
                "caveats": STANDING_CAVEATS}

        result = region.score_point(lat, lon)
        if result["scores"] is None:
            return self._no_data(region, lat, lon, result, top=top,
                                 nearby=nearby)

        hard, advisory = self._mask_findings(result["mask_value"])
        base = {
            "status": "ok",
            "latitude": lat, "longitude": lon,
            "cell": {"latitude": result["cell_lat"],
                     "longitude": result["cell_lon"],
                     "row": result["row"], "col": result["col"]},
            "region": self._region_block(region),
            "advisory_flags": advisory,
            "exclusions": hard,
            "missing_fraction": round(result["missing_fraction"], 3),
            "caveats": STANDING_CAVEATS,
        }

        if hard and not override_exclusion:
            headline = " and ".join(item["title"].lower() for item in hard)
            base.update({
                "status": "excluded",
                "headline": f"Not plantable land — mapped as {headline}",
                "detail": "No species list is shown for this point. The "
                          "land-use mask is only published on an 18.5 km "
                          "grid, so this verdict is about the wider cell, not "
                          "the exact metre you asked about. If you believe the "
                          "spot itself is plantable, ask again with "
                          "'show anyway'.",
                "can_override": True,
                "recommendations": None,
                "nearby": self._nearby_block(
                    lat, lon,
                    f"This spot is mapped as {headline}, so here are the "
                    f"nearest places that are not.", top=top)
                    if nearby else None})
            return base

        base.update(self.rank(region, result["scores"], top=top))
        if hard and override_exclusion:
            base["status"] = "ok_overridden"
            base["headline"] = (
                "Shown despite a land-use exclusion — "
                + ", ".join(item["title"].lower() for item in hard))
            base["detail"] = ("The 18.5 km land-use cell around this point is "
                              "flagged as not plantable. The list below "
                              "ignores that flag because you asked it to; "
                              "check the spot on the ground.")
        elif not base["worth_planting"]:
            # Not "these trees will die" — every score here is relative. It
            # means nothing on the list rises above unremarkable for this
            # place, which happens on about 1% of covered cells.
            base["status"] = "ok_poor"
            base["headline"] = "Nothing here stands out as a good planting spot"
            base["detail"] = (
                "Every species the model can rank comes out unremarkable or "
                "worse for this location — no tree rates above the middle of "
                "its own range here. The list below is still the best of "
                "them, and nearby alternatives follow.")
            if nearby:
                base["nearby"] = self._nearby_block(
                    lat, lon,
                    "Nothing rated well at the spot you asked about, so these "
                    "are the nearest places that do.", top=top)
        else:
            base["headline"] = "Tree planting options for this location"
            base["detail"] = (
                f"Compared against the rest of {base['region']['name']}. A "
                f"high rating means this spot suits that tree better than "
                f"most of the region does — it is not a survival chance.")
        return base

    def _no_data(self, region, lat, lon, result, top=TOP_RECOMMENDATIONS,
                 nearby=True):
        hard, advisory = self._mask_findings(result.get("mask_value", 0))
        if result["reason"] == "missing_predictors":
            wet = [item for item in hard
                   if item["flag"] in ("ocean", "inland_water")]
            if wet:
                headline = f"That point is {wet[0]['title'].lower()}"
                detail = ("It falls just inside the drawn national boundary, "
                          "but there is no climate, soil or terrain data for "
                          "its 1 km cell, which is what open water looks like "
                          "in these rasters. Try a point further inland.")
            else:
                headline = "No environmental data at this exact cell"
                detail = ("This point is inside the covered region but the "
                          "climate, soil and terrain rasters have no usable "
                          "values for its 1 km cell — in practice a narrow "
                          "coastline, a small island or a sliver of sea inside "
                          "the boundary. Try a point a kilometre or two "
                          "inland.")
        else:
            headline = "Outside the model's raster grid"
            detail = "This point falls off the edge of the predictor rasters."
        return {
            "status": "no_data",
            "latitude": lat, "longitude": lon,
            "region": self._region_block(region),
            "headline": headline,
            "detail": detail,
            "exclusions": hard,
            "advisory_flags": advisory,
            "missing_fraction": round(result.get("missing_fraction", 1.0), 3),
            "caveats": STANDING_CAVEATS,
            "recommendations": None,
            "nearby": self._nearby_block(
                lat, lon,
                "There is no environmental data for this exact cell, so here "
                "are the nearest places there is.", top=top)
                if nearby else None}

    def _region_block(self, region):
        mean = region.summary.get("mean") or {}
        return {
            "name": region.country["name"] if region.country else region.profile,
            "profile": region.profile,
            "run": region.name,
            "label": feat.GRID_PROFILES[region.profile]["label"],
            "n_species": len(region.species),
            "mean_auc": _finite(mean.get("auc")),
            "mean_tss": _finite(mean.get("tss")),
            "mean_boyce": _finite(mean.get("boyce")),
            "reference_cells": region.reference.n_cells,
        }

    def _mask_findings(self, value):
        """Split the bit-coded mask into refusals and notes."""
        hard, advisory = [], []
        for name, bit in feat.EXCLUSION_BITS.items():
            if not value & bit:
                continue
            if bit & feat.HARD_EXCLUSION_BITS:
                title, detail = EXCLUSION_NOTES[name]
                hard.append({"flag": name, "title": title, "detail": detail})
            elif name in ADVISORY_NOTES:
                title, detail = ADVISORY_NOTES[name]
                advisory.append({"flag": name, "title": title,
                                 "detail": detail})
        return hard, advisory

    # ---- ranking -------------------------------------------------------
    def describe_species(self, region, scores):
        """Every species as a plain-language row, grouped. Nothing truncated.

        The grouping is the point. Sorting all 96 by score puts cork oak, holm
        oak and Aleppo pine at the top of a Mediterranean answer and leaves
        hawthorn, hazel, ash and sycamore near the bottom — not because they
        are poor choices but because they grow everywhere in the training
        region, so the model was never shown a contrast and learned AUC 0.59
        to 0.61 for them. Those four are among the most commonly planted trees
        in France. Burying them would be the single most misleading thing this
        page could do, so they are lifted into their own group and the reason
        is written next to them in words.
        """
        percentiles = region.reference.percentile(scores)
        rows = []
        for index, species in enumerate(region.species):
            metric = region.metrics.get(species, {})
            auc = metric.get("auc")
            occupancy = metric.get("n_presence")
            naming = self.names.get(species, {})
            caution, caution_note = _planting_caution(naming.get("notes", ""))
            percentile = float(percentiles[index])
            band, rating_phrase, rating_detail = _band(percentile)
            confidence = _confidence_key(auc, occupancy)
            confidence_short, confidence_phrase, confidence_detail = \
                CONFIDENCE[confidence]
            common = naming.get("common_name", "") or species
            rows.append({
                "species": species,
                "common_name": naming.get("common_name", ""),
                "display_name": common,
                "caution": caution,
                "caution_note": caution_note,
                "use": naming.get("use", ""),
                "use_label": USE_LABELS.get(naming.get("use", ""),
                                            naming.get("use", "").replace(
                                                "_", " ").capitalize()),
                "notes": naming.get("notes", ""),
                # ---- plain language, the primary presentation ----
                "rating_phrase": rating_phrase,
                "rating_detail": rating_detail,
                "confidence_short": confidence_short,
                "confidence_phrase": confidence_phrase,
                "confidence_detail": confidence_detail,
                "summary": f"{common} — {rating_phrase}.",
                "cutoff_note": _cutoff_note(scores[index],
                                            region.thresholds[index]),
                # ---- the figures, for the block at the end ----
                "site_percentile": round(percentile, 1),
                "site_index": round(float(scores[index]), 4),
                "band": band,
                "confidence": confidence,
                "model_auc": round(auc, 3) if auc is not None else None,
                "occupancy": round(occupancy, 1) if occupancy is not None
                             else None,
                "above_model_threshold": (
                    bool(scores[index] >= region.thresholds[index])
                    if np.isfinite(region.thresholds[index]) else None),
            })

        groups = bucket_species(rows)
        for members in groups.values():
            # Species that clear their own trained cut-off come first, then
            # percentile within that. Ordering by percentile alone floats
            # species whose whole distribution sits near zero — see
            # REQUIRE_MODEL_CUTOFF. Even in the region-wide group the score is
            # worth ordering by: a hawthorn site at 2,600 m really is at the
            # bottom of hawthorn's range.
            members.sort(key=lambda r: (r["above_model_threshold"] is False,
                                        -r["site_percentile"]))
        for name, members in groups.items():
            for row in members:
                row["group"] = name
        return groups

    def rank(self, region, scores, top=TOP_RECOMMENDATIONS,
             well_matched_share=None):
        """Cut the grouped species down to one screenful.

        `top` is a cap on the *whole* recommendation rather than on each group,
        and the two groups backfill each other: if the model has only one
        well-matched species here, the remaining four slots go to dependable
        region-wide ones rather than being left empty. Invasives never take a
        slot — they are a warning, and one that is only raised when the species
        scores well enough here to be tempting.

        The 3-of-5 default is held as a proportion rather than a fixed count,
        so raising `top` keeps well-matched species in the majority instead of
        pouring every extra slot into the general group. Every count is
        clamped at zero: a negative slot count silently becomes a negative
        list slice, which returns most of the list rather than none of it, and
        `top=1` duly returned 34 species before this was pinned down.
        """
        top = max(1, int(top))
        groups = self.describe_species(region, scores)
        well, general = groups["well_matched"], groups["general"]

        share = int(np.ceil(top * WELL_MATCHED_SHARE / TOP_RECOMMENDATIONS)) \
            if well_matched_share is None else int(well_matched_share)
        n_well = max(0, min(share, len(well), top))
        n_general = max(0, min(top - n_well, len(general)))
        n_well = max(0, min(len(well), top - n_general))   # backfill both ways
        picked = well[:n_well] + general[:n_general]
        picked.sort(key=lambda r: (r["group"] != "well_matched",
                                   r["above_model_threshold"] is False,
                                   -r["site_percentile"]))

        warnings_invasive = [row for row in groups["invasive"]
                             if row["site_percentile"]
                             >= INVASIVE_WARNING_PERCENTILE][:top]
        # "Worth planting" is judged over every species, not just the five
        # shown, so a short list never makes a location look worse than it is.
        candidates = groups["well_matched"] + groups["general"]
        qualifying = [row["site_percentile"] for row in candidates
                      if row["site_percentile"] >= WORTH_PLANTING_PERCENTILE
                      and (row["above_model_threshold"] is not False
                           or not REQUIRE_MODEL_CUTOFF)]
        best = max([row["site_percentile"] for row in candidates] or [0.0])
        return {
            "recommendations": picked,
            "invasive_warnings": warnings_invasive,
            "best_percentile": round(float(best), 1),
            "worth_planting": bool(qualifying),
            "totals": {name: len(members)
                       for name, members in groups.items()},
            "shown": {"well_matched": n_well, "general": n_general,
                      "cap": int(top)},
            "group_labels": GROUP_LABELS,
            "group_blurbs": GROUP_BLURBS,
            "metrics_key": METRICS_KEY,
            "threshold_note":
                f"The 'passes the model's own cut-off' column uses the "
                f"{region.threshold_strategy} threshold fitted during "
                f"training. Treat it as a hint: re-optimising the threshold on "
                f"held-out folds moved mean TSS from 0.326 to 0.398, so the "
                f"cut-off does not transfer cleanly across the region.",
        }

    # ---- nearby alternatives -------------------------------------------
    def find_nearby(self, lat, lon, top=NEARBY_SPECIES_SHOWN,
                    wanted=NEARBY_SUGGESTIONS, max_km=NEARBY_MAX_KM):
        """Nearest covered, unexcluded, decent places within `max_km`.

        Searches every loaded region, so a point just off the French coast can
        be answered with French land — but only ever returns cells inside some
        region's coverage polygon, which is what keeps Barcelona out of an
        answer for the Gulf of Lion no matter how close it is.

        The candidate set is filtered entirely with arrays already in memory
        (the coverage polygon and the land-use mask), thinned to a 3 km
        lattice, and only then scored, in one batch. So the whole search costs
        about as much as a single extra point query.
        """
        found = []
        for region in self.regions:
            found.extend(self._nearby_in_region(region, lat, lon, top, max_km))
        found.sort(key=lambda item: item["distance_km"])

        # Spread the answers out: three adjacent cells in one field are one
        # suggestion, not three.
        chosen = []
        for item in found:
            if all(distance_km(item["latitude"], item["longitude"],
                               other["latitude"], other["longitude"])
                   >= NEARBY_MIN_SEPARATION_KM for other in chosen):
                chosen.append(item)
            if len(chosen) >= wanted:
                break
        return chosen

    def _nearby_in_region(self, region, lat, lon, top, max_km):
        grid = region.grid
        cell_deg = grid["cell"]
        # Half-window in cells, generous on longitude because a degree of
        # longitude is shorter than a degree of latitude at these latitudes.
        span_lat = max_km / 110.574
        span_lon = max_km / max(111.320 * np.cos(np.radians(lat)), 1e-6)
        row0, col0, _ = feat.lonlat_to_rowcol([lat], [lon], grid)
        half_rows = int(np.ceil(span_lat / cell_deg))
        half_cols = int(np.ceil(span_lon / cell_deg))
        r0 = max(0, int(row0[0]) - half_rows)
        r1 = min(grid["height"], int(row0[0]) + half_rows + 1)
        c0 = max(0, int(col0[0]) - half_cols)
        c1 = min(grid["width"], int(col0[0]) + half_cols + 1)
        if r1 <= r0 or c1 <= c0:
            return []

        # Thin to a lattice: neighbouring 1 km cells share almost all their
        # climate and soil, so scoring every one of ~2,000 to pick three is
        # wasted work.
        stride = max(1, int(round(NEARBY_STRIDE_KM / (cell_deg * 110.574))))
        rows = np.arange(r0, r1, stride)
        cols = np.arange(c0, c1, stride)
        if not len(rows) or not len(cols):
            return []
        mesh_rows, mesh_cols = np.meshgrid(rows, cols, indexing="ij")
        mesh_rows, mesh_cols = mesh_rows.ravel(), mesh_cols.ravel()

        covered = region.coverage[mesh_rows, mesh_cols]
        clear = (region.mask[mesh_rows, mesh_cols].astype("int32")
                 & int(feat.HARD_EXCLUSION_BITS)) == 0
        keep = covered & clear
        if not keep.any():
            return []
        mesh_rows, mesh_cols = mesh_rows[keep], mesh_cols[keep]

        cand_lat, cand_lon = feat.cell_centres(mesh_rows, mesh_cols, grid)
        km = distance_km(lat, lon, cand_lat, cand_lon)
        near = km <= max_km
        if not near.any():
            return []
        order = np.argsort(km[near])[:NEARBY_MAX_CANDIDATES]
        mesh_rows = mesh_rows[near][order]
        mesh_cols = mesh_cols[near][order]

        scores, usable = region.score_cells(mesh_rows, mesh_cols)
        out = []
        for i in range(len(mesh_rows)):
            if not usable[i]:
                continue
            ranked = self.rank(region, scores[i], top=top,
                               well_matched_share=max(1, top - 1))
            if not ranked["worth_planting"]:
                continue
            here_lat, here_lon = feat.cell_centres(mesh_rows[i], mesh_cols[i],
                                                   grid)
            phrase, km_away, direction = plain_direction(
                lat, lon, float(here_lat), float(here_lon))
            _, advisory = self._mask_findings(
                int(region.mask[mesh_rows[i], mesh_cols[i]]))
            out.append({
                "latitude": round(float(here_lat), 4),
                "longitude": round(float(here_lon), 4),
                "distance_km": round(float(km_away), 1),
                "direction": direction,
                "how_far": phrase,
                "region": region.country["name"] if region.country
                          else region.profile,
                "recommendations": ranked["recommendations"],
                "advisory_flags": advisory,
                "best_percentile": ranked["best_percentile"],
            })
        return out

    def _nearby_block(self, lat, lon, why, top=NEARBY_SPECIES_SHOWN):
        """The whole nearby section, or None when the search finds nothing.

        An alternative location gets a shorter list than the location actually
        asked about — it is a pointer to somewhere else, not a second answer.
        """
        suggestions = self.find_nearby(
            lat, lon, top=max(1, min(int(top), NEARBY_SPECIES_SHOWN)))
        if not suggestions:
            return None
        return {
            "why": why,
            "searched_km": NEARBY_MAX_KM,
            "caveat": NEARBY_CAVEAT,
            "suggestions": suggestions,
        }


def bucket_species(rows, reliable_auc=RELIABLE_AUC,
                   widespread_presences=WIDESPREAD_PRESENCES):
    """Split scored species into the four groups the answer distinguishes.

    Invasives leave first and unconditionally, whatever they scored. Then AUC
    decides: species the model can discriminate are well matched to a specific
    place, species it cannot but which occupy most of the region are the
    dependable general choices, and the remainder is honestly unrankable.
    Unknown AUC (no metrics table) is treated as rankable rather than
    discarded, but its confidence line reads "no reliability score" so the
    interface can say so.
    """
    groups = {"well_matched": [], "general": [], "unrankable": [],
              "invasive": []}
    for row in rows:
        if row["caution"] == "do_not_plant":
            groups["invasive"].append(row)
            continue
        auc = row["model_auc"]
        occupancy = row["occupancy"]
        if auc is None or auc >= reliable_auc:
            groups["well_matched"].append(row)
        elif occupancy is not None and occupancy >= widespread_presences:
            groups["general"].append(row)
        else:
            groups["unrankable"].append(row)
    return groups


def _planting_caution(notes):
    """Read the curated free-text note for a reason not to plant something.

    Crude on purpose: it is a substring match over a hand-written column, not
    an invasiveness database. It errs towards flagging, because the cost of
    wrongly warning about a tree is far below the cost of recommending
    Ailanthus altissima to someone who then plants it.
    """
    text = (notes or "").lower()
    if any(token in text for token in DO_NOT_PLANT_TOKENS):
        return "do_not_plant", (f"The species list flags this as invasive "
                                f"({notes}). It is scored here because the "
                                f"model scores every species, not as a "
                                f"suggestion to plant it.")
    if any(token in text for token in CAUTION_TOKENS):
        return "caution", (f"Note from the species list: {notes}. The model "
                           f"does not know about pests or disease — it only "
                           f"sees climate, soil and terrain.")
    return None, ""


def _band(percentile):
    """(key, short phrase, sentence) for how a location rates."""
    for floor, key, phrase, detail in BANDS:
        if percentile >= floor:
            return key, phrase, detail
    return BANDS[-1][1], BANDS[-1][2], BANDS[-1][3]


def _cutoff_note(score, threshold):
    """Said only when the model would not itself pick this tree for this spot.

    Worded as a caveat rather than a rejection on purpose: the cut-off is the
    max-TSS threshold from training and it does not transfer cleanly, so it is
    worth knowing and not worth obeying blindly.
    """
    if not np.isfinite(threshold) or score >= threshold:
        return ""
    # Kept to one line. At a genuinely poor location every species trips this,
    # and five copies of a three-line explanation is the wall of text the
    # plain-language rewrite exists to remove. The reasoning sits once in
    # `threshold_note`, at the end with the other figures.
    return "A fallback, not a real fit — below the model's own bar for this tree."


def _confidence_key(auc, occupancy):
    """How much the model knows about a species — never about the location.

    The widespread case is split out from plain "low" because the two look
    identical in the number and mean opposite things to a reader: both score
    near 0.6, but one is a tree that grows everywhere in the region and the
    other is a tree the model simply cannot place.
    """
    if auc is None:
        return "unknown"
    if auc >= GOOD_AUC:
        return "high"
    if auc >= RELIABLE_AUC:
        return "moderate"
    if occupancy is not None and occupancy >= WIDESPREAD_PRESENCES:
        return "widespread"
    return "low"


# ─────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────
# The regression points. They exist because they exercise five different code
# paths — a Mediterranean answer, an alpine one, a lowland one, open sea, and
# a coordinate no model covers — not because anyone wants to read them. Behind
# --self-test, so asking about one location does not print five others.
SELF_TEST_POINTS = (
    ("Mediterranean south (Herault)", 43.55, 3.30),
    ("Alpine (Ecrins)", 44.95, 6.35),
    ("Northern lowland (Somme)", 49.85, 2.35),
    ("Bay of Biscay, open sea", 46.00, -3.50),
    ("Tokyo, outside coverage", 35.68, 139.69),
)
WRAP = 74


def _wrap(text, indent=2, hang=None):
    """Paragraph wrapping, because these are sentences now rather than codes."""
    import textwrap
    return textwrap.fill(text, width=WRAP, initial_indent=" " * indent,
                         subsequent_indent=" " * (indent if hang is None
                                                  else hang))


def print_answer(answer, show_metrics=True):
    """The plain-language report. Figures last, behind their own heading."""
    print()
    print(answer.get("headline", answer.get("status", "")))
    print("-" * WRAP)
    if answer.get("detail"):
        print(_wrap(answer["detail"], indent=0))
    if answer.get("land_cover_reading"):
        print(_wrap(answer["land_cover_reading"], indent=0))

    for flag in (answer.get("exclusions") or []):
        print(f"\n  ! {flag['title']}")
        print(_wrap(flag["detail"], indent=4))
    for flag in (answer.get("advisory_flags") or []):
        print(f"\n  note: {flag['title']}")
        print(_wrap(flag["detail"], indent=4))

    rows = answer.get("recommendations") or []
    if rows:
        labels = answer.get("group_labels", GROUP_LABELS)
        print(f"\nWHAT TO PLANT HERE  (best {len(rows)})")
        last_group = None
        for i, row in enumerate(rows, 1):
            if row["group"] != last_group:
                print(f"\n  {labels.get(row['group'], row['group'])}")
                last_group = row["group"]
            latin = f" ({row['species']})" if row["common_name"] else ""
            use = f" — {row['use_label'].lower()}" if row.get("use_label") \
                else ""
            print()
            print(_wrap(f"{i}. {row['display_name']}{latin}{use}",
                        indent=2, hang=5))
            print(_wrap(row["rating_phrase"].capitalize() + ".",
                        indent=5))
            print(_wrap(f"Model confidence: {row['confidence_short']}.",
                        indent=5))
            if row.get("cutoff_note"):
                print(_wrap(row["cutoff_note"], indent=5))
            if row.get("caution") == "caution":
                print(_wrap(f"Watch out: {row['notes']}.", indent=5))

    invasive = answer.get("invasive_warnings") or []
    if invasive:
        names = ", ".join(row["display_name"] for row in invasive)
        print("\n  ! Do not plant: " + names)
        print(_wrap("The species list flags these as invasive. The model "
                    "rates them well here because it knows nothing about "
                    "invasiveness — that is why they are kept out of the "
                    "list above.", indent=4))

    nearby = answer.get("nearby")
    if nearby and nearby.get("suggestions"):
        print("\nNEARBY PLACES WORTH TRYING INSTEAD")
        print(_wrap(nearby["why"], indent=2))
        for item in nearby["suggestions"]:
            print(f"\n  {item['how_far']}  ({item['latitude']}, "
                  f"{item['longitude']})")
            for row in item["recommendations"]:
                print(_wrap(f"- {row['display_name']}: "
                            f"{row['rating_phrase']}", indent=5, hang=7))
            for flag in item.get("advisory_flags") or []:
                print(f"     note: {flag['title'].lower()}")
        print()
        print(_wrap(nearby["caveat"], indent=2))

    if answer.get("caveats"):
        print("\nHOW TO READ THIS")
        for line in answer["caveats"]:
            print(_wrap(f"- {line}", indent=2, hang=4))

    if show_metrics and rows:
        print_metrics(answer)


def print_metrics(answer):
    """The numbers, last, labelled, with the key next to them."""
    print("\n" + "=" * WRAP)
    print("THE NUMBERS BEHIND THIS  (not needed to use the advice)")
    print("=" * WRAP)
    header = (f"{'species':<26s} {'rating':>7s} {'confid':>7s} "
              f"{'raw':>7s}  {'cut-off':<8s}")
    print(header)
    print("-" * len(header))
    for row in (answer.get("recommendations") or []) + \
               (answer.get("invasive_warnings") or []):
        auc = "n/a" if row["model_auc"] is None else f"{row['model_auc']:.3f}"
        cut = {True: "passes", False: "below", None: "n/a"}[
            row["above_model_threshold"]]
        print(f"{row['display_name'][:26]:<26s} "
              f"{row['site_percentile']:>7.1f} {auc:>7s} "
              f"{row['site_index']:>7.3f}  {cut:<8s}")
    print()
    print(_wrap(answer.get("metrics_key", METRICS_KEY), indent=0))
    region = answer.get("region") or {}
    if region:
        print()
        print(_wrap(
            f"Model: {region.get('run')} on {region.get('label')}, "
            f"{region.get('n_species')} species. Cross-validated mean AUC "
            f"{region.get('mean_auc'):.3f}, TSS {region.get('mean_tss'):.3f}, "
            f"Boyce {region.get('mean_boyce'):.3f}. Ratings are against "
            f"{region.get('reference_cells', 0):,} sampled cells of covered "
            f"land.", indent=0))
    if answer.get("threshold_note"):
        print()
        print(_wrap(answer["threshold_note"], indent=0))
    totals = answer.get("totals")
    if totals:
        print()
        print(_wrap(
            f"Considered {sum(totals.values())} species: "
            f"{totals.get('well_matched', 0)} the model can match to a "
            f"specific place, {totals.get('general', 0)} that grow region-wide, "
            f"{totals.get('invasive', 0)} flagged invasive, "
            f"{totals.get('unrankable', 0)} it cannot rank.", indent=0))


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def parse_coordinates(values):
    """Accept '43.5 5.4', '43.5,5.4' and '43.5, 5.4' — all get typed."""
    text = " ".join(str(v) for v in values).replace(",", " ")
    parts = [p for p in text.split() if p]
    if len(parts) != 2:
        raise ValueError(f"expected two numbers, got {len(parts)}: {text!r}")
    return float(parts[0]), float(parts[1])


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Tree planting advice for a location.",
        epilog="examples:  advisory_service.py 43.55 3.30\n"
               "           advisory_service.py --point 44.95,6.35 --top 3\n"
               "           advisory_service.py --self-test",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("coordinates", nargs="*", default=None,
                        help="LAT LON, or LAT,LON")
    parser.add_argument("--point", action="append", default=None,
                        metavar="LAT,LON",
                        help="a location to score; repeatable")
    parser.add_argument("--self-test", action="store_true",
                        help="run the five built-in regression points instead")
    parser.add_argument("--top", type=int, default=TOP_RECOMMENDATIONS,
                        metavar="N",
                        help=f"how many species to recommend in total "
                             f"(default {TOP_RECOMMENDATIONS})")
    parser.add_argument("--no-metrics", action="store_true",
                        help="leave out the figures block at the end")
    parser.add_argument("--no-nearby", action="store_true",
                        help="do not search for nearby alternatives")
    parser.add_argument("--device", default=DEVICE,
                        help="auto (default), cpu, or cuda")
    parser.add_argument("--reference-cells", type=int, default=REFERENCE_CELLS)
    parser.add_argument("--verbose", action="store_true",
                        help="print the model-loading banner")
    return parser.parse_args(argv)


def main(argv=None):
    """One location in, plain advice out. The regression points need a flag."""
    args = parse_args(argv)

    points = []
    for spec in args.point or []:
        points.append((None, *parse_coordinates([spec])))
    if args.coordinates:
        points.append((None, *parse_coordinates(args.coordinates)))
    if args.self_test:
        points = [(label, lat, lon) for label, lat, lon in SELF_TEST_POINTS]
    if not points:
        raise SystemExit(
            "Give a location: advisory_service.py LAT LON\n"
            "   for example: advisory_service.py 43.55 3.30\n"
            "   or run the built-in regression points: --self-test")

    advisory = Advisory(device=args.device,
                        reference_cells=args.reference_cells,
                        verbose=args.verbose or args.self_test)

    for label, lat, lon in points:
        if label:
            divider(f"{label}  ({lat}, {lon})")
        else:
            print(f"\nLocation: {lat}, {lon}")
        answer = advisory.recommend(lat, lon, top=args.top,
                                    nearby=not args.no_nearby)
        print_answer(answer, show_metrics=not args.no_metrics)


if __name__ == "__main__":
    main()
