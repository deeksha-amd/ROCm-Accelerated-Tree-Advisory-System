"""Species cards for the recommendation POC.

New GBIF models: drop a .json in models/ (via xgboost_training.py) and this
module will pick it up. Add an optional row to data/species_traits.csv to
override the genus defaults (common name, goals, native region, warnings).
"""

import os

import pandas as pd

TRAITS_CSV = os.path.join("data", "species_traits.csv")
METRICS_PATH = os.path.join("models", "metrics.csv")
MIN_AUC_DEFAULT = 0.70

GENUS_ENGLISH = {
    "Quercus": "oak",
    "Pinus": "pine",
    "Acer": "maple",
    "Fagus": "beech",
    "Betula": "birch",
}

# Coarse horticulture defaults. Species rows in the CSV override these.
GENUS_DEFAULTS = {
    "Quercus": dict(
        goals="shade,beauty",
        sun="full",
        evergreen="no",
        care="Give it room and a deep soak while young. Oaks often resent compacted or constantly wet soil.",
        warning="",
        native_regions="n_america,europe,e_asia,mexico_ca,mediterranean",
        invasive_in="",
    ),
    "Pinus": dict(
        goals="beauty,shade",
        sun="full",
        evergreen="yes",
        care="Needs sun and soil that drains. Do not keep the roots soggy.",
        warning="",
        native_regions="n_america,europe,e_asia,mexico_ca,himalaya,mediterranean",
        invasive_in="",
    ),
    "Acer": dict(
        goals="beauty,shade",
        sun="part",
        evergreen="no",
        care="Prefers even moisture and mulch. Afternoon shade helps in hot summers.",
        warning="",
        native_regions="n_america,europe,e_asia",
        invasive_in="",
    ),
    "Fagus": dict(
        goals="shade,beauty",
        sun="part",
        evergreen="no",
        care="Likes rich, well-drained soil. Young trees need water in drought; not a dry-site tree.",
        warning="",
        native_regions="europe,n_america,e_asia",
        invasive_in="",
    ),
    "Betula": dict(
        goals="beauty",
        sun="full",
        evergreen="no",
        care="Likes cool, moist ground. In warm regions, bronze birch borer is a real risk.",
        warning="Short-lived compared with oak or beech.",
        native_regions="europe,n_america,e_asia,siberia",
        invasive_in="",
    ),
}

# Hand-tuned cards for well-known species. Unknown GBIF names fall back to genus.
SPECIES_OVERRIDES = {
    "Quercus robur": dict(
        common_name="English oak",
        goals="shade,beauty",
        native_regions="europe",
        care="A big, long-lived shade tree for deep soils. Slow at first, then massive.",
    ),
    "Quercus petraea": dict(
        common_name="Sessile oak",
        goals="shade,beauty",
        native_regions="europe",
        care="Similar to English oak; copes a little better on drier, rockier ground.",
    ),
    "Quercus cerris": dict(
        common_name="Turkey oak",
        goals="shade,beauty",
        native_regions="europe,mediterranean",
        warning="Can be weedy outside southern Europe.",
    ),
    "Quercus ilex": dict(common_name="Holm oak", evergreen="yes", native_regions="mediterranean"),
    "Quercus coccifera": dict(
        common_name="Kermes oak",
        goals="beauty",
        sun="full",
        evergreen="yes",
        native_regions="mediterranean",
        care="A tough, shrubby evergreen oak for hot, dry, limestone sites.",
    ),
    "Quercus suber": dict(common_name="Cork oak", evergreen="yes", native_regions="mediterranean"),
    "Quercus palustris": dict(common_name="Pin oak", native_regions="n_america"),
    "Quercus phellos": dict(
        common_name="Willow oak",
        goals="shade,beauty",
        native_regions="n_america",
        care="Fine-leaved shade oak for moist lowlands in the eastern US.",
    ),
    "Quercus shumardii": dict(common_name="Shumard oak", native_regions="n_america"),
    "Quercus kelloggii": dict(common_name="California black oak", native_regions="n_america"),
    "Quercus vaccinifolia": dict(common_name="Huckleberry oak", goals="beauty", native_regions="n_america"),
    "Quercus michauxii": dict(common_name="Swamp chestnut oak", native_regions="n_america"),
    "Quercus pyrenaica": dict(common_name="Pyrenean oak", native_regions="europe,mediterranean"),
    "Quercus pubescens": dict(common_name="Downy oak", native_regions="europe,mediterranean"),
    "Quercus canariensis": dict(common_name="Algerian oak", native_regions="mediterranean"),
    "Quercus infectoria": dict(common_name="Aleppo oak", native_regions="mediterranean"),
    "Quercus glauca": dict(common_name="Ring-cupped oak", native_regions="e_asia"),
    "Pinus sylvestris": dict(
        common_name="Scots pine",
        goals="beauty,shade",
        native_regions="europe,siberia",
        care="Hardy pine for poor, sandy, or rocky soils. Needs full sun.",
    ),
    "Pinus nigra": dict(common_name="European black pine", native_regions="europe,mediterranean"),
    "Pinus pinaster": dict(
        common_name="Maritime pine",
        native_regions="mediterranean",
        invasive_in="australia,southern_africa,s_america",
        warning="Widely planted; invasive in parts of Australia, South Africa, and South America.",
    ),
    "Pinus pinea": dict(
        common_name="Italian stone pine",
        goals="food,beauty,shade",
        native_regions="mediterranean",
        care="The pine-nut pine. Needs heat, sun, and well-drained soil.",
    ),
    "Pinus halepensis": dict(
        common_name="Aleppo pine",
        native_regions="mediterranean",
        care="Drought-tough Mediterranean pine. Good on dry, limestone slopes.",
    ),
    "Pinus brutia": dict(common_name="Turkish pine", native_regions="mediterranean"),
    "Pinus radiata": dict(
        common_name="Monterey pine",
        native_regions="n_america",
        invasive_in="australia,s_america,southern_africa",
        warning="Tiny native range in California; a plantation tree (and invader) elsewhere. Prefer a native pine if restoration is the goal.",
    ),
    "Pinus contorta": dict(common_name="Lodgepole pine", native_regions="n_america"),
    "Pinus ponderosa": dict(common_name="Ponderosa pine", native_regions="n_america"),
    "Pinus strobus": dict(
        common_name="Eastern white pine",
        goals="shade,beauty",
        native_regions="n_america",
        care="Fast, tall, soft-needled pine. Needs space and some shelter from salt wind.",
    ),
    "Pinus taeda": dict(common_name="Loblolly pine", native_regions="n_america"),
    "Pinus palustris": dict(common_name="Longleaf pine", native_regions="n_america"),
    "Pinus elliottii": dict(common_name="Slash pine", native_regions="n_america"),
    "Pinus resinosa": dict(common_name="Red pine", native_regions="n_america"),
    "Pinus banksiana": dict(common_name="Jack pine", native_regions="n_america"),
    "Pinus lambertiana": dict(common_name="Sugar pine", native_regions="n_america"),
    "Pinus sabiniana": dict(common_name="Gray pine", native_regions="n_america"),
    "Pinus coulteri": dict(common_name="Coulter pine", native_regions="n_america"),
    "Pinus canariensis": dict(
        common_name="Canary Island pine",
        native_regions="canary",
        care="Drought-tough once established. Popular street tree in warm-summer climates.",
    ),
    "Pinus patula": dict(common_name="Patula pine", native_regions="mexico_ca", warning="Common plantation pine outside Mexico."),
    "Pinus wallichiana": dict(common_name="Himalayan blue pine", native_regions="himalaya"),
    "Pinus roxburghii": dict(common_name="Chir pine", native_regions="himalaya"),
    "Pinus koraiensis": dict(
        common_name="Korean pine",
        goals="food,beauty,shade",
        native_regions="e_asia",
        care="Source of Korean pine nuts. Slow; wants cool summers.",
    ),
    "Pinus sibirica": dict(
        common_name="Siberian pine",
        goals="food,beauty",
        native_regions="siberia,e_asia",
    ),
    "Pinus cembra": dict(
        common_name="Swiss stone pine",
        goals="food,beauty",
        native_regions="europe",
    ),
    "Pinus mugo": dict(common_name="Mountain pine", goals="beauty", native_regions="europe"),
    "Pinus peuce": dict(common_name="Macedonian pine", native_regions="europe"),
    "Acer palmatum": dict(
        common_name="Japanese maple",
        goals="beauty",
        sun="part",
        native_regions="e_asia",
        care="Small ornamental. Shelter from hot afternoon sun and drying wind.",
    ),
    "Acer japonicum": dict(common_name="Fullmoon maple", goals="beauty", sun="part", native_regions="e_asia"),
    "Acer rubrum": dict(
        common_name="Red maple",
        goals="shade,beauty",
        native_regions="n_america",
        care="Fast shade tree. Tolerates wet soil better than most maples.",
    ),
    "Acer saccharum": dict(
        common_name="Sugar maple",
        goals="shade,food,beauty",
        native_regions="n_america",
        care="Classic fall color and maple syrup. Wants deep, well-drained soil; dislikes heat and road salt.",
    ),
    "Acer saccharinum": dict(
        common_name="Silver maple",
        goals="shade",
        native_regions="n_america",
        warning="Brittle wood; keep away from houses and pipes.",
        care="Very fast shade. Roots are aggressive.",
    ),
    "Acer negundo": dict(
        common_name="Boxelder",
        goals="shade",
        native_regions="n_america",
        invasive_in="europe",
        warning="Weedy, weak-wooded. Fine for tough sites, poor as a specimen.",
    ),
    "Acer platanoides": dict(
        common_name="Norway maple",
        goals="shade,beauty",
        native_regions="europe",
        invasive_in="n_america",
        warning="Invasive in much of North America. Choose a native maple there.",
    ),
    "Acer pseudoplatanus": dict(
        common_name="Sycamore maple",
        goals="shade,beauty",
        native_regions="europe",
        warning="Can seed aggressively outside mountain Europe.",
    ),
    "Acer campestre": dict(
        common_name="Field maple",
        goals="beauty,shade",
        native_regions="europe",
        care="A smaller European maple, good for hedges and modest gardens.",
    ),
    "Acer ginnala": dict(common_name="Amur maple", native_regions="e_asia", invasive_in="n_america"),
    "Acer tataricum": dict(common_name="Tatar maple", native_regions="europe,e_asia"),
    "Acer macrophyllum": dict(common_name="Bigleaf maple", native_regions="n_america"),
    "Acer glabrum": dict(common_name="Rocky Mountain maple", native_regions="n_america"),
    "Acer spicatum": dict(common_name="Mountain maple", native_regions="n_america"),
    "Acer griseum": dict(
        common_name="Paperbark maple",
        goals="beauty",
        sun="part",
        native_regions="e_asia",
        care="Grown for cinnamon bark. Slow, small, and needs good drainage.",
    ),
    "Acer buergerianum": dict(common_name="Trident maple", native_regions="e_asia"),
    "Acer monspessulanum": dict(common_name="Montpellier maple", native_regions="mediterranean,europe"),
    "Acer opalus": dict(common_name="Italian maple", native_regions="europe,mediterranean"),
    "Acer cappadocicum": dict(common_name="Cappadocian maple", native_regions="europe,e_asia"),
    "Fagus sylvatica": dict(
        common_name="European beech",
        goals="shade,beauty",
        native_regions="europe",
        care="A cathedral shade tree. Needs moisture; struggles in hot, dry, or compacted sites.",
    ),
    "Fagus grandifolia": dict(
        common_name="American beech",
        goals="shade,beauty",
        native_regions="n_america",
        warning="Beech bark disease is spreading in eastern North America.",
    ),
    "Fagus orientalis": dict(common_name="Oriental beech", native_regions="europe,mediterranean"),
    "Fagus japonica": dict(common_name="Japanese beech", native_regions="e_asia"),
    "Betula pendula": dict(
        common_name="Silver birch",
        goals="beauty",
        native_regions="europe,siberia",
        care="Light canopy, white bark. Needs light and moisture; not for hot, dry yards.",
    ),
    "Betula pubescens": dict(common_name="Downy birch", native_regions="europe,siberia"),
    "Betula alleghaniensis": dict(common_name="Yellow birch", native_regions="n_america"),
    "Betula lenta": dict(common_name="Sweet birch", native_regions="n_america"),
    "Betula papyrifera": dict(common_name="Paper birch", native_regions="n_america"),
    "Betula nigra": dict(common_name="River birch", native_regions="n_america"),
    "Betula neoalaskana": dict(common_name="Alaska birch", native_regions="n_america"),
    "Betula utilis": dict(common_name="Himalayan birch", native_regions="himalaya,e_asia"),
    "Betula ermanii": dict(common_name="Erman's birch", native_regions="e_asia"),
}

COUNTRY_TO_REGION = {
    "US": "n_america", "CA": "n_america",
    "MX": "mexico_ca", "GT": "mexico_ca", "HN": "mexico_ca", "NI": "mexico_ca",
    "CR": "mexico_ca", "PA": "mexico_ca", "BZ": "mexico_ca", "SV": "mexico_ca",
    "GB": "europe", "IE": "europe", "FR": "europe", "DE": "europe", "NL": "europe",
    "BE": "europe", "LU": "europe", "CH": "europe", "AT": "europe", "PL": "europe",
    "CZ": "europe", "SK": "europe", "HU": "europe", "DK": "europe", "SE": "europe",
    "NO": "europe", "FI": "europe", "EE": "europe", "LV": "europe", "LT": "europe",
    "UA": "europe", "BY": "europe", "RO": "europe", "BG": "europe", "RS": "europe",
    "HR": "europe", "SI": "europe", "BA": "europe", "AL": "europe", "MK": "europe",
    "ES": "mediterranean", "PT": "mediterranean", "IT": "mediterranean",
    "GR": "mediterranean", "TR": "mediterranean", "CY": "mediterranean",
    "IL": "mediterranean", "LB": "mediterranean", "SY": "mediterranean",
    "DZ": "mediterranean", "TN": "mediterranean", "MA": "mediterranean",
    "CN": "e_asia", "JP": "e_asia", "KR": "e_asia", "KP": "e_asia", "TW": "e_asia",
    "MN": "e_asia",
    "IN": "himalaya", "NP": "himalaya", "BT": "himalaya", "PK": "himalaya",
    "RU": "siberia",
    "AU": "australia", "NZ": "australia",
    "ZA": "southern_africa", "CL": "s_america", "AR": "s_america", "UY": "s_america",
    "BR": "s_america",
}


def default_common_name(species):
    parts = str(species).split()
    if len(parts) < 2:
        return species
    genus, epithet = parts[0], parts[1]
    english = GENUS_ENGLISH.get(genus, genus.lower())
    return f"{epithet.replace('_', ' ').title()} {english}"


def traits_for(species):
    """Genus defaults, then in-code overrides, then optional CSV row."""
    genus = str(species).split()[0] if species else ""
    row = dict(
        species=species,
        common_name=default_common_name(species),
        goals="beauty",
        sun="full",
        evergreen="no",
        care="Plant in a hole no deeper than the root flare; water in the first two summers.",
        warning="",
        native_regions="",
        invasive_in="",
    )
    row.update(GENUS_DEFAULTS.get(genus, {}))
    row.update(SPECIES_OVERRIDES.get(species, {}))
    row["species"] = species
    return row


def load_saved_models(min_auc=MIN_AUC_DEFAULT, metrics_path=METRICS_PATH):
    """Every status=saved model above min_auc. New GBIF runs just append here."""
    metrics = pd.read_csv(metrics_path)
    saved = metrics[metrics["status"] == "saved"].copy()
    saved = saved[saved["auc"] >= float(min_auc)]
    saved = saved[saved["model_path"].map(os.path.isfile)]
    return saved.reset_index(drop=True)


def load_traits_table(species_list):
    """Merge code defaults with data/species_traits.csv if present."""
    rows = [traits_for(sp) for sp in species_list]
    table = pd.DataFrame(rows)
    if os.path.isfile(TRAITS_CSV):
        extra = pd.read_csv(TRAITS_CSV)
        if "species" in extra.columns:
            extra = extra.drop_duplicates("species")
            table = table.set_index("species")
            extra = extra.set_index("species")
            table.update(extra)
            missing = extra.index.difference(table.index)
            if len(missing):
                table = pd.concat([table, extra.loc[missing]])
            table = table.reset_index()
    return table


def write_traits_stub(species_list, path=TRAITS_CSV):
    """Write an editable CSV so new species can be annotated without code edits."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    load_traits_table(species_list).to_csv(path, index=False)
    return path


def region_for_country(country_code):
    if not country_code:
        return ""
    return COUNTRY_TO_REGION.get(str(country_code).upper(), "")
