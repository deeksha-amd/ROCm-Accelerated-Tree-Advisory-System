"""
GBIF Data Collector — FIXED VERSION
Properly finds 500 UNIQUE valid species with occurrence data
"""

from pygbif import species as gbif_species
from pygbif import occurrences as occ
import pandas as pd
from tqdm import tqdm
import time

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
N_SPECIES = 500
MIN_RECORDS = 30              # LOWERED from 100 (more species qualify)
MAX_RECORDS_PER_SPECIES = 3000
OUTPUT_FILE = "gbif_500_species.csv"

# MORE genera so we can reach 500 unique species
TREE_GENERA = [
    "Quercus", "Pinus", "Acer", "Fagus", "Betula",
    "Populus", "Salix", "Fraxinus", "Ulmus", "Tilia",
    "Abies", "Picea", "Larix", "Carpinus", "Alnus",
    "Prunus", "Sorbus", "Juniperus", "Eucalyptus", "Cedrus",
    "Magnolia", "Malus", "Corylus", "Castanea", "Platanus",
    "Aesculus", "Robinia", "Catalpa", "Liquidambar", "Nyssa",
    "Cornus", "Crataegus", "Ilex", "Rhamnus", "Viburnum",
    "Eucalyptus", "Acacia", "Ficus", "Cinnamomum", "Terminalia"
]


def get_species_list(genera, target_count):
    """Find UNIQUE, VALID species with proper deduplication."""
    seen_keys = set()        # track unique species keys
    seen_names = set()       # track unique species names
    species_list = []

    for genus in genera:
        print(f"Searching genus: {genus}  (found so far: {len(species_list)})")

        try:
            results = gbif_species.name_lookup(
                q=genus,
                rank="SPECIES",
                status="ACCEPTED",
                limit=200                    # more results per genus
            )
        except Exception as e:
            print(f"  Error searching {genus}: {e}")
            continue

        for result in results.get('results', []):
            # Use the CANONICAL name and the SPECIES key
            sci_name = result.get('canonicalName') or result.get('scientificName', '')
            species_key = result.get('nubKey') or result.get('key')

            # ── DEDUPLICATION ──
            if not sci_name or not species_key:
                continue
            if species_key in seen_keys:      # skip duplicate key
                continue
            if sci_name in seen_names:        # skip duplicate name
                continue

            # ── VALIDATION: must be a proper 2-word species name ──
            words = sci_name.split()
            if len(words) < 2:                # skip "Quercus sp."
                continue
            if any(x in sci_name.lower() for x in ['sp.', 'subsp.', 'var.', '×']):
                continue                      # skip subspecies/hybrids

            seen_keys.add(species_key)
            seen_names.add(sci_name)
            species_list.append({
                'name': sci_name,
                'key': species_key
            })

            if len(species_list) >= target_count:
                print(f"\nReached target: {target_count} unique species")
                return species_list[:target_count]

    print(f"\nCollected {len(species_list)} unique species "
          f"(wanted {target_count})")
    return species_list


def count_occurrences(species_key):
    """Quick check: how many records does this species have?"""
    try:
        response = occ.search(
            taxonKey=species_key,
            hasCoordinate=True,
            hasGeospatialIssue=False,
            limit=0                          # just get the count
        )
        return response.get('count', 0)
    except Exception:
        return 0


def download_occurrences(species_name, species_key):
    """Download occurrence records for one species."""
    all_records = []
    offset = 0
    page_size = 300

    while offset < MAX_RECORDS_PER_SPECIES:
        try:
            response = occ.search(
                taxonKey=species_key,
                hasCoordinate=True,
                hasGeospatialIssue=False,
                limit=page_size,
                offset=offset
            )
        except Exception as e:
            print(f"  Error for {species_name}: {e}")
            break

        results = response.get('results', [])
        if not results:
            break

        for r in results:
            lat = r.get('decimalLatitude')
            lon = r.get('decimalLongitude')
            uncertainty = r.get('coordinateUncertaintyInMeters')

            if lat is not None and lon is not None:
                if uncertainty is None or uncertainty <= 10000:
                    all_records.append({
                        'species': species_name,
                        'latitude': lat,
                        'longitude': lon,
                        'year': r.get('year'),
                        'country': r.get('countryCode'),
                        'basis': r.get('basisOfRecord')
                    })

        offset += page_size
        time.sleep(0.05)

        if response.get('endOfRecords', False):
            break

    return all_records


def main():
    print("=" * 50)
    print("GBIF DATA COLLECTOR — FIXED VERSION")
    print("=" * 50)

    # Step 1: build UNIQUE species list
    print("\n[1/3] Building UNIQUE species list...")
    species = get_species_list(TREE_GENERA, N_SPECIES)
    print(f"Unique species found: {len(species)}")

    # Step 2: PRE-FILTER by occurrence count (fast check)
    print("\n[2/3] Pre-filtering species with enough records...")
    valid_species = []
    for sp in tqdm(species, desc="Checking counts"):
        count = count_occurrences(sp['key'])
        if count >= MIN_RECORDS:
            sp['count'] = count
            valid_species.append(sp)
        time.sleep(0.03)

    print(f"Species with >= {MIN_RECORDS} records: {len(valid_species)}")

    # Step 3: download occurrences for valid species only
    print("\n[3/3] Downloading occurrences...")
    all_data = []
    kept_species = 0

    for sp in tqdm(valid_species, desc="Downloading"):
        records = download_occurrences(sp['name'], sp['key'])
        if len(records) >= MIN_RECORDS:
            all_data.extend(records)
            kept_species += 1

    # Save
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Species kept:    {kept_species}")
    print(f"Total records:   {len(df):,}")
    print(f"Saved to:        {OUTPUT_FILE}")
    if kept_species > 0:
        print(f"Records/species: {len(df) // kept_species:,} avg")


if __name__ == "__main__":
    main()