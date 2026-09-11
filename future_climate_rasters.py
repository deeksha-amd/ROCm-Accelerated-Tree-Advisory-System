"""
Download FUTURE climate rasters from WorldClim (2050)
CMIP6 projections, 19 bioclimatic variables
"""

import os
import requests
import zipfile
from tqdm import tqdm

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUTPUT_DIR = "climate_future_2050"
RESOLUTION = "10m"        # match your current climate resolution!

# Global Climate Model (GCM) — pick one
GCM = "MPI-ESM1-2-HR"     # a common, well-regarded model

# SSP scenario — emission pathway
SSP = "ssp245"            # options:
                          #   ssp126 = optimistic (low emissions)
                          #   ssp245 = middle-of-road   ← common choice
                          #   ssp370 = high emissions
                          #   ssp585 = worst case

# Time period for "2050"
PERIOD = "2041-2060"      # this represents ~2050

# WorldClim 2.1 future climate URL
BASE_URL = "https://geodata.ucdavis.edu/cmip6"
FILENAME = f"wc2.1_{RESOLUTION}_bioc_{GCM}_{SSP}_{PERIOD}.tif"
DOWNLOAD_URL = f"{BASE_URL}/{RESOLUTION}/{GCM}/{SSP}/{FILENAME}"


def download_file(url, output_path):
    """Download with progress bar."""
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"ERROR: could not download (status {response.status_code})")
        print(f"URL tried: {url}")
        print("Check GCM/SSP/PERIOD spelling against worldclim.org")
        return False

    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    return True


def main():
    print("=" * 50)
    print("FUTURE CLIMATE RASTER DOWNLOADER (2050)")
    print("=" * 50)
    print(f"GCM:        {GCM}")
    print(f"Scenario:   {SSP}")
    print(f"Period:     {PERIOD}")
    print(f"Resolution: {RESOLUTION}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, FILENAME)

    print(f"\nURL: {DOWNLOAD_URL}\n")
    success = download_file(DOWNLOAD_URL, output_path)

    if success:
        print("\n" + "=" * 50)
        print("DONE")
        print("=" * 50)
        print(f"Saved to: {output_path}")
        print("\nNote: future data comes as ONE multi-band file")
        print("      (19 bands = 19 bioclim variables)")


if __name__ == "__main__":
    main()

