"""
Download CURRENT climate rasters from WorldClim
19 bioclimatic variables, 1970-2000 average
"""

import os
import sys
import requests
import zipfile
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from repo_paths import data

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
OUTPUT_DIR = data("climate_current")
RESOLUTION = "10m"   # options: "10m", "5m", "2.5m", "30s"
                     # 10m = coarse/fast, 30s = fine/slow+large

# WorldClim 2.1 current climate URL
BASE_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
FILENAME = f"wc2.1_{RESOLUTION}_bio.zip"
DOWNLOAD_URL = f"{BASE_URL}/{FILENAME}"


def download_file(url, output_path):
    """Download with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True,
                  desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def main():
    print("=" * 50)
    print("CURRENT CLIMATE RASTER DOWNLOADER")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    zip_path = os.path.join(OUTPUT_DIR, FILENAME)

    # Download
    print(f"\nResolution: {RESOLUTION}")
    print(f"URL: {DOWNLOAD_URL}\n")
    download_file(DOWNLOAD_URL, zip_path)

    # Extract
    print("\nExtracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(OUTPUT_DIR)

    os.remove(zip_path)  # clean up zip

    # List extracted files
    tif_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.tif')]
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Downloaded {len(tif_files)} climate variables:")
    for f in sorted(tif_files):
        print(f"  {f}")
    print(f"\nSaved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

