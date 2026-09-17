"""
One entry point for the whole occurrence-data pipeline

Runs the four stages below in order, skipping any whose output already exists,
so an interrupted run is restarted with the same command.

    countries   measure candidate countries            ~2 min,  no records
    plan        resolve the species list, size the job  ~2 min,  no records
    download    fetch occurrences by taxonKey          ~25 min,  ~25 MB
    clean       drop bad records, thin to the grid      ~1 min
    effort      sampling-effort raster + background     ~2 min

Everything is scoped to one country, set by COUNTRY in download_species.py or
by --country here. The species list is tree_species_list.csv, which is meant
to be edited.

Usage:

    python3 prepare_species_data.py --countries   # which country to use
    python3 prepare_species_data.py --plan        # look before you leap
    python3 prepare_species_data.py --all         # everything, resumable
    python3 prepare_species_data.py --all --country GB
    python3 prepare_species_data.py --stage clean --stage effort
    python3 prepare_species_data.py --all --force # ignore existing outputs

Nothing here touches gbif_500_species.csv, soil_data/, topography/, satellite/,
climate_*/ or validation/. All new output goes to species_occurrences/ and
sampling_effort/.

The stages are also runnable directly — download_species.py, clean_species.py
and sampling_effort.py all have their own flags, and the runbook
(SPECIES_DATA_RUNBOOK.md) lists them. This file exists so that the normal case
is one command.
"""

import argparse
import os
import subprocess
import sys
import time

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
from download_species import COUNTRY

# This file lives in deep_learning_sdm/; the species_occurrences/ and
# sampling_effort/ trees it writes live at the repo root. Stage scripts are
# located under HERE and run with their working directory set to DATA_ROOT.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.dirname(HERE)

# name -> (argv, output that proves it finished, one-line description).
# Outputs carry the country, so switching country reruns everything rather
# than silently skipping stages that were built for somewhere else.
def stages(country):
    base = f"species_occurrences/{country}"
    return {
        "countries": (["download_species.py", "--countries"],
                      "species_occurrences/country_comparison.csv",
                      "measure each candidate country, download no records"),
        "plan": (["download_species.py", "--plan"],
                 f"{base}/species_plan.json",
                 "resolve the species list and size the job"),
        "download": (["download_species.py", "--download"],
                     f"{base}/gbif_trees_{country}_raw.csv",
                     "fetch occurrence records by taxonKey"),
        "clean": (["clean_species.py"],
                  f"{base}/gbif_trees_{country}_thinned.csv",
                  "drop bad records and thin to the 18.5 km grid"),
        "effort": (["sampling_effort.py", "--surface", "--background"],
                   f"sampling_effort/background_{country}.csv",
                   "effort raster and target-group background points"),
    }


STAGE_ORDER = ["countries", "plan", "download", "clean", "effort"]
DEFAULT_STAGES = ["plan", "download", "clean", "effort"]


def run_stage(name, country, force=False, extra=None):
    argv, output, description = stages(country)[name]
    argv = argv + ["--country", country]

    if output and os.path.exists(output) and not force:
        print(f"\n### {name}: SKIPPED — {output} already exists "
              f"(use --force to rebuild)")
        return True

    print(f"\n{'=' * 62}\n### {name}: {description}\n{'=' * 62}")
    command = ([sys.executable, os.path.join(HERE, argv[0])] + list(argv[1:])
               + list(extra or []))
    print(f"$ {' '.join(command)}\n")

    start = time.monotonic()
    # cwd is pinned to DATA_ROOT, the repo root, so the child resolves
    # species_occurrences/ and sampling_effort/ the same way this file does.
    # Pinning it also keeps a stray /tmp/enum.py from shadowing the stdlib,
    # which has happened before.
    result = subprocess.run(command, cwd=DATA_ROOT)
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        print(f"\n### {name}: FAILED after {elapsed / 60:.1f} min "
              f"(exit {result.returncode})")
        return False

    print(f"\n### {name}: done in {elapsed / 60:.1f} min")
    if output and not os.path.exists(output):
        print(f"### warning: expected {output} but it is not there")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build the training-ready tree occurrence dataset.")
    parser.add_argument("--all", action="store_true",
                        help="run plan, download, clean and effort in order")
    parser.add_argument("--plan", action="store_true",
                        help="run the planning stage only (no records)")
    parser.add_argument("--countries", action="store_true",
                        help="compare candidate countries only")
    parser.add_argument("--stage", action="append", choices=STAGE_ORDER,
                        default=[], help="run specific stages, repeatable")
    parser.add_argument("--country", default=COUNTRY,
                        help=f"ISO country code (default {COUNTRY})")
    parser.add_argument("--force", action="store_true",
                        help="rerun stages whose output already exists")
    parser.add_argument("--status", action="store_true",
                        help="report which outputs exist, then exit")
    args = parser.parse_args()

    country = args.country.upper()
    table = stages(country)

    if args.status:
        print(f"{'stage':10s} {'output':52s} {'size':>10s}")
        for name in STAGE_ORDER:
            output = table[name][1]
            size = (f"{os.path.getsize(output) / 1e6:.1f} MB"
                    if os.path.exists(output) else "-")
            print(f"{name:10s} {output:52s} {size:>10s}")
        return

    if args.countries:
        requested = ["countries"]
    elif args.plan:
        requested = ["plan"]
    elif args.all:
        requested = DEFAULT_STAGES
    elif args.stage:
        requested = [s for s in STAGE_ORDER if s in args.stage]
    else:
        parser.print_help()
        return

    print("=" * 62)
    print(f"TREE OCCURRENCE DATA PIPELINE — {country}")
    print("=" * 62)
    print(f"Stages: {' -> '.join(requested)}")

    for name in requested:
        if not run_stage(name, country, force=args.force):
            raise SystemExit(1)

    thinned = table["clean"][1]
    background = table["effort"][1]

    print("\n" + "=" * 62)
    print("PIPELINE COMPLETE")
    print("=" * 62)
    for name in requested:
        output = table[name][1]
        if os.path.exists(output):
            print(f"  {output}  ({os.path.getsize(output) / 1e6:.1f} MB)")
    if os.path.exists(thinned):
        print(f"\nTrain on {thinned} with background from {background}.")
        print("See SPECIES_DATA_RUNBOOK.md for what to check.")


if __name__ == "__main__":
    main()
