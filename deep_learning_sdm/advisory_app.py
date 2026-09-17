"""
Web front end for the deep SDM planting advisory — coordinates in, trees out

What this is
------------
A thin Flask shell over `advisory_service.Advisory`. Every judgement about
coverage, land use, ranking and reliability lives in the service module; this
file only turns HTTP into a function call and back, which is why it is short
and why it has no opinion of its own.

    ./venv/bin/python advisory_app.py          # then open http://127.0.0.1:8000

Startup cost
------------
The checkpoint, the country polygon, the 18.5 km land-use mask and 61 open
raster handles are all built once in `create_app()`, before the first request
is served. The first run per checkpoint also scores 20,000 random covered
cells to give every species a reference distribution to be a percentile of —
about five seconds — and caches it under `advisory_cache/`, so later starts are
roughly a second. A point query after that is 61 one-pixel reads and one
64-wide forward pass, comfortably under 20 ms on CPU.

Device
------
`--device` follows deep_sdm_training.py's convention and defaults to `auto`,
which resolves to the MI210 when one is visible. `/api/health` reports the
resolved device as well as the requested one, because reporting "cpu" because
that was the hardcoded default — while a GPU sat there unused — is exactly the
kind of thing a health check exists to catch. Pass `--device cpu` when the card
is busy training: one row through a 64-wide two-layer MLP is far below the
point where a GPU helps.

Endpoints
---------
    GET /                         the page
    GET /api/health               what loaded, what failed, which device
    GET /api/coverage             the regions this build can answer for
    GET /api/recommend?lat=&lon=  the advice, as JSON

The advice is capped at five species by default — three the model matches to
this specific place and two dependable region-wide ones — with invasives
reported outside that count. The cap lives in `advisory_service.rank()` rather
than in the page, so the JSON and the page can never disagree about what the
answer is. `?top=N` raises it for a script that wants the long list.

Failure behaviour
-----------------
The server starts even with no usable checkpoint — `/api/health` then reports
503 and says which run failed and why, which is a far more useful thing to
find than a process that exited during boot. Bad coordinates, points outside
coverage, points on water or in a city, and cells with no predictor data are
all normal 200 responses carrying a `status` the page renders differently;
only a genuinely broken server is a 5xx.
"""

import argparse
import os
import traceback

from flask import Flask, jsonify, request, send_from_directory

from advisory_service import (CACHE_DIR, DEVICE, REFERENCE_CELLS,
                              TOP_RECOMMENDATIONS, Advisory, divider)

# ─────────────────────────────────────────────────────
# CONFIG — server
# ─────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8000
# advisory_web/ is code, not data: it ships inside deep_learning_sdm/ and moves
# with this file. Data paths elsewhere in the project stay relative to the
# working directory, which is the repo root; this one is anchored to the module
# directory so the page is still found when the app is launched from the root.
WEB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "advisory_web")
# The cap is applied in the service, not in the page, so that the API and the
# page cannot drift: whatever a user sees is what /api/recommend returned. An
# API caller who genuinely wants the long list asks for it with ?top=N, which
# keeps the default honest without making the endpoint useless for scripting.
DEFAULT_TOP = TOP_RECOMMENDATIONS
MAX_TOP = 96

TRUTHY = {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────
def create_app(device=DEVICE, reference_cells=REFERENCE_CELLS,
               output_root=None, verbose=True):
    """Build the Flask app with the advisory already warm.

    Loading here rather than lazily on first request is the whole point: a
    click on the map must not pay for a checkpoint load, and a broken
    checkpoint must be visible at startup rather than to the first user.
    """
    app = Flask(__name__, static_folder=None)
    kwargs = {"device": device, "reference_cells": reference_cells,
              "verbose": verbose}
    if output_root:
        kwargs["output_root"] = output_root

    try:
        advisory = Advisory(**kwargs)
        boot_error = None
    except Exception as error:                           # noqa: BLE001
        # A missing or corrupt checkpoint, a missing raster, a torch failure —
        # all of it lands here and becomes a diagnosable /api/health rather
        # than a traceback on a dead process.
        advisory = None
        boot_error = f"{type(error).__name__}: {error}"
        print("advisory failed to load:\n" + traceback.format_exc())

    app.config["ADVISORY"] = advisory
    app.config["BOOT_ERROR"] = boot_error

    @app.get("/")
    def index():
        return send_from_directory(WEB_ROOT, "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename):
        return send_from_directory(WEB_ROOT, filename)

    @app.get("/api/health")
    def health():
        ready = advisory is not None and advisory.ready
        body = {
            "ready": ready,
            "boot_error": boot_error,
            "regions": advisory.coverage_summary() if advisory else [],
            "failures": advisory.failures if advisory else [],
            # Both, because "auto" is what was asked for and "cuda"/"cpu" is
            # what is actually in use. The container health check was
            # reporting the request as though it were the outcome.
            "device_requested": device,
            "device": advisory.device if advisory else None,
            "top_default": DEFAULT_TOP,
        }
        if not ready and not body["failures"] and not boot_error:
            body["boot_error"] = (
                "No trained checkpoint was found. Train one with "
                "deep_sdm_training.py --profile fra_30s, or point the server "
                "at an output directory that has a deep_sdm.pt in it.")
        return jsonify(body), (200 if ready else 503)

    @app.get("/api/coverage")
    def coverage():
        if advisory is None:
            return jsonify({"regions": [], "boot_error": boot_error}), 503
        return jsonify({"regions": advisory.coverage_summary()})

    @app.get("/api/recommend")
    def recommend():
        if advisory is None:
            return jsonify({
                "status": "unavailable",
                "headline": "The advisory did not start",
                "detail": boot_error or "No checkpoint could be loaded.",
            }), 503

        # `limit` is still accepted as the old name for `top`, so an existing
        # bookmark or script does not start silently getting a different
        # number of species than it asked for.
        top = _int_arg("top", _int_arg("limit", DEFAULT_TOP, 1, MAX_TOP),
                       1, MAX_TOP)
        override = str(request.args.get("override", "")).lower() in TRUTHY
        nearby = str(request.args.get("nearby", "1")).lower() in TRUTHY
        answer = advisory.recommend(request.args.get("lat"),
                                    request.args.get("lon"),
                                    top=top, override_exclusion=override,
                                    nearby=nearby)
        code = {"bad_request": 400, "unavailable": 503}.get(answer["status"],
                                                            200)
        return jsonify(answer), code

    @app.errorhandler(Exception)
    def unhandled(error):
        # Anything that reaches here is a real bug, so it is logged in full and
        # reported as a server error rather than dressed up as advice.
        print(traceback.format_exc())
        return jsonify({"status": "error",
                        "headline": "The server hit an unexpected problem",
                        "detail": f"{type(error).__name__}: {error}"}), 500

    return app


def _int_arg(name, default, low, high):
    try:
        return max(low, min(high, int(request.args.get(name, default))))
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="serve the deep SDM tree-planting advisory over HTTP")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--device", default=DEVICE,
                        help="cpu (default), cuda, or auto")
    parser.add_argument("--reference-cells", type=int, default=REFERENCE_CELLS,
                        help="cells sampled to build per-species percentiles")
    parser.add_argument("--output-dir", default=None,
                        help="root to discover trained runs under "
                             "(default deep_sdm_output)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isdir(WEB_ROOT):
        raise SystemExit(f"{WEB_ROOT}/ is missing; the page cannot be served")

    app = create_app(device=args.device, reference_cells=args.reference_cells,
                     output_root=args.output_dir)
    divider("SERVER")
    advisory = app.config["ADVISORY"]
    if advisory is None or not advisory.ready:
        print("WARNING          no model is loaded; /api/recommend will "
              "return 503 and the page will say so")
    print(f"cache            {CACHE_DIR}/")
    print(f"listening        http://{args.host}:{args.port}")
    # threaded=True because a request does 61 small raster reads; the service
    # holds a lock around those, so concurrency is safe and the page stays
    # responsive while one query runs.
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()
