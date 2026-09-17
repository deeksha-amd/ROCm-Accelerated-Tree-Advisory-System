"""Self-contained Leaflet HTML for one recommend.py result."""

from __future__ import annotations

import json
import os

import numpy as np


def _safe(obj):
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    return obj


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Tree suggestions — __TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
  :root {
    --ink: #12202b;
    --muted: #5b6b76;
    --paper: #f4f1ea;
    --card: #fffdf8;
    --line: #d9d1c3;
    --today: #1f6f4a;
    --future: #c05621;
    --blocked: #9b2c2c;
    --warn: #b7791f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    background: var(--paper);
    color: var(--ink);
  }
  header {
    padding: 18px 28px 8px;
    border-bottom: 1px solid var(--line);
    background: #efe8d8;
  }
  header h1 { margin: 0 0 4px; font-size: 1.45rem; font-weight: 600; }
  header p { margin: 0; color: var(--muted); font-size: 0.95rem; }
  .layout {
    display: grid;
    grid-template-columns: minmax(320px, 1fr) minmax(380px, 420px);
    min-height: calc(100vh - 88px);
  }
  #map { min-height: 420px; background: #cdd9c8; }
  aside {
    padding: 18px 22px 32px;
    overflow: auto;
    border-left: 1px solid var(--line);
    background: var(--card);
  }
  .note { font-size: 0.88rem; color: var(--muted); margin: 8px 0 14px; line-height: 1.4; }
  .mix { display: flex; height: 18px; border-radius: 9px; overflow: hidden; background: #ece6da; margin: 8px 0 4px; }
  .mix span { display: block; height: 100%; }
  .legend { font-size: 0.78rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: 8px 12px; margin-bottom: 16px; }
  .legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; }
  .status {
    display: inline-block;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 999px;
    margin-bottom: 10px;
  }
  .status.plantable { background: #d9eddf; color: #1f6f4a; }
  .status.city, .status.farm { background: #f6e4c1; color: #8a5a10; }
  .status.blocked { background: #f4d6d6; color: var(--blocked); }
  .card {
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px 14px 10px;
    margin: 0 0 12px;
    background: #fff;
  }
  .card h2 { margin: 0 0 2px; font-size: 1.05rem; }
  .latin { color: var(--muted); font-size: 0.82rem; font-style: italic; }
  .badge {
    float: right;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 3px 7px;
    border-radius: 999px;
    background: #e7f3ec;
    color: var(--today);
  }
  .badge.risky { background: #f8e6d8; color: var(--future); }
  .row { display: grid; grid-template-columns: 52px 1fr 44px; gap: 8px; align-items: center; font-size: 0.82rem; margin-top: 6px; }
  .track { height: 9px; background: #eee8dc; border-radius: 6px; overflow: hidden; }
  .track i { display: block; height: 100%; border-radius: 6px; }
  .track.today i { background: var(--today); }
  .track.future i { background: var(--future); }
  .pct { text-align: right; font-variant-numeric: tabular-nums; }
  .why, .care, .fi { font-size: 0.88rem; line-height: 1.35; margin: 8px 0 0; }
  .fi { color: var(--muted); font-size: 0.8rem; }
  .care { color: var(--muted); }
  .warn { color: var(--blocked); font-size: 0.85rem; margin-top: 6px; }
  .future-line { font-size: 0.85rem; margin-top: 6px; color: #7a3e12; }
  footer { font-size: 0.75rem; color: var(--muted); margin-top: 18px; line-height: 1.35; }
  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    aside { border-left: 0; border-top: 1px solid var(--line); }
    #map { height: 320px; }
  }
</style>
</head>
<body>
<header>
  <h1 id="headline">Tree suggestions</h1>
  <p id="place"></p>
</header>
<div class="layout">
  <div id="map"></div>
  <aside id="panel"></aside>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __DATA__;

function pct(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Math.round(v * 100) + "%";
}
function clamp01(v) { return Math.max(0, Math.min(1, v || 0)); }

const coords = (typeof DATA.lat === "number" && typeof DATA.lon === "number")
  ? DATA.lat.toFixed(4) + ", " + DATA.lon.toFixed(4)
  : "";
const place = [DATA.address, coords, DATA.goal && DATA.goal !== "any" ? "goal: " + DATA.goal : ""]
  .filter(Boolean).join("  ·  ");
document.getElementById("headline").textContent = DATA.title || "Tree suggestions";
document.getElementById("place").textContent = place || coords;

const cell = DATA.cell || {};
const status = cell.status || "plantable";
const color = cell.color || "#1f6f4a";
const map = L.map("map", { scrollWheelZoom: true });
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Tiles &copy; Esri",
  maxZoom: 12
}).addTo(map);
if (cell.south !== undefined) {
  const bounds = [[cell.south, cell.west], [cell.north, cell.east]];
  L.rectangle(bounds, { color, weight: 2, fillColor: color, fillOpacity: 0.28 }).addTo(map);
  map.fitBounds(bounds, { padding: [40, 40], maxZoom: 8 });
} else {
  map.setView([DATA.lat, DATA.lon], 7);
}
const grid = DATA.grid || {};
const markerNote = grid.marker || "Score is for this ~18 km climate cell, not a backyard.";
L.marker([DATA.lat, DATA.lon]).addTo(map)
  .bindPopup(markerNote);

const sat = (DATA.satellite && DATA.satellite.values) || {};
const parts = [
  ["water", sat.water, "#3b82f6"],
  ["built", sat.built, "#6b7280"],
  ["crop", sat.crop, "#ca8a04"],
  ["tree", sat.tree, "#16a34a"],
  ["snow", sat.snow, "#94a3b8"]
];
let used = 0;
let mixHtml = "";
let legend = "";
parts.forEach(([name, val, col]) => {
  const v = clamp01(val);
  if (v < 0.02) return;
  used += v;
  mixHtml += `<span style="width:${v*100}%;background:${col}" title="${name} ${pct(v)}"></span>`;
  legend += `<span><i style="background:${col}"></i>${name} ${pct(v)}</span>`;
});
const other = Math.max(0, 1 - used);
if (other > 0.02) {
  mixHtml += `<span style="width:${other*100}%;background:#e7e1d4" title="other ${pct(other)}"></span>`;
  legend += `<span><i style="background:#e7e1d4"></i>other ${pct(other)}</span>`;
}

const notes = ((DATA.satellite && DATA.satellite.notes) || []).map(n => `<div class="note">${n}</div>`).join("");
const blocked = DATA.satellite && DATA.satellite.blocked;
const futureMeta = DATA.future || {};
let cards = "";
if (blocked) {
  cards = `<p class="warn">No planting list — satellite says this pin is not plantable.</p>`;
} else if (!DATA.picks || !DATA.picks.length) {
  cards = `<p class="note">No species cleared the record-likeness + goal filters.</p>`;
} else {
  DATA.picks.forEach((p, i) => {
    const risky = (p.confidence === "Risky" || p.confidence === "Weak likeness") ? " risky" : "";
    const fut = (p.p_2050 === null || p.p_2050 === undefined)
      ? ""
      : `<div class="row"><span>2050</span><div class="track future"><i style="width:${clamp01(p.p_2050)*100}%"></i></div><span class="pct">${pct(p.p_2050)}</span></div>
         <div class="future-line">${p.future_note || ""}</div>`;
    const fi = (p.feature_importance || []).slice(0, 5);
    const fiLead = DATA.importance_caption
      || "Species-wide XGBoost gain (not this pin):";
    const fiHtml = fi.length
      ? `<p class="fi">${fiLead} ${fi.map(x => x.label + " " + Math.round((x.share||0)*100) + "%").join(" · ")}</p>`
      : "";
    cards += `<article class="card">
      <span class="badge${risky}">${p.confidence || ""}</span>
      <h2>${i+1}. ${p.common_name}</h2>
      <div class="latin">${p.species}</div>
      <div class="row"><span>Today</span><div class="track today"><i style="width:${clamp01(p.p)*100}%"></i></div><span class="pct">${pct(p.p)}</span></div>
      ${fut}
      <p class="why">${p.reason || ""}</p>
      ${fiHtml}
      <p class="care">${p.native || ""} ${p.care || ""}</p>
      ${p.warning ? `<p class="warn">${p.warning}</p>` : ""}
    </article>`;
  });
}

const avoid = DATA.avoid || [];
let avoidHtml = "";
if (avoid.length) {
  avoidHtml = `<p class="warn">Do not plant — this cell looks like records of invasive or naturalised trees:</p>`
    + avoid.map(a => `<p class="warn">• ${a.common_name} <span class="latin">${a.species}</span> (p=${pct(a.p)})${a.warning ? " — " + a.warning : ""}</p>`).join("");
}

const disclaimer = DATA.disclaimer
  || "p is how much this climate–soil cell looks like recorded sites of that species, not a planting permit.";

document.getElementById("panel").innerHTML = `
  <div class="status ${status}">${status.replace("_", " ")} cell</div>
  <p>${DATA.site || ""}</p>
  <p class="note">${grid.note || "The box on the map is one 1/6° cell (~18 km). Shade, frost pockets and watering in a yard are not in the model."}</p>
  ${mixHtml ? `<div class="mix">${mixHtml}</div><div class="legend">${legend}</div>` : ""}
  ${notes}
  ${cards}
  ${avoidHtml}
  <footer>
    ${disclaimer}
    ${futureMeta.available
      ? "2050 uses CMIP6 " + (futureMeta.ssp || "ssp245") + " " + (futureMeta.period || "2041–2060") + "; soil and elevation stay as they are. Same XGBoost models, new BIO values — not a second training run."
      : "2050 bars omitted (future climate GeoTIFF not on disk)."}
    Feature-gain lines are species-wide XGBoost gain, not a local explanation of this pin.
    Tree cover is a satellite filter, not an XGBoost feature.
  </footer>
`;
</script>
</body>
</html>
"""


def write_html(result, path, address=None):
    payload = _safe(result)
    if address:
        payload["address"] = address
    title = payload.get("address") or f"{payload.get('lat')}, {payload.get('lon')}"
    html = TEMPLATE.replace("__TITLE__", str(title).replace("&", "and")[:80])
    html = html.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
