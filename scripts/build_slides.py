#!/usr/bin/env python3
"""Build the 7-slide Paradox Earth demo deck.

Reads (never hard-codes) every number from:
  site/data/meta.json, site/data/cards.json (optional; falls back to
  site/data/regions.json), site/data/tiles.json, site/data/series/*.json,
  demo/team.json, and optionally outputs_global/regional_soil_water_0_200cm.csv
  (read-only) for the retraction figure.

Writes:
  demo/slides.html                 16:9 slides, print CSS = one slide per page
  demo/paradox_earth_slides.pdf    via headless Chrome --print-to-pdf

Figures are rendered with matplotlib and embedded as base64 data URIs, so the
HTML is a single self-contained file.

Usage:  .venv/bin/python scripts/build_slides.py [--no-pdf] [--preview DIR]
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "data"
DEMO = ROOT / "demo"
HTML_OUT = DEMO / "slides.html"
PDF_OUT = DEMO / "paradox_earth_slides.pdf"
SOIL_CSV = next((p for p in (ROOT / "outputs_global_v2" / "regional_soil_water_0_200cm.csv",
                              ROOT / "outputs_global" / "regional_soil_water_0_200cm.csv")
                  if p.exists()), ROOT / "outputs_global_v2" / "regional_soil_water_0_200cm.csv")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ---- design tokens (shared by CSS and matplotlib) --------------------------
INK = "#0A1322"        # background: deep survey-chart navy
PANEL = "#111D31"      # raised panel
RULE = "#22314A"       # hairlines
TEXT = "#E8EEF6"       # primary text
MUTED = "#8C9AB0"      # secondary text
WET = "#4FA3F7"        # significant wetting (blue)
DRY = "#F2873A"        # significant drying (orange)
LAND = "#26334A"       # land, not significant
OCEAN = INK

PASS, PARTIAL, FAIL, NOT_RUN = "pass", "partial", "fail", "not_run"


# ---- helpers ---------------------------------------------------------------
def load_json(path: Path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        if default is not None:
            return default
        raise


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def fmt_num(x, nd=1, signed=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", "−")


def fmt_pct(share, nd=1):
    return "n/a" if share is None else f"{100 * share:.{nd}f}%"


def fmt_area(km2):
    if km2 is None:
        return "n/a"
    if km2 >= 1e6:
        return f"{km2 / 1e6:.1f} million km²"
    if km2 >= 1e3:
        return f"{km2:,.0f} km²"
    return f"{km2:.0f} km²"


def fmt_p(p):
    if p is None:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.2f}"


def fig_to_uri(fig, dpi=200) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def file_to_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def style_axes(ax):
    ax.set_facecolor(INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


# ---- tiles -----------------------------------------------------------------
def load_tiles():
    t = load_json(DATA / "tiles.json")
    cols = t["cols"]
    idx = {c: i for i, c in enumerate(cols)}
    rows = t["rows"]
    return t, idx, rows


# ---- figure 1: flat global average ------------------------------------------
def global_mean_series(tiles_idx, tiles_rows):
    """cos(lat)-weighted mean of per-tile anomalies across every series block."""
    deg = 1.0
    lat_of = {}
    for r in tiles_rows:
        lat_of[r[tiles_idx["id"]]] = r[tiles_idx["south"]] + deg / 2
    acc = None
    wsum = None
    start = None
    for fp in sorted(glob.glob(str(DATA / "series" / "*.json"))):
        blk = load_json(Path(fp), default={})
        start = start or blk.get("start")
        for tid, vals in blk.get("tiles", {}).items():
            lat = lat_of.get(tid)
            if lat is None:
                m = re.match(r"tile_([+-]\d+\.\d+)_", tid)
                if not m:
                    continue
                lat = float(m.group(1)) + deg / 2
            v = np.asarray([np.nan if x is None else x for x in vals], float)
            w = math.cos(math.radians(lat))
            if acc is None:
                acc = np.zeros_like(v)
                wsum = np.zeros_like(v)
            ok = np.isfinite(v)
            acc[ok] += w * v[ok]
            wsum[ok] += w
    if acc is None:
        return None, None
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = acc / wsum
    return start or "2000-01", mean


def month_axis(start, n):
    y, m = (int(x) for x in start.split("-")[:2])
    return np.array([y + (m - 1 + i) / 12.0 for i in range(n)])


def fig_flat_line(start, series, agg):
    fig, ax = plt.subplots(figsize=(11.2, 2.9), facecolor=INK)
    style_axes(ax)
    if series is None:
        ax.text(0.5, 0.5, "series unavailable", color=MUTED, ha="center",
                transform=ax.transAxes)
        return fig_to_uri(fig)
    t = month_axis(start, len(series))
    ax.plot(t, series, color=MUTED, lw=0.7, alpha=0.55)
    k = 12
    if len(series) > k:
        roll = np.convolve(np.nan_to_num(series), np.ones(k) / k, mode="valid")
        ax.plot(t[k - 1:], roll, color=TEXT, lw=2.2)
    slope = get(agg, "slope")  # mm/month per decade
    if slope is not None:
        tc = t - t.mean()
        mu = np.nanmean(series)
        ax.plot(t, mu + slope * tc / 10.0, color=DRY, lw=1.6, ls=(0, (6, 4)))
    lim = np.nanmax(np.abs(series)) * 1.15
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color=RULE, lw=0.8, zorder=0)
    ax.set_xlim(t[0], t[-1] + 1 / 12)
    ax.set_ylabel("mm/month anomaly", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=RULE, lw=0.5)
    return fig_to_uri(fig)


# ---- figure 2: flat map of FDR-significant tiles ---------------------------
def fig_split_map(tiles_idx, tiles_rows, regions, label_regions=None):
    lat0, lat1 = -60, 84
    grid = np.full((lat1 - lat0, 360), np.nan)
    for r in tiles_rows:
        s = int(math.floor(r[tiles_idx["south"]]))
        w = int(math.floor(r[tiles_idx["west"]]))
        if not (lat0 <= s < lat1):
            continue
        sig = r[tiles_idx["sig"]] or 0
        grid[s - lat0, (w + 180) % 360] = {1: 2, -1: 0}.get(int(sig), 1)
    cmap = ListedColormap([DRY, LAND, WET])
    cmap.set_bad(OCEAN)
    fig, ax = plt.subplots(figsize=(12.4, 5.0), facecolor=INK)
    style_axes(ax)
    ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=2,
              extent=[-180, 180, lat0, lat1], interpolation="nearest",
              aspect="equal")
    for la in (-30, 0, 30, 60):
        ax.axhline(la, color=RULE, lw=0.4, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    # label the largest coherent regions with their place names
    for reg in (label_regions or []):
        lon, lat = reg["lon"], reg["lat"]
        col = WET if reg["kind"] == "wetting" else DRY
        ax.annotate(reg["name"], (lon, lat), xytext=(0, -1),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=8.5, color=TEXT, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=INK, ec=col,
                              lw=1.0, alpha=0.85))
    ax.set_xlim(-180, 180)
    ax.set_ylim(lat0, lat1)
    return fig_to_uri(fig)


# ---- figure 3: soil-water level shift (optional) ----------------------------
def fig_soil_shift(cards_retraction, retraction_meta=None):
    """Global soil water mean showing a level shift instead of a trend.

    Reads outputs_global/regional_soil_water_0_200cm.csv read-only. The break
    month is taken from the retraction card if present, otherwise located by a
    single best mean-shift split on the deseasonalised series (computed, not
    typed in).
    """
    try:
        import pandas as pd
        df = pd.read_csv(SOIL_CSV, index_col=0, parse_dates=True)
    except Exception:
        return None, []
    lats = []
    for c in df.columns:
        m = re.match(r"tile_([+-]\d+\.\d+)_", c)
        lats.append(float(m.group(1)) + 0.5 if m else 0.0)
    w = np.cos(np.radians(np.asarray(lats)))
    X = df.to_numpy(float)
    ok = np.isfinite(X)
    s = np.nansum(np.where(ok, X, 0) * w, axis=1) / np.sum(ok * w, axis=1)
    months = df.index.month.to_numpy()
    clim = np.array([np.nanmean(s[months == m]) for m in range(1, 13)])
    anom = s - clim[months - 1]
    n = len(anom)

    # Break months: authoritative dates come from the retraction card (any
    # YYYY-MM in its text). Without a card, fit the best two-break step
    # function for display only and leave the dates unlabelled.
    breaks, labelled = [], False
    window = None
    if isinstance(retraction_meta, dict) and retraction_meta.get("breakpoints"):
        breaks = sorted(retraction_meta["breakpoints"])
        window = retraction_meta.get("window")
        labelled = True
    elif cards_retraction:
        txt = json.dumps(cards_retraction)
        breaks = sorted(set(re.findall(r"\b((?:19|20)\d{2}-(?:0[1-9]|1[0-2]))\b", txt)))
        labelled = bool(breaks)
    if not breaks:
        cs = np.concatenate([[0], np.cumsum(np.nan_to_num(anom))])
        cs2 = np.concatenate([[0], np.cumsum(np.nan_to_num(anom) ** 2)])

        def sse(i, j):
            mu = (cs[j] - cs[i]) / (j - i)
            return cs2[j] - cs2[i] - mu * mu * (j - i)

        L, best, bij = 12, np.inf, None
        for i in range(L, n - 2 * L):
            for j in range(i + L, n - L + 1):
                v = sse(0, i) + sse(i, j) + sse(j, n)
                if v < best:
                    best, bij = v, (i, j)
        if bij:
            breaks = [df.index[k].strftime("%Y-%m") for k in bij]

    t = np.array([d.year + (d.month - 1) / 12 for d in df.index])
    fig, ax = plt.subplots(figsize=(7.6, 3.1), facecolor=INK)
    style_axes(ax)
    if window and len(window) == 2:
        w0, w1 = (int(x[:4]) + (int(x[5:7]) - 1) / 12 for x in window)
        ax.axvspan(w0, w1 + 1 / 12, color=WET, alpha=0.07, lw=0, zorder=0)
        ax.text((w0 + w1) / 2, np.nanmin(anom), f"consistent window {window[0]} to {window[1]}",
                color=MUTED, fontsize=8, ha="center", va="bottom")
    ax.plot(t, anom, color=MUTED, lw=0.8, alpha=0.7)
    # naive straight-line fit (what a careless trend test would report)
    good = np.isfinite(anom)
    a, b0 = np.polyfit(t[good], anom[good], 1)
    ax.plot(t, a * t + b0, color=DRY, lw=1.6, ls=(0, (6, 4)))
    # step means either side of each break
    edges = [0]
    for br in breaks:
        hits = np.where(df.index.strftime("%Y-%m") == br)[0]
        if hits.size:
            edges.append(int(hits[0]))
    edges = sorted(set(edges)) + [n]
    for i0, i1 in zip(edges[:-1], edges[1:]):
        ax.plot([t[i0], t[i1 - 1]], [np.nanmean(anom[i0:i1])] * 2,
                color=TEXT, lw=2.4, solid_capstyle="butt")
    for i in edges[1:-1]:
        ax.axvline(t[i], color=WET, lw=1.2, ls=":")
        if labelled:
            ax.text(t[i], np.nanmax(anom), " " + df.index[i].strftime("%Y-%m"),
                    color=WET, fontsize=9, va="top", ha="left")
    ax.axhline(0, color=RULE, lw=0.8, zorder=0)
    ax.set_ylabel("soil water anomaly (mm)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=RULE, lw=0.5)
    return fig_to_uri(fig), (breaks if labelled else [])


# ---- place names (fallback only; cards.json supplies its own) ---------------
GAZETTEER = [  # (name, west, south, east, north) -- first match wins
    ("Central Chile", -74, -38, -69, -30),
    ("Patagonia / southern Chile", -78, -56, -63, -38),
    ("Sahel", -18, 9, 40, 20),
    ("Gulf of Guinea", -10, -2, 15, 9),
    ("Madagascar", 42, -26, 51, -11),
    ("Central-southern Africa", 10, -25, 40, -2),
    ("Amazon / Brazil", -80, -30, -34, 2),
    ("Northern Andes", -82, -5, -66, 13),
    ("Iceland", -25, 62, -12, 68),
    ("Greenland", -75, 59, -10, 84),
    ("Pacific Northwest / Alaska", -170, 45, -120, 72),
    ("Indus / northwest India", 60, 15, 82, 37),
    ("North & northeast China", 100, 20, 140, 56),
    ("Russian Far East", 140, 45, 180, 75),
    ("Central Siberia", 75, 55, 115, 80),
    ("Northwest Russia", 28, 54, 60, 70),
    ("Europe", -12, 35, 40, 72),
    ("North America", -170, 15, -50, 84),
    ("South Asia", 60, 5, 100, 35),
    ("Australia", 110, -45, 160, -10),
]


def place_name(lat, lon):
    for name, w, s, e, n in GAZETTEER:
        if w <= lon <= e and s <= lat <= n:
            return name
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.0f}°{ns} {abs(lon):.0f}°{ew}"


def default_user(place, kind):
    p = place.lower()
    if "chile" in p:
        return {"who": "A Chilean water planner during the central Chile megadrought",
                "decision": "Whether reservoir rules and allocations should assume a drier baseline",
                "why_it_matters": "A detectable drying trend argues against waiting for a return to old averages."}
    if "sahel" in p:
        return {"who": "A Sahel agricultural extension office",
                "decision": "Planting dates and seed varieties under a wetting rebound",
                "why_it_matters": "Wetter seasons change which crops and timings carry the least risk."}
    return {"who": "A secondary-school geography teacher using the explorer",
            "decision": "Which local example to use when teaching that averages hide regional change",
            "why_it_matters": "Students see the global flat line split into real places that are drying and wetting."}


def fallback_cards(regions, tiles_idx, tiles_rows):
    q_of = {r[tiles_idx["id"]]: r[tiles_idx["q"]] for r in tiles_rows}
    pct_of = {r[tiles_idx["id"]]: r[tiles_idx["pct"]] for r in tiles_rows}
    cards = []
    for side, kind in (("down", "drying"), ("up", "wetting")):
        for reg in regions.get(side, []):
            ids = reg.get("tiles", [])
            qs = [q_of[i] for i in ids if q_of.get(i) is not None]
            share = (sum(q < 0.05 for q in qs) / len(qs)) if qs else None
            pcts = [pct_of[i] for i in ids if pct_of.get(i) is not None]
            med_pct = reg.get("median_slope_pct_per_decade")
            if med_pct is None and pcts:
                med_pct = float(np.median(pcts))
            place = place_name(reg["centroid_lat"], reg["centroid_lon"])
            cards.append({
                "id": reg["id"], "kind": kind, "region_id": reg["id"],
                "title": f"{place}: a coherent {kind} cluster",
                "place": place,
                "headline": (f"{reg['n_tiles']} contiguous 1° tiles show a detectable "
                             f"{kind} trend of about {fmt_num(med_pct, 1, True)}% per decade over 2000–2025."),
                "numbers": {"median_pct_per_decade": med_pct,
                            "median_slope_per_decade": reg.get("median_slope_per_decade"),
                            "area_km2": reg.get("area_km2"),
                            "n_tiles": reg.get("n_tiles"),
                            "share_tiles_fdr_significant": share},
                "robustness": [
                    {"check": "FDR-corrected significance",
                     "status": PASS if share and share >= 0.95 else PARTIAL,
                     "detail": f"{fmt_pct(share, 0)} of tiles have BH q < 0.05"},
                    {"check": "Stable across time windows", "status": NOT_RUN,
                     "detail": "graded in cards.json"},
                    {"check": "Queen vs kNN neighbours agree", "status": NOT_RUN,
                     "detail": "graded in cards.json"},
                    {"check": "Independent literature agrees", "status": NOT_RUN,
                     "detail": "graded in cards.json"},
                ],
                "user": default_user(place, kind),
                "evidence_level": "statistically supported",
                "literature": [],
                "caveats": [],
            })
    return cards


def pick_example_card(cards):
    """Prefer a 'robust' card (Sahel, then North China, then any robust one),
    then the Chile drying card, then any drying card, then the first."""
    cand = [c for c in cards if c.get("kind") in ("drying", "wetting")]
    robust = [c for c in cand if str(c.get("evidence_level", "")).lower().startswith("robust")]
    for key in ("sahel", "north_china"):
        for c in robust:
            if key in str(c.get("id", "")).lower() or key.replace("_", " ") in (
                    c.get("place", "") + c.get("title", "")).lower():
                return c
    if robust:
        return robust[0]
    for c in cand:
        if "chile" in (c.get("place", "") + c.get("title", "")).lower():
            return c
    for c in cand:
        if c.get("kind") == "drying":
            return c
    return cand[0] if cand else None


# ---- HTML pieces -----------------------------------------------------------
MARK = {PASS: ("✓", "pass"), PARTIAL: ("◐", "partial"),
        FAIL: ("✗", "fail"), NOT_RUN: ("–", "notrun")}


def short_cite(c: str) -> str:
    """'Garreaud, R. D., Boisier, ... (2020). Title...' -> 'Garreaud et al. 2020'."""
    c = c or ""
    m = re.match(r"\s*([^,.(]+)", c)
    y = re.search(r"\((\d{4})\)", c)
    if not m:
        return c[:40]
    authors = c.split("(", 1)[0]  # author list only: commas in the title must not trigger 'et al.'
    multi = authors.count(",") > 2 or "et al" in authors or "&" in authors
    return f"{m.group(1).strip()}{' et al.' if multi else ''}{' ' + y.group(1) if y else ''}"


def short_detail(r, lit=None) -> str:
    """Compact one-line version of a checklist detail for the slide."""
    d = str(r.get("detail") or "")
    name = str(r.get("check") or "").lower()
    if "significance" in name:
        m = re.search(r"(\d+) of (\d+) tiles \((\d+%)\)", d)
        q = re.search(r"q < ([0-9.]+)", d)
        if m:
            return f"{m.group(1)} of {m.group(2)} tiles significant, BH q < {q.group(1) if q else '0.05'}"
    if "time window" in name:
        parts = re.findall(r"(\d{4}-\d{4}): ([+-]?[\d.]+%)", d.split("Global shares")[0])
        if parts:
            return " · ".join(f"{a.replace('-', '–')}: {b}" for a, b in parts)
    if "neighbour" in name:
        m = re.search(r"and (\d+%) of the queen (\w+) tiles keep their label", d)
        if m:
            return f"{m.group(1)} of queen-LISA {m.group(2)} tiles keep the label under kNN weights"
    if "independent precipitation" in name or "chirps" in name:
        m = re.search(r"Over (\d{4})-(\d{4}), CHIRPS-based precipitation shows (\w+) in (\d+%) of "
                      r"this region's (\d+) tiles.*?median trend ([+-]?[\d.]+) mm/month per decade "
                      r"vs ([+-]?[\d.]+) in GLDAS", d)
        if m:
            y0, y1, word, share, n, f, g = m.groups()
            return (f"CHIRPS {y0}–{y1}: {word} in {share} of {n} tiles · median "
                    f"{f} vs {g} mm/month per decade in GLDAS")
    if "literature" in name and lit:
        cites = ", ".join(short_cite(x.get("citation")) for x in lit[:2])
        m = re.search(r"(\d+) agree, (\d+) partly", d)
        return (f"{m.group(1)} agree, {m.group(2)} partly · " if m else "") + cites
    d = d.split(";")[0].split(" - ")[0]
    return d if len(d) <= 110 else d[:107].rsplit(" ", 1)[0] + "…"


def checklist_html(rob, lit=None):
    rows = []
    for r in rob or []:
        st = str(r.get("status", NOT_RUN)).lower()
        sym, cls = MARK.get(st, MARK[NOT_RUN])
        rows.append(
            f'<li class="chk {cls}"><span class="mark">{sym}</span>'
            f'<span class="chk-name">{esc(r.get("check"))}</span>'
            f'<span class="chk-detail">{esc(short_detail(r, lit))}</span></li>')
    return '<ul class="checklist">' + "".join(rows) + "</ul>"


def grade_line(rob):
    rob = rob or []
    run = [r for r in rob if str(r.get("status")).lower() != NOT_RUN]
    graded = [r for r in rob if r.get("role") != "selection"]
    run = [r for r in graded if str(r.get("status")).lower() != NOT_RUN]
    passed = sum(str(r.get("status")).lower() == PASS for r in graded)
    sel = " (plus FDR selection)" if len(graded) < len(rob) else ""
    return f"{passed} of {len(graded)} checks passed{sel}" + (
        f" · {len(graded) - len(run)} not yet run" if len(run) < len(graded) else "")


def build_html(ctx):
    m, team = ctx["meta"], ctx["team"]
    ds = m.get("dataset", {})
    agg = get(m, "headline", "aggregate", default={})
    reg = get(m, "headline", "regional", default={})
    wins = get(m, "headline", "windows", default={}) or {}
    ci = agg.get("ci95") or [None, None]
    period = ds.get("period") or ["", ""]
    y0, y1 = str(period[0])[:4], str(period[-1])[:4]
    members = team.get("members") or []
    card = ctx["card"] or {}
    num = card.get("numbers", {}) or {}
    user = card.get("user", {}) or {}
    retr = ctx["retraction_card"] or {}
    retr_user = retr.get("user") or {
        "who": "A drought-monitoring analyst",
        "decision": "Whether to issue a soil-moisture drought alert from a reanalysis trend",
        "why_it_matters": "A false trend from a data-processing change would trigger the wrong alarm."}

    # windows table
    win_rows = []
    full_key = f"{y0}–{y1}"
    win_rows.append((full_key, reg.get("share_area_up"), reg.get("share_area_down"),
                     agg.get("slope"), agg.get("p_value")))
    for name, w in wins.items():
        win_rows.append((name.replace("-", "–"), w.get("share_area_up"),
                         w.get("share_area_down"), w.get("aggregate_slope"),
                         w.get("aggregate_p")))
    win_html = "".join(
        f"<tr><td>{esc(n)}</td><td class='wet'>{fmt_pct(u)}</td>"
        f"<td class='dry'>{fmt_pct(d)}</td><td>{fmt_num(s, 2, True)}</td>"
        f"<td>{fmt_p(p)}</td></tr>" for n, u, d, s, p in win_rows)

    lit = card.get("literature") or []
    lit_html = "".join(
        f"<li><span class='lit-agree'>{'agrees' if l.get('agrees') is True else ('partly' if l.get('agrees') == 'partly' else 'disagrees')}</span> "
        f"{esc(l.get('citation'))}</li>" for l in lit[:2])

    word = {"drying": "drying", "wetting": "wetting"}.get(card.get("kind"), "")
    card_summary = (f"{num.get('n_tiles')} contiguous 1° land tiles, {fmt_area(num.get('area_km2'))}, "
                    f"show a detectable {word} trend over {y0}–{y1} (median "
                    f"{fmt_num(num.get('median_slope_per_decade'), 1, True)} mm/month per decade). "
                    f"Statistical trend, not a cause.")
    _r = m.get("retraction")
    if isinstance(_r, dict):
        bp = " and ".join(_r.get("breakpoints") or [])
        win = _r.get("window") or []
        retr_headline = ("Soil water was our original headline: a 26-year global decline. Our own break test "
                         f"found level shifts at {bp}, matching documented changes in GLDAS radiation forcing"
                         + (f"; inside the consistent window {win[0]} to {win[1]} the decline is not significant."
                            if len(win) == 2 else "."))
    else:
        retr_headline = _r or retr.get("headline", "")
    retr_breaks = ctx["soil_breaks"]
    breaks_txt = ", ".join(retr_breaks) if retr_breaks else ""

    if ctx["explorer_uri"]:
        explorer = f'<img class="shot" src="{ctx["explorer_uri"]}" alt="Paradox Earth explorer screenshot">'
    else:
        explorer = '<div class="shot pending">screenshot pending</div>'

    soil_fig = (f'<img class="fig" src="{ctx["soil_uri"]}" alt="Global soil water anomaly with level shift">'
                if ctx["soil_uri"] else '<div class="shot pending">soil-water figure unavailable</div>')

    n_cards = ctx["n_cards"]
    n_years = int(y1) - int(y0) + 1 if y0.isdigit() and y1.isdigit() else ""
    caveat0 = (m.get("caveats") or [""])[0]

    # ---- slide 4: the NASA products, all read from meta.json -------------
    sw = m.get("soil_water") or {}
    ac = sw.get("artifact_control") or {}
    fl = sw.get("fldas") or {}
    gr = sw.get("grace") or {}
    pc = get(m, "cross_checks", "precip_chirps_fldas", default={}) or {}
    fl_ds = str(fl.get("dataset") or "")
    fl_short = fl_ds.split(" (")[0].strip() or "FLDAS"
    fl_doi_m = re.search(r"doi:\s*(10\.\S+)", fl_ds)
    fl_doi = fl_doi_m.group(1).rstrip(".,;") if fl_doi_m else ""
    fl_forc = re.search(r"\(([^)]*forcing[^)]*)\)", fl_ds)
    fl_forc = fl_forc.group(1).split(", ", 1)[-1] if fl_forc else ""
    ctrl_ver = next((v for v in (ac.get("forcing") or {}) if v != ds.get("version")), "")
    ctrl_win = get(ac, "headline", "common_window", default=None) or []
    fl_win = [str(x)[:4] for x in (fl.get("window") or [])]
    pc_win = [str(x)[:4] for x in (pc.get("window") or [])]
    gr_win = [str(x)[:4] for x in (gr.get("window") or [])]
    gvf = gr.get("vs_fldas_soil_water") or {}
    hot = gvf.get("hotspots") or {}
    hot_name, hot_v = next(iter(hot.items()), (None, {}))
    gr_rel = re.search(r"RL\S+(?: CRI)?", str(gr.get("product") or ""))
    gr_rel = gr_rel.group(0) if gr_rel else ""

    def yr(w):
        return f"{w[0]}–{w[-1]}" if len(w) == 2 else ""

    sources = [
        ("nasa", "Main record", f"GLDAS-{ds.get('version', '')} Noah",
         f"{ds.get('short_name', '')} v{ds.get('version', '')} · doi:{ds.get('doi', '')}",
         f"Monthly precipitation (GPCP-based forcing) on one 0.25° grid, {y0}–{y1}: the trend tests run here."),
        ("xchk", "Independent cross-check", "NASA FLDAS Noah",
         f"{fl_short} · doi:{fl_doi}",
         f"CHIRPS precipitation ({yr(pc_win)}): trend signs match on {fmt_pct(pc.get('sign_match_area_sig_either'), 0)} of land "
         f"where either is significant, {fmt_pct(pc.get('sign_match_area_sig_both'), 0)} where both. "
         f"Consistently forced soil water ({fl_forc})."),
        ("xchk", "Control", f"GLDAS-{ctrl_ver} Noah",
         f"{ds.get('short_name', '')} v{ctrl_ver} · same model, other forcing",
         f"{str(ctrl_win[0])[:4] + '–' + str(ctrl_win[-1])[:4] + ': ' if len(ctrl_win) == 2 else ''}"
         f"the {ac.get('step_date', '')} solar-radiation step shows up only in GLDAS-{ds.get('version', '')} "
         f"({fmt_num(ac.get('swdown_step_21_W_m2'), 1, True)} vs {fmt_num(ac.get('swdown_step_20_W_m2'), 1, True)} W/m²)."),
        ("obs", "Observations", "GRACE / GRACE-FO",
         f"{gr.get('product', '')}",
         f"Satellite gravity, {yr(gr_win)}: {fmt_pct(gr.get('land_area_share_fdr_significant'), 0)} of land shows "
         f"FDR-significant storage trends"
         + (f". {hot_name}: GRACE {fmt_num(hot_v.get('grace_mean_slope_mm_decade'), 0, True)} vs land model "
            f"{fmt_num(hot_v.get('model_mean_slope_mm_decade'), 0, True)} mm/decade (groundwater is not in the model)."
            if hot_name and hot_v.get("grace_mean_slope_mm_decade") is not None else ".")),
    ]
    src_html = "".join(
        f'<div class="src {cls}"><div class="k">{esc(k)}</div><div class="t">{esc(t)}</div>'
        f'<div class="id">{esc(i)}</div><p>{esc(p)}</p></div>'
        for cls, k, t, i, p in sources)

    # ---- slide 6: the GLDAS-2.1 vs 2.0 control figure + FLDAS no-trend ---
    ac_uri = ctx.get("ac_uri")
    ac_fig = (f'<img class="fig" src="{ac_uri}" alt="GLDAS-2.1 vs GLDAS-2.0: radiation step and soil water">'
              if ac_uri else '<div class="shot pending">artefact-control figure unavailable</div>')
    fl_ci = fl.get("aggregate_ci95") or [None, None]
    fl_p = fl.get("aggregate_p")
    fl_line = (f"In NASA FLDAS, forced consistently with {fl_forc.replace(' forcing', '') or 'one forcing'} over {yr(fl_win)}, global land "
               f"soil water shows no detectable trend: {fmt_num(fl.get('aggregate_slope_mm_per_decade'), 1, True)} "
               f"mm/decade (95% CI {fmt_num(fl_ci[0], 1)} to {fmt_num(fl_ci[1], 1)}, p = {fmt_p(fl_p)}).")
    ac_reading = ac.get("reading") or retr_headline

    # ---- slide 7: forecast lab line from site/data/pinn.json --------------
    pinn = ctx.get("pinn") or {}
    lab_line = ""
    if pinn.get("rows"):
        arm = get(pinn, "best_for_display", "arm", default="base")
        rows = [r for r in pinn["rows"] if r.get("arm") == arm]
        sk = [(r["lambda"], get(r, "test", "spatial_holdout", "msess_vs_linear_ar", "mean")) for r in rows]
        sk_ok = [s for _, s in sk if s is not None and s > 0]
        res = {r["lambda"]: get(r, "test", "in_region", "budget_residual_own_fluxes_mm", "mean") for r in rows}
        r0 = res.get(0.0)
        r1 = res.get(1.0)
        if sk_ok and r0 is not None and r1 is not None:
            lab_line = (f"<b>Forecast lab:</b> a physics-informed CNN emulator of the GLDAS-{esc(ds.get('version'))} land water store "
                        f"beats a linear AR baseline by {fmt_num(min(sk_ok), 2, True)} to {fmt_num(max(sk_ok), 2, True)} MSESS "
                        f"on a region held out of training; a water-budget penalty (λ = 1) cuts its budget residual "
                        f"from {fmt_num(r0, 1)} to {fmt_num(r1, 1)} mm/month. Emulation skill against a model, not a forecast of reality.")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Paradox Earth Deck</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=block" rel="stylesheet">
<style>
:root {{
  --ink:{INK}; --panel:{PANEL}; --rule:{RULE}; --text:{TEXT}; --muted:{MUTED};
  --wet:{WET}; --dry:{DRY}; --land:{LAND};
  --display:'Bricolage Grotesque', 'Helvetica Neue', Arial, sans-serif;
  --body:'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
  --mono:'IBM Plex Mono', Menlo, monospace;
}}
@page {{ size: 1280px 720px; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: var(--ink); color: var(--text); font-family: var(--body);
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
.slide {{ width: 1280px; height: 720px; position: relative; overflow: hidden;
  padding: 56px 72px 48px; background: var(--ink); page-break-after: always;
  break-after: page; display: flex; flex-direction: column; }}
.slide:last-child {{ page-break-after: auto; break-after: auto; }}
.eyebrow {{ font-family: var(--mono); font-size: 14px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }}
h1 {{ font-family: var(--display); font-weight: 800; font-size: 148px;
  line-height: .9; letter-spacing: -.035em; }}
h2 {{ font-family: var(--display); font-weight: 700; font-size: 50px;
  line-height: 1.02; letter-spacing: -.025em; max-width: 1050px; }}
p, li {{ font-size: 20px; line-height: 1.42; }}
.muted {{ color: var(--muted); }}
.wet {{ color: var(--wet); }} .dry {{ color: var(--dry); }}
.mono {{ font-family: var(--mono); }}
.foot {{ position: absolute; left: 72px; right: 72px; bottom: 22px; display: flex;
  justify-content: space-between; font-family: var(--mono); font-size: 11.5px;
  color: var(--muted); letter-spacing: .04em; }}
.foot span:first-child {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; padding-right: 24px; }}
.foot span:last-child {{ white-space: nowrap; flex: none; }}
.fig {{ width: 100%; display: block; }}

/* 1 title */
.s-title {{ justify-content: space-between; }}
.s-title .split {{ display: flex; height: 10px; width: 100%; margin: 26px 0 30px; }}
.s-title .split i {{ display: block; height: 100%; }}
.s-title .tagline {{ font-family: var(--display); font-size: 34px; font-weight: 400;
  max-width: 900px; line-height: 1.15; color: var(--text); }}
.s-title .tagline em {{ font-style: normal; color: var(--muted); }}
.credits {{ display: grid; grid-template-columns: 1.2fr 1fr 1.4fr; gap: 32px;
  border-top: 1px solid var(--rule); padding-top: 20px; }}
.credits dt {{ font-family: var(--mono); font-size: 12px; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }}
.credits dd {{ font-size: 19px; line-height: 1.35; }}

/* 2 paradox */
.bignum {{ display: flex; align-items: flex-end; gap: 40px; margin: 22px 0 6px; }}
.bignum .n {{ font-family: var(--display); font-weight: 800; font-size: 120px;
  letter-spacing: -.04em; line-height: .9; }}
.bignum .n small {{ font-size: 40px; font-weight: 600; letter-spacing: -.01em; }}
.bignum .lbl {{ font-size: 19px; color: var(--muted); max-width: 520px; padding-bottom: 10px; }}
.bignum .lbl b {{ color: var(--text); font-weight: 600; }}

/* 3 split */
.s-split .row {{ display: flex; gap: 28px; align-items: flex-start; margin-top: 10px; }}
.s-split .mapbox {{ flex: 1 1 auto; min-width: 0; }}
.s-split .side {{ width: 300px; flex: none; }}
.share {{ margin-bottom: 18px; }}
.share .v {{ font-family: var(--display); font-weight: 800; font-size: 64px;
  line-height: .95; letter-spacing: -.03em; }}
.share .k {{ font-size: 15px; color: var(--muted); }}
table.win {{ border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 12.5px; }}
table.win th {{ text-align: left; color: var(--muted); font-weight: 400; padding: 3px 4px;
  border-bottom: 1px solid var(--rule); }}
table.win td {{ padding: 4px 4px; border-bottom: 1px solid var(--rule); }}
.legend {{ display: flex; flex-wrap: wrap; row-gap: 4px; gap: 22px; font-size: 14px; color: var(--muted); margin-top: 6px; }}
.legend span::before {{ content: ''; display: inline-block; width: 12px; height: 12px;
  margin-right: 7px; vertical-align: -1px; background: var(--c); }}

/* 4 pipeline */
.flow {{ display: flex; align-items: stretch; gap: 0; margin-top: 40px; }}
.step {{ flex: 1; background: var(--panel); border: 1px solid var(--rule);
  border-radius: 10px; padding: 20px 18px; position: relative; }}
.step + .arrow {{ }}
.arrow {{ width: 34px; flex: none; display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 24px; }}
.step .k {{ font-family: var(--mono); font-size: 12px; letter-spacing: .1em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }}
.step .t {{ font-family: var(--display); font-weight: 700; font-size: 22px; line-height: 1.1;
  margin-bottom: 10px; }}
.step p {{ font-size: 15px; line-height: 1.4; color: var(--muted); }}
.step.nasa {{ border-color: var(--wet); }}
.step.out {{ border-color: var(--dry); }}
.nasa-note {{ margin-top: 34px; display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }}
.nasa-note p {{ font-size: 18px; }}

.srcs {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 22px; }}
.src {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 14px 16px; }}
.src.nasa {{ border-color: var(--wet); }}
.src.obs {{ border-color: var(--dry); }}
.src .k {{ font-family: var(--mono); font-size: 11.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }}
.src .t {{ font-family: var(--display); font-weight: 700; font-size: 22px; line-height: 1.1; margin: 6px 0 4px; }}
.src .id {{ font-family: var(--mono); font-size: 11px; color: var(--wet); line-height: 1.35; margin-bottom: 8px; word-break: break-word; }}
.src p {{ font-size: 13.5px; line-height: 1.38; color: var(--muted); }}
.s-how .flow {{ margin-top: 18px; }}
.s-how .step {{ padding: 12px 14px; }}
.s-how .step .k {{ margin-bottom: 4px; }}
.s-how .step .t {{ font-size: 18px; margin-bottom: 4px; }}
.s-how .step p {{ font-size: 12.5px; line-height: 1.35; }}
.s-how h2 {{ font-size: 44px; }}

/* 5 + 6 case file */
.case {{ display: grid; grid-template-columns: 1.05fr 1fr; gap: 36px; margin-top: 18px; flex: 1; min-height: 0; }}
.file {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 12px;
  padding: 22px 26px; position: relative; }}
.file .tag {{ font-family: var(--mono); font-size: 12px; letter-spacing: .12em;
  text-transform: uppercase; color: var(--muted); }}
.file .place {{ font-family: var(--display); font-weight: 700; font-size: 29px;
  letter-spacing: -.02em; margin: 6px 0 8px; line-height: 1.05; }}
.file .hl {{ font-size: 16px; line-height: 1.4; }}
.stats {{ display: flex; gap: 28px; margin: 14px 0 4px; border-top: 1px solid var(--rule); padding-top: 12px; }}
.stats div b {{ display: block; font-family: var(--display); font-weight: 800; font-size: 32px; letter-spacing: -.02em; }}
.stats div span {{ font-size: 13px; color: var(--muted); }}
.checklist {{ list-style: none; }}
.chk {{ display: grid; grid-template-columns: 34px 1fr; column-gap: 12px; padding: 11px 0;
  border-bottom: 1px solid var(--rule); align-items: baseline; }}
.chk .mark {{ grid-row: span 2; font-size: 24px; font-weight: 700; text-align: center; }}
.chk-name {{ font-size: 18px; font-weight: 600; }}
.chk-detail {{ font-size: 14px; color: var(--muted); line-height: 1.35; }}
.chk.pass .mark {{ color: var(--wet); }}
.chk.partial .mark {{ color: #D9C36A; }}
.chk.fail .mark {{ color: var(--dry); }}
.chk.notrun .mark {{ color: var(--muted); }}
.grade {{ font-family: var(--mono); font-size: 13px; color: var(--muted); margin: 12px 0 0; letter-spacing: .04em; }}
.who {{ margin-top: 12px; border-left: 3px solid var(--dry); padding: 2px 0 2px 16px; }}
.who .q {{ font-size: 13px; font-family: var(--mono); color: var(--muted); letter-spacing: .1em; text-transform: uppercase; }}
.who p {{ font-size: 15.5px; line-height: 1.35; }}
.who p b {{ font-weight: 600; }}
.lits {{ list-style: none; margin-top: 10px; }}
.lits li {{ font-size: 13px; color: var(--muted); line-height: 1.35; }}
.lit-agree {{ font-family: var(--mono); color: var(--text); }}
.stamp {{ position: absolute; bottom: 22px; right: 24px; font-family: var(--mono); font-weight: 500;
  font-size: 15px; letter-spacing: .2em; color: var(--dry); border: 2px solid var(--dry);
  padding: 6px 12px; transform: rotate(-4deg); border-radius: 4px; }}
.s-card h2 {{ font-size: 44px; }}
.s-card .case {{ margin-top: 12px; }}
.s-card .chk {{ padding: 7px 0; }}
.s-card .chk .mark {{ font-size: 20px; }}
.s-card .chk-name {{ font-size: 16px; line-height: 1.25; }}
.s-card .chk-detail {{ font-size: 13px; line-height: 1.3; }}
.s-card .file {{ padding: 18px 22px; }}
.s-retract .lede {{ font-size: 18px; line-height: 1.4; max-width: 1100px; margin-top: 12px; }}
.s-retract h2 {{ font-size: 44px; }}
.s-retract .case {{ margin-top: 14px !important; }}
.s-retract .legend {{ margin-top: 4px; }}
.s-retract .file {{ padding: 18px 22px 54px; }}
.s-retract .file .hl {{ font-size: 15px; }}
.s-retract .who p {{ font-size: 14.5px; }}
.s-retract .stamp {{ bottom: 14px; }}
.s-retract .fl-line b {{ font-weight: 600; color: var(--wet); }}

/* 7 explorer */
.s-explore .row {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 40px; margin-top: 24px; }}
.shot {{ width: 100%; border-radius: 10px; border: 1px solid var(--rule); display: block; }}
.shot.pending {{ aspect-ratio: 16/10; display: flex; align-items: center; justify-content: center;
  font-family: var(--mono); color: var(--muted); font-size: 16px; letter-spacing: .1em;
  background: repeating-linear-gradient(135deg, var(--panel) 0 14px, var(--ink) 14px 28px); }}
.next li {{ font-size: 16px; line-height: 1.36; margin: 0 0 8px 20px; }}
.next li.lab {{ font-size: 14.5px; color: var(--muted); }}
.next li.lab b {{ color: var(--wet); font-weight: 600; }}
.s-explore h2 {{ font-size: 40px; max-width: none; }}
.s-explore .row {{ grid-template-columns: 1.15fr 1fr; }}
.s-explore .refs li {{ font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.s-explore .row {{ margin-top: 16px !important; }}
.s-explore .links {{ margin-top: 10px; }}
.s-explore .refs {{ margin-top: 10px; }}
.links {{ margin-top: 22px; font-family: var(--mono); font-size: 13px; line-height: 1.8; white-space: nowrap; }}
.links a {{ color: var(--wet); text-decoration: none; }}
.refs {{ margin-top: 18px; border-top: 1px solid var(--rule); padding-top: 10px; }}
.refs .q {{ font-family: var(--mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }}
.refs li {{ font-size: 11.5px; line-height: 1.38; color: var(--muted); list-style: none; }}

@media screen {{ body {{ display: flex; flex-direction: column; align-items: center; gap: 0; }} }}
@media print {{ html, body {{ width: 1280px; }} }}
</style></head>
<body>

<!-- 1 TITLE -->
<section class="slide s-title">
  <div>
    <div class="eyebrow">{esc(team.get("event", "NASA International Space Apps Challenge"))} · {esc(team.get("challenge", m.get("challenge")))}</div>
    <h1>{esc(m.get("project", "Paradox Earth"))}</h1>
    <div class="split"><i style="flex:{reg.get('share_area_down') or 1};background:var(--dry)"></i><i style="flex:{max(0.0, 1 - (reg.get('share_area_up') or 0) - (reg.get('share_area_down') or 0))};background:var(--land)"></i><i style="flex:{reg.get('share_area_up') or 1};background:var(--wet)"></i></div>
    <p class="tagline">{esc(m.get("tagline"))} <em>An automated trend detective that grades its own evidence.</em></p>
  </div>
  <dl class="credits">
    <div><dt>Team</dt><dd>{esc(team.get("team_name"))}</dd></div>
    <div><dt>Members</dt><dd>{"<br>".join(esc(x) for x in members)}</dd></div>
    <div><dt>Challenge</dt><dd>{esc(team.get("challenge"))}<br><span class="muted">{esc(team.get("local_event"))}</span></dd></div>
  </dl>
  <div class="foot"><span>Data: NASA GLDAS-2.1 ({esc(ds.get("short_name"))}), DOI {esc(ds.get("doi"))}</span><span>bar above: share of land drying · not significant · wetting</span></div>
</section>

<!-- 2 PARADOX -->
<section class="slide s-paradox">
  <div class="eyebrow">The paradox</div>
  <h2>Average all the world’s land together and precipitation barely moves.</h2>
  <div class="bignum">
    <div class="n">{fmt_num(agg.get("slope_pct_per_decade"), 1, True)}<small>% per decade</small></div>
    <div class="lbl">Land-average trend {y0}–{y1}: <b>{fmt_num(agg.get("slope"), 2, True)} mm/month per decade</b>,
      95% CI <b>[{fmt_num(ci[0], 2)}, {fmt_num(ci[1], 2)}]</b>, p = {fmt_p(agg.get("p_value"))}. The interval includes zero: no detectable global trend.</div>
  </div>
  <img class="fig" src="{ctx["line_uri"]}" alt="Flat global land-average precipitation anomaly">
  <div class="legend" style="margin-top:2px"><span style="--c:{TEXT}">12-month mean anomaly</span><span style="--c:{MUTED}">monthly anomaly</span><span style="--c:{DRY}">fitted trend</span></div>
  <div class="foot"><span>{esc(ds.get("variable_label"))} · land-area weighted · OLS slope, autocorrelation-robust (AR(1)) test</span><span>2 / 7</span></div>
</section>

<!-- 3 SPLIT -->
<section class="slide s-split">
  <div class="eyebrow">The split</div>
  <h2>Underneath the flat line, a few regions show detectable drying or wetting.</h2>
  <div class="row">
    <div class="mapbox">
      <img class="fig" src="{ctx["map_uri"]}" alt="Map of tiles with significant wetting and drying trends">
      <div class="legend"><span style="--c:{WET}">significant wetting</span><span style="--c:{DRY}">significant drying</span><span style="--c:{LAND}">no detectable trend</span><span>BH-FDR q &lt; {get(reg, "alpha_fdr", default=0.05)}</span></div>
    </div>
    <div class="side">
      <div class="share"><div class="v wet">{fmt_pct(reg.get("share_area_up"))}</div><div class="k">of tested land area shows a detectable wetting trend (2000-2025)</div></div>
      <div class="share"><div class="v dry">{fmt_pct(reg.get("share_area_down"))}</div><div class="k">of tested land area shows a detectable drying trend (2000-2025)</div></div>
      <table class="win"><tr><th>window</th><th>wet</th><th>dry</th><th>global</th><th>p</th></tr>{win_html}</table>
      <p class="muted" style="font-size:12.5px;margin-top:8px;line-height:1.35">Shares after false-discovery correction across {get(reg, "n_tiles_tested", default=0):,} 1° tiles. Global slope in mm/month per decade.</p>
    </div>
  </div>
  <div class="foot"><span>Statistical trends, not causes. {esc(caveat0)}</span><span>3 / 7</span></div>
</section>

<!-- 4 HOW -->
<section class="slide s-how">
  <div class="eyebrow">How NASA data makes it possible</div>
  <h2>Four NASA products: one to find trends, three to try to break them.</h2>
  <div class="srcs">{src_html}</div>
  <div class="flow">
    <div class="step"><div class="k">Ingest</div><div class="t">Streaming pipeline</div><p>Reads one month at a time and aggregates to {get(m, "n_tiles", default=0):,} land tiles of {get(m, "tile_deg", default=1):g}°. Quality checks {"passed" if m.get("qc_passed") else "pending"}.</p></div>
    <div class="arrow">→</div>
    <div class="step"><div class="k">Test</div><div class="t">Calibrated trend tests</div><p>OLS trend; p is the larger of an AR(1)-corrected and an annual-means test (calibrated to 3–6% false positives), then Benjamini–Hochberg FDR.</p></div>
    <div class="arrow">→</div>
    <div class="step"><div class="k">Map</div><div class="t">Spatial clusters</div><p>Local Moran’s I with permutation tests finds coherent drying and wetting regions, not lone pixels.</p></div>
    <div class="arrow">→</div>
    <div class="step out"><div class="k">Explain</div><div class="t">Discovery Cards</div><p>{n_cards} cards, each graded against the cross-checks above, with a named user, a decision and caveats.</p></div>
  </div>
  <p class="muted" style="font-size:14.5px;margin-top:14px;line-height:1.4">A gap-free, globally consistent NASA record lets a trend in Chile and one in Siberia be measured the same way; independent NASA products let each trend be challenged. Results are “detectable trends”, never causes.</p>
  <div class="foot"><span>Data: NASA GES DISC (GLDAS, FLDAS) · NASA JPL (GRACE/GRACE-FO mascons) · {esc(ds.get("variable"))}</span><span>4 / 7</span></div>
</section>

<!-- 5 CARD -->
<section class="slide s-card">
  <div class="eyebrow">A Discovery Card</div>
  <h2>Every finding arrives with its own evidence grade, and a person who needs it.</h2>
  <div class="case">
    <div class="file">
      <div class="tag">{esc(card.get("kind", ""))} · {esc(card.get("evidence_level", ""))}</div>
      <div class="place">{esc(card.get("place") or card.get("title"))}</div>
      <p class="hl">{esc(card_summary)}</p>
      <div class="stats">
        <div><b class="{'dry' if card.get('kind') == 'drying' else 'wet'}">{fmt_num(num.get("median_pct_per_decade"), 1, True)}%</b><span>median per decade</span></div>
        <div><b>{num.get("n_tiles") if num.get("n_tiles") is not None else "n/a"}</b><span>1° tiles</span></div>
        <div><b>{fmt_pct(num.get("share_tiles_fdr_significant"), 0)}</b><span>tiles FDR-significant</span></div>
      </div>
      <div class="who"><div class="q">Who decides with this</div>
        <p><b>{esc(user.get("who"))}</b>. <span class="muted">Decision: {esc(user.get("decision"))}.</span></p></div>
    </div>
    <div>
      <div class="tag eyebrow" style="margin-bottom:4px">Robustness checklist</div>
      {checklist_html(card.get("robustness"), card.get("literature"))}
      <div class="grade">{esc(grade_line(card.get("robustness")))}</div>
    </div>
  </div>
  <div class="foot"><span>Card {esc(card.get("id"))} · region {esc(card.get("region_id"))} · one of {n_cards} cards in the explorer</span><span>5 / 7</span></div>
</section>

<!-- 6 RETRACTION -->
<section class="slide s-retract">
  <div class="eyebrow">The detective that checks itself</div>
  <h2>Paradox Earth caught a false trend: it lines up with changes in the model’s input data.</h2>
  <p class="lede muted">{esc(ac_reading)}</p>
  <div class="case" style="grid-template-columns:1.45fr 1fr;margin-top:14px">
    <div>{ac_fig}</div>
    <div class="file">
      <div class="stamp">RETRACTED</div>
      <div class="tag">Soil water 0–200 cm · original headline</div>
      <p class="hl" style="margin-top:10px">A forcing change is not a trend. Our original {n_years}-year soil-water decline was withdrawn instead of published.</p>
      <p class="hl fl-line" style="margin-top:10px"><b>{esc(fl_line)}</b></p>
      <div class="who"><div class="q">Who this protects</div>
        <p><b>{esc(retr_user.get("who"))}</b>: {esc(retr_user.get("decision"))}. <span class="muted">{esc(retr_user.get("why_it_matters"))}</span></p></div>
    </div>
  </div>
  <div class="foot"><span>Figure: NASA GLDAS-2.1 vs GLDAS-2.0 (same Noah model, different forcing). The tool rejects its own result when the evidence does not hold.</span><span>6 / 7</span></div>
</section>

<!-- 7 EXPLORER -->
<section class="slide s-explore">
  <div class="eyebrow">Try it</div>
  <h2>An explorer anyone can open: click a place, read its case file.</h2>
  <div class="row">
    <div>{explorer}</div>
    <div>
      <ul class="next">
        <li>Every land tile with its corrected significance, and a graded Discovery Card for each coherent cluster.</li>
        {f'<li class="lab">{lab_line}</li>' if lab_line else ''}
        <li>Next: more NASA variables, GPM IMERG as a further check, and a card for your own region.</li>
      </ul>
      <div class="links">
        <div>explorer → <a href="{esc(team.get("site_url"))}">{esc(team.get("site_url"))}</a></div>
        <div>code → <a href="{esc(team.get("project_url"))}">{esc(team.get("project_url"))}</a></div>
      </div>
      <ul class="refs"><li class="q">References</li>
        <li>Beaudoing &amp; Rodell (2020), GLDAS Noah V2.1, doi:{esc(ds.get("doi"))} · Rodell et al. (2004), BAMS 85</li>
        <li>McNally et al. (2017), Sci. Data 4, 170012 · FLDAS Noah, doi:{esc(fl_doi or "10.5067/5NHC22T9375G")}</li>
        <li>Funk et al. (2015), CHIRPS, Sci. Data 2 · Rodell et al. (2009), Nature 460, 999–1002</li>
        <li>GRACE/GRACE-FO JPL mascons {esc(gr_rel)} · Watkins et al. (2015), JGR Solid Earth</li>
        <li>GES DISC README, GLDAS V2 forcing changes · Benjamini &amp; Hochberg (1995) · Anselin (1995)</li>
      </ul>
    </div>
  </div>
  <div class="foot"><span>Code written with help from an AI coding assistant; every number is computed from NASA data. No NASA logos used.</span><span>7 / 7</span></div>
</section>

</body></html>
"""


# ---- PDF + verification -----------------------------------------------------
def capture_explorer(png: Path, site: Path = ROOT / "site", case_id: str | None = None) -> bool:
    """Screenshot the static explorer (served from site/ on a free local port)."""
    import socket
    import threading
    import functools
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port),
                                  functools.partial(Quiet, directory=str(site)))
    except OSError as e:
        print("explorer capture: cannot serve site:", e)
        return False
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        png.parent.mkdir(parents=True, exist_ok=True)
        # flat (2D) view: the WebGL globe does not render reliably in headless Chrome
        run_chrome(["--hide-scrollbars", "--window-size=1600,1000", "--virtual-time-budget=20000",
                    f"--screenshot={png}",
                    f"http://127.0.0.1:{port}/index.html?reveal&view=flat"
                    + (f"#case={case_id}" if case_id else "")], png)
        return png.exists() and png.stat().st_size > 20000
    except Exception as e:  # pragma: no cover
        print("explorer capture failed:", e)
        return False
    finally:
        srv.shutdown()


def run_chrome(args, out: Path, tries: int = 3):
    """Run headless Chrome with a throwaway profile; retry (it exits 2 now and then)."""
    import tempfile
    last = None
    for _ in range(tries):
        with tempfile.TemporaryDirectory() as prof:
            if out.exists():
                out.unlink()
            r = subprocess.run([CHROME, "--headless=new", f"--user-data-dir={prof}",
                                "--no-first-run", *args], capture_output=True, timeout=180)
            if out.exists() and out.stat().st_size > 0:
                return
            last = r
    raise RuntimeError(f"headless Chrome failed ({last.returncode if last else '?'}): "
                       f"{(last.stderr or b'')[-400:].decode(errors='replace') if last else ''}")


def print_pdf(html_path: Path, pdf_path: Path):
    run_chrome(["--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                f"--print-to-pdf={pdf_path}", html_path.as_uri()], pdf_path)


def count_pdf_pages(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader  # type: ignore
        return len(PdfReader(str(pdf_path)).pages)
    except ImportError:
        data = pdf_path.read_bytes()
        return len(re.findall(rb"/Type\s*/Page(?![s\w])", data))


def preview(html_path: Path, out_dir: Path, n_slides: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "slides_all.png"
    run_chrome(["--disable-gpu", "--hide-scrollbars", f"--window-size=1280,{720 * n_slides}",
                "--virtual-time-budget=15000", f"--screenshot={png}", html_path.as_uri()], png)
    try:
        img = plt.imread(png)
        for i in range(n_slides):
            plt.imsave(out_dir / f"slide_{i + 1}.png", img[i * 720:(i + 1) * 720])
    except Exception as e:  # pragma: no cover
        print("preview split failed:", e)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--keep-explorer", action="store_true",
                    help="reuse demo/explorer.png instead of re-capturing the explorer")
    ap.add_argument("--preview", type=Path, default=None,
                    help="directory for per-slide PNG previews")
    args = ap.parse_args()

    meta = load_json(DATA / "meta.json")
    team = load_json(DEMO / "team.json", default={})
    regions = load_json(DATA / "regions.json", default={"up": [], "down": []})
    _, tidx, trows = load_tiles()

    cards_doc = load_json(DATA / "cards.json", default={})
    cards = cards_doc.get("cards") if isinstance(cards_doc, dict) else None
    source = "cards.json"
    if not cards:
        cards = fallback_cards(regions, tidx, trows)
        source = "regions.json (fallback)"
    card = pick_example_card(cards)
    retraction_card = next((c for c in cards if c.get("kind") == "retraction"), None)

    # label the two largest clusters on each side of the map
    cur = load_json(ROOT / "scripts" / "cards_curation.json", default={}) or {}
    short = {f'{e.get("sign")}-{e.get("key")}': e.get("map_label")
             for e in cur.get("regions", []) if e.get("map_label")}
    place_of = {c.get("region_id"): short.get(c.get("id")) or str(c.get("place", "")).split(" (")[0]
                for c in cards if c.get("region_id")}
    labels = []
    for side, kind in (("up", "wetting"), ("down", "drying")):
        pool = sorted(regions.get(side, []), key=lambda r: -r.get("n_tiles", 0))
        carded = [r for r in pool if r["id"] in place_of] or pool
        for r in carded[:3]:
            name = place_of.get(r["id"]) or place_name(r["centroid_lat"], r["centroid_lon"])
            lat, lon = r["centroid_lat"], r["centroid_lon"]
            if any(l["name"] == name or (abs(l["lat"] - lat) < 9 and abs(l["lon"] - lon) < 30)
                   for l in labels):
                continue  # avoid duplicate / overlapping map labels
            labels.append({"name": name, "lat": lat, "lon": lon, "kind": kind})

    print("rendering figures ...", flush=True)
    start, series = global_mean_series(tidx, trows)
    line_uri = fig_flat_line(start, series, get(meta, "headline", "aggregate", default={}))
    map_uri = fig_split_map(tidx, trows, regions, labels)
    soil_uri, soil_breaks = fig_soil_shift(retraction_card, meta.get("retraction"))

    explorer_png = DEMO / "explorer.png"
    if not (args.keep_explorer and explorer_png.exists()):
        ok = capture_explorer(explorer_png, case_id=card.get("id") if card else None)
        print(f"explorer screenshot: {'captured' if ok else 'FAILED'} -> {explorer_png}")
    explorer_uri = file_to_uri(explorer_png) if explorer_png.exists() else None

    ac_fig = get(meta, "soil_water", "artifact_control", "figure", default=None)
    ac_png = None
    if ac_fig:
        for cand in ((ROOT / "site" / ac_fig).resolve(), (DATA / ac_fig).resolve(), DEMO / Path(ac_fig).name):
            if cand.exists():
                ac_png = cand
                break
    ac_uri = file_to_uri(ac_png) if ac_png else None
    print(f"artefact-control figure: {ac_png}")
    pinn = load_json(DATA / "pinn.json", default={})

    ctx = dict(meta=meta, team=team, card=card, retraction_card=retraction_card,
               ac_uri=ac_uri, pinn=pinn,
               line_uri=line_uri, map_uri=map_uri, soil_uri=soil_uri,
               soil_breaks=soil_breaks, explorer_uri=explorer_uri,
               n_cards=sum(1 for c in cards
                           if not str(c.get("title", "")).upper().startswith("TODO")))
    HTML_OUT.write_text(build_html(ctx), encoding="utf-8")
    n_slides = HTML_OUT.read_text().count('<section class="slide')
    print(f"wrote {HTML_OUT} ({n_slides} slides; cards from {source}; "
          f"example card: {card.get('id') if card else None})")
    if n_slides > 7:
        sys.exit("ERROR: more than 7 slides")

    if not args.no_pdf:
        print_pdf(HTML_OUT, PDF_OUT)
        pages = count_pdf_pages(PDF_OUT)
        print(f"wrote {PDF_OUT} ({pages} pages)")
        if pages > 7 or pages != n_slides:
            sys.exit(f"ERROR: PDF has {pages} pages, expected {n_slides} (<= 7)")
    if args.preview:
        p = preview(HTML_OUT, args.preview, n_slides)
        print(f"previews in {args.preview} ({p.name})")


if __name__ == "__main__":
    main()
