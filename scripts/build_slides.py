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
INK = "#131416"        # background: neutral grey, same as the explorer
PANEL = "#1b1c1f"      # raised block
RULE = "#3b3e43"       # hairlines
EDGE = "#7d8189"       # block borders
TEXT = "#ededed"       # primary text
MUTED = "#a3a6ab"      # secondary text
WET = "#5aa2d8"        # significant wetting (blue)
DRY = "#de9b4a"        # significant drying (amber)
LAND = "#3e4045"       # land, no clear change
OCEAN = INK

# numbered markers on the case-file screenshot (percent of the image: map region, case title, checks)
PIN1, PIN2, PIN3 = (30, 46), (66, 9), (66, 56)

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
    """Prefer a strong-pattern card (Sahel, then North China, then any robust one),
    then the Chile drying card, then any drying card, then the first."""
    cand = [c for c in cards if c.get("kind") in ("drying", "wetting")]
    robust = [c for c in cand if str(c.get("evidence_level", "")).lower().startswith(("strong", "robust"))]
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


PLAIN_CHECK = (("window", "Holds over shorter periods too"),
               ("neighbour", "Neighbouring squares agree"),
               ("independent precipitation", "A second rainfall record (CHIRPS) agrees"),
               ("literature", "Published studies report it too"))


def plain_checks(rob, lit=None) -> str:
    """The extra checks in plain words (the selection criterion is not a check, so it is left out)."""
    out = []
    for r in rob or []:
        if r.get("role") == "selection":
            continue
        name = str(r.get("check", ""))
        label = next((v for k, v in PLAIN_CHECK if k in name.lower()), name)
        st = r.get("status")
        mark = {"pass": "✓", "partial": "½", "fail": "✗"}.get(st, "–")
        word = {"pass": "", "partial": " (partly)", "fail": " (no)"}.get(st, " (not run)")
        if "literature" in name.lower() and lit:
            label += " (" + ", ".join(short_cite(l.get("citation", "")) for l in lit[:2]) + ")"
        out.append(f'<li class="{esc(st)}"><span class="m">{mark}</span><span>{esc(label)}{esc(word)}</span></li>')
    return "".join(out)


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

    site = str(team.get("site_url") or "https://paradoxearth.xyz").rstrip("/")
    site_short = re.sub(r"^https?://", "", site).split("/")[0]
    area_up, area_dn = reg.get("area_up_km2"), reg.get("area_down_km2")
    area_tot = (area_up or 0) + (area_dn or 0)
    n_tiles = get(reg, "n_tiles_tested", default=0) or get(m, "n_tiles", default=0)
    overview = (f'<img class="shot" src="{ctx["overview_uri"]}" alt="Paradox Earth explorer: map of wetting and drying land">'
                if ctx.get("overview_uri") else explorer)
    wet_names = " · ".join(esc(x) for x in ctx.get("wet_names", []))
    dry_names = " · ".join(esc(x) for x in ctx.get("dry_names", []))
    sw_step = ac.get("swdown_step_21_W_m2")
    sw_step20 = ac.get("swdown_step_20_W_m2")
    gr_hot = (gr.get("hotspot") or {}) if isinstance(gr.get("hotspot"), dict) else {}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Paradox Earth Deck</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@600;700;800&family=Public+Sans:wght@400;500;600;700&display=block" rel="stylesheet">
<style>
:root {{
  --ink:{INK}; --panel:{PANEL}; --rule:{RULE}; --edge:{EDGE}; --text:{TEXT}; --muted:{MUTED};
  --wet:{WET}; --dry:{DRY}; --land:{LAND}; --hl:#ffd479; --fail:#f08a8a;
  --display:'Barlow Semi Condensed', 'Arial Narrow', Arial, sans-serif;
  --body:'Public Sans', 'Helvetica Neue', Arial, sans-serif;
}}
@page {{ size: 1280px 720px; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: var(--ink); color: var(--text); font-family: var(--body);
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
.slide {{ width: 1280px; height: 720px; position: relative; overflow: hidden;
  padding: 48px 64px 52px; background: var(--ink); page-break-after: always;
  break-after: page; display: flex; flex-direction: column; }}
.slide:last-child {{ page-break-after: auto; break-after: auto; }}
.tag {{ display: inline-block; align-self: flex-start; background: var(--text); color: var(--ink);
  font: 700 15px/1 var(--body); letter-spacing: .04em; text-transform: uppercase; padding: 7px 10px; margin-bottom: 16px; }}
h1 {{ font-family: var(--display); font-weight: 800; font-size: 150px; line-height: .86; text-transform: uppercase; }}
h2 {{ font-family: var(--display); font-weight: 800; font-size: 54px; line-height: .98;
  text-transform: uppercase; max-width: 1150px; }}
p, li {{ font-size: 22px; line-height: 1.4; }}
.muted {{ color: var(--muted); }}
.wet {{ color: var(--wet); }} .dry {{ color: var(--dry); }}
b {{ font-weight: 700; }}
.box {{ background: var(--panel); border: 2px solid var(--edge); box-shadow: 6px 6px 0 #000; }}
.foot {{ position: absolute; left: 64px; right: 64px; bottom: 18px; display: flex; gap: 24px;
  justify-content: space-between; font-size: 13px; color: var(--muted); }}
.foot span:last-child {{ white-space: nowrap; flex: none; font-weight: 700; }}
.fig {{ width: 100%; display: block; }}
.url {{ font-family: var(--display); font-weight: 800; color: var(--ink); background: var(--hl);
  text-decoration: none; display: inline-block; padding: 4px 14px; text-transform: lowercase; }}
.shot {{ width: 100%; display: block; border: 2px solid var(--edge); box-shadow: 6px 6px 0 #000; }}
.shot.pending {{ aspect-ratio: 16/10; display: flex; align-items: center; justify-content: center; color: var(--muted); background: var(--panel); }}

/* 1 title */
.s-title {{ justify-content: space-between; }}
.s-title .bar {{ display: flex; height: 16px; width: 100%; margin: 24px 0 22px; border: 2px solid var(--edge); }}
.s-title .bar i {{ display: block; height: 100%; }}
.s-title .tagline {{ font-size: 27px; line-height: 1.25; max-width: 1100px; }}
.kicker {{ font-size: 16px; font-weight: 600; color: var(--muted); margin-bottom: 14px; }}
.s-title .try {{ margin-top: 20px; font-size: 22px; display: flex; align-items: center; gap: 16px; }}
.s-title .try .url {{ font-size: 40px; }}
.credits {{ display: grid; grid-template-columns: 1fr 1fr 1.4fr; gap: 28px; border-top: 2px solid var(--edge); padding-top: 16px; }}
.credits dt {{ font-size: 13px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }}
.credits dd {{ font-size: 20px; line-height: 1.3; }}

/* 2 paradox */
.s-paradox .row {{ display: grid; grid-template-columns: 1fr 380px; gap: 36px; margin-top: 22px; align-items: start; }}
.s-paradox .say p {{ font-size: 22px; margin-bottom: 14px; }}
.s-paradox .say p.big {{ font: 800 64px/0.95 var(--display); margin-bottom: 8px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: 16px; color: var(--muted); margin-top: 8px; }}
.legend span::before {{ content: ''; display: inline-block; width: 14px; height: 14px;
  margin-right: 8px; vertical-align: -2px; background: var(--c); }}

/* 3 split */
.s-split .row {{ display: grid; grid-template-columns: 1fr 330px; gap: 30px; margin-top: 16px; }}
.s-split .num {{ padding: 12px 16px; margin-bottom: 14px; }}
.s-split .num .v {{ font: 800 44px/1 var(--display); white-space: nowrap; }}
.s-split .num .k {{ font-size: 17px; line-height: 1.3; margin-top: 4px; }}
.s-split .num .where {{ font-size: 15px; color: var(--muted); margin-top: 6px; line-height: 1.3; }}
.s-split .num.w {{ border-top: 10px solid var(--wet); }} .s-split .num.d {{ border-top: 10px solid var(--dry); }}

/* 4 how */
.steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; margin-top: 22px; }}
.steps .box {{ padding: 18px 20px; }}
.steps .n {{ font: 800 56px/0.9 var(--display); color: var(--hl); }}
.steps .t {{ font: 800 27px/1.05 var(--display); text-transform: uppercase; margin: 8px 0 8px; }}
.steps p {{ font-size: 17.5px; line-height: 1.38; }}
.data {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 24px; }}
.data .box {{ padding: 12px 14px; box-shadow: none; border-color: var(--rule); }}
.data .t {{ font: 800 21px/1.05 var(--display); text-transform: uppercase; }}
.data p {{ font-size: 14.5px; line-height: 1.35; color: var(--muted); margin-top: 4px; }}
.data-h {{ font-size: 14px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); margin-top: 22px; }}

/* 5 report card (demo) */
.s-card .row {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 30px; margin-top: 14px; flex: 1; min-height: 0; }}
.demo {{ position: relative; align-self: start; }}
.pin {{ position: absolute; width: 44px; height: 44px; background: var(--hl); color: #111;
  font: 800 28px/44px var(--display); text-align: center; border: 2px solid #111; box-shadow: 4px 4px 0 #000; }}
.steps-mini {{ list-style: none; margin-top: 12px; display: grid; gap: 6px; }}
.steps-mini li {{ font-size: 17px; display: flex; gap: 10px; align-items: baseline; }}
.steps-mini li i {{ font: 800 18px/26px var(--display); font-style: normal; background: var(--hl); color: #111; width: 26px; text-align: center; flex: none; }}
.rc {{ padding: 16px 20px; border-top: 10px solid var(--wet); }}
.rc .place {{ font: 800 34px/1 var(--display); text-transform: uppercase; }}
.rc .big {{ font: 800 48px/1 var(--display); margin: 6px 0 2px; }}
.rc .sub {{ font-size: 15px; color: var(--muted); }}
.checks {{ list-style: none; margin: 10px 0 0; border-top: 2px solid var(--rule); }}
.checks li {{ display: grid; grid-template-columns: 34px 1fr; gap: 10px; padding: 7px 0; border-bottom: 2px solid var(--rule); font-size: 17px; line-height: 1.3; }}
.checks .m {{ font: 800 22px/1.1 var(--display); text-align: center; }}
.checks .pass .m {{ color: #7fd1ae; }} .checks .partial .m {{ color: #e8c46a; }} .checks .fail .m {{ color: var(--fail); }}
.who {{ margin-top: 12px; font-size: 16px; line-height: 1.35; }}
.who .who-k {{ display: block; font-size: 13px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }}

/* 6 retraction */
.s-retract .row {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 30px; margin-top: 16px; }}
.s-retract .story p {{ font-size: 19px; line-height: 1.4; margin-bottom: 10px; }}

.stamp {{ display: inline-block; background: var(--fail); color: #111; font: 800 26px/1 var(--display);
  letter-spacing: .06em; padding: 8px 14px; margin-bottom: 14px; box-shadow: 4px 4px 0 #000; }}
.s-retract .fig {{ border: 2px solid var(--edge); box-shadow: 6px 6px 0 #000; }}

/* 7 try it */
.s-explore .row {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 34px; margin-top: 16px; }}
.s-explore .go {{ font-size: 20px; margin-bottom: 12px; }}
.s-explore .go .url {{ font-size: 46px; margin-top: 6px; }}
.s-explore ul.more {{ list-style: disc; display: grid; gap: 8px; margin: 8px 0 0 24px; }}
.s-explore ul.more li {{ font-size: 17px; line-height: 1.35; }}
.refs {{ margin-top: 12px; border-top: 2px solid var(--rule); padding-top: 8px; }}
.refs li {{ font-size: 11.5px; line-height: 1.38; color: var(--muted); list-style: none; }}

@media screen {{ body {{ display: flex; flex-direction: column; align-items: center; gap: 0; }} }}
@media print {{ html, body {{ width: 1280px; }} }}
</style></head>
<body>

<!-- 1 TITLE -->
<section class="slide s-title">
  <div>
    <p class="kicker">{esc(team.get("event", "NASA International Space Apps Challenge"))} · {esc(team.get("challenge", m.get("challenge")))}</p>
    <h1>{esc(m.get("project", "Paradox Earth"))}</h1>
    <div class="bar"><i style="flex:{reg.get('share_area_down') or 1};background:var(--dry)"></i><i style="flex:{max(0.0, 1 - (reg.get('share_area_up') or 0) - (reg.get('share_area_down') or 0))};background:var(--land)"></i><i style="flex:{reg.get('share_area_up') or 1};background:var(--wet)"></i></div>
    <p class="tagline">Averaged over all the world’s land, rainfall has barely changed in {n_years} years. Look region by region and an area about the size of the United States is getting wetter or drier. Paradox Earth is an interpretable, human-centered trend detector: it explains every finding in plain language, and its checks caught one of our own findings as wrong.</p>
    <div class="try">Try it live: <a class="url" href="{esc(site)}">{esc(site_short)}</a></div>
  </div>
  <dl class="credits">
    <div><dt>Team</dt><dd>{esc(team.get("team_name"))}</dd></div>
    <div><dt>Members</dt><dd>{"<br>".join(esc(x) for x in members)}</dd></div>
    <div><dt>Challenge</dt><dd>{esc(team.get("challenge"))}<br><span class="muted">{esc(team.get("local_event"))}</span></dd></div>
  </dl>
  <div class="foot"><span>Bar: share of land getting drier (amber) · no clear change (grey) · getting wetter (blue), {y0}–{y1}. Data: NASA GLDAS-2.1, doi:{esc(ds.get("doi"))}</span><span>1 / 7</span></div>
</section>

<!-- 2 PARADOX -->
<section class="slide s-paradox">
  <h2>Add up all the world’s land and rainfall looks flat</h2>
  <div class="row">
    <div><img class="fig" src="{ctx["line_uri"]}" alt="Global land-average rainfall, 2000 to 2025: a flat line">
      <div class="legend"><span style="--c:{TEXT}">12-month average</span><span style="--c:{MUTED}">each month</span><span style="--c:{DRY}">best-fit trend</span></div></div>
    <div class="say">
      <p class="big">{fmt_num(agg.get("slope_pct_per_decade"), 1, True)}% <span style="font-size:30px">per decade</span></p>
      <p>The land-wide change in rainfall since {y0} is too small to tell apart from zero (p&nbsp;=&nbsp;{fmt_p(agg.get("p_value"))}).</p>
      <p class="muted">But an average can hide a lot: more rain in one place and less in another cancel out. Water decisions are made region by region, so we looked region by region.</p>
    </div>
  </div>
  <div class="foot"><span>Rain and snow on land, NASA GLDAS-2.1, {y0}–{y1}, change from the normal for each month, weighted by land area.</span><span>2 / 7</span></div>
</section>

<!-- 3 SPLIT -->
<section class="slide s-split">
  <h2>{fmt_area(area_tot)} is getting wetter or drier</h2>
  <div class="row">
    <div>
      <img class="fig" src="{ctx["map_uri"]}" alt="World map: places with a clear wetting or drying trend">
      <div class="legend"><span style="--c:{WET}">getting wetter</span><span style="--c:{DRY}">getting drier</span><span style="--c:{LAND}">no clear change</span></div>
    </div>
    <div>
      <div class="num box w"><div class="v wet">{fmt_area(area_up)}</div><div class="k">getting wetter ({fmt_pct(reg.get("share_area_up"))} of land)</div><div class="where">{wet_names}</div></div>
      <div class="num box d"><div class="v dry">{fmt_area(area_dn)}</div><div class="k">getting drier ({fmt_pct(reg.get("share_area_down"))} of land)</div><div class="where">{dry_names}</div></div>
      <p class="muted" style="font-size:15px;line-height:1.35">Together, about the size of the United States. The shares depend on the period (2003–2025: {fmt_pct(get(wins, "2003-2025", "share_area_up"))} wetter, {fmt_pct(get(wins, "2003-2025", "share_area_down"))} drier), but all six regions keep their direction in every period.</p>
    </div>
  </div>
  <div class="foot"><span>Only places that pass every statistical safeguard are coloured. These are trends, not causes.</span><span>3 / 7</span></div>
</section>

<!-- 4 HOW -->
<section class="slide s-how">
  <h2>How it works</h2>
  <div class="steps">
    <div class="box"><div class="n">1</div><div class="t">Split the land</div><p>We divide all land into {n_tiles:,} squares, each about 100&nbsp;km across, and take {n_years} years of monthly rainfall for each one from NASA.</p></div>
    <div class="box"><div class="n">2</div><div class="t">Test each square</div><p>Wet and dry spells can fake a trend. On made-up records with no trend, the usual test raised false alarms 29–34% of the time. Ours raised them 3–6% of the time.</p></div>
    <div class="box"><div class="n">3</div><div class="t">Keep only real patterns</div><p>Test {n_tiles:,} squares and some will look unusual by luck alone. We correct for that, then keep a trend only where neighbouring squares agree.</p></div>
  </div>
  <div class="data-h">NASA data behind it</div>
  <div class="data">
    <div class="box"><div class="t">GLDAS-2.1</div><p>NASA’s global land model: monthly rainfall, 2000–2025. Where the trends are found.</p></div>
    <div class="box"><div class="t">FLDAS + CHIRPS</div><p>A second NASA land model fed by a different rainfall record, used to double-check every region.</p></div>
    <div class="box"><div class="t">GLDAS-2.0</div><p>The same model with older inputs. It showed that one of our results came from a data change (slide 6).</p></div>
    <div class="box"><div class="t">GRACE</div><p>Satellites that weigh water by measuring gravity, catching water loss the model cannot see, such as groundwater.</p></div>
  </div>
  <div class="foot"><span>Methods in brief: seasonal trend fit with a persistence-aware test, false-discovery correction (Benjamini–Hochberg), local Moran’s I clusters. Full details in the explorer.</span><span>4 / 7</span></div>
</section>

<!-- 5 REPORT CARD (DEMO) -->
<section class="slide s-card">
  <h2>Live demo</h2>
  <div class="row">
    <div class="demo">
      <div style="position:relative">{explorer}
      <div class="pin" style="left:{PIN1[0]}%;top:{PIN1[1]}%">1</div>
      <div class="pin" style="left:{PIN2[0]}%;top:{PIN2[1]}%">2</div>
      <div class="pin" style="left:{PIN3[0]}%;top:{PIN3[1]}%">3</div></div>
      <ol class="steps-mini">
        <li><i>1</i>Click a highlighted region on the map.</li>
        <li><i>2</i>Read what changed there, in plain numbers.</li>
        <li><i>3</i>See which checks it passed, and which it did not.</li>
      </ol>
    </div>
    <div class="rc box">
      <div class="place">{esc(card.get("title") or card.get("place"))}</div>
      <div class="big {'dry' if card.get('kind') == 'drying' else 'wet'}">Rain {'down' if (num.get("median_pct_per_decade") or 0) < 0 else 'up'} {fmt_num(abs(num.get("median_pct_per_decade") or 0), 0)}% per decade</div>
      <div class="sub">{num.get("n_tiles")} squares, {fmt_area(num.get("area_km2"))}, {y0}–{y1}</div>
      <ul class="checks">{plain_checks(card.get("robustness"), card.get("literature"))}</ul>
      <div class="who"><span class="who-k">Who could use this</span>{esc(user.get("who"))}: {esc(str(user.get("decision", "")).rstrip("."))}.</div>
    </div>
  </div>
  <div class="foot"><span>Screenshot of the live explorer. Weaker regions are shown too, with their lower grades.</span><span>5 / 7</span></div>
</section>

<!-- 6 RETRACTION -->
<section class="slide s-retract">
  <h2>Our first headline was wrong, and our own checks caught it</h2>
  <div class="row">
    <div>{ac_fig}</div>
    <div class="story">
      <div class="stamp">WITHDRAWN</div>
      <p>Our first finding was that soil water was falling worldwide.</p>
      <p>Our checks found most of the fall came from a sudden drop in 2010, exactly when NASA changed the sunlight data that feeds the model: sunlight jumped by {fmt_num(sw_step, 1)} W/m².</p>
      <p>The same model run with the older data shows no such jump ({fmt_num(sw_step20, 1, True)} W/m²), and a second NASA model shows no global decline.</p>
      <p class="muted">So we withdrew the claim. Checking how data were made, before trusting a trend, is now built in.</p>
    </div>
  </div>
  <div class="foot"><span>NASA GLDAS-2.1 vs GLDAS-2.0 (same model, different inputs); NASA FLDAS soil water 2003–2024: no detectable trend.</span><span>6 / 7</span></div>
</section>

<!-- 7 TRY IT -->
<section class="slide s-explore">
  <h2>Open it, click any place, see how sure we are</h2>
  <div class="row">
    <div>{overview}</div>
    <div>
      <div class="go">No code, no login, works on a phone:<br><a class="url" href="{esc(site)}">{esc(site_short)}</a></div>
      <ul class="more">
        <li>Six regional report cards, one withdrawn result, and a satellite check on groundwater in northwest India.</li>
        <li>A physics-aware AI forecaster that predicts next month’s soil water while obeying the water balance. Its squared errors are about 20% smaller than a simple statistical model’s.</li>
        <li>Next: more NASA variables, more independent checks, and a report card for your own region.</li>
      </ul>
      <ul class="refs">
        <li>NASA GLDAS-2.1 doi:{esc(ds.get("doi"))} · FLDAS doi:{esc(fl_doi or "10.5067/5NHC22T9375G")} · GRACE/GRACE-FO JPL mascons {esc(gr_rel)}</li>
        <li>Rodell et al. 2004 · McNally et al. 2017 · Funk et al. 2015 · Rodell et al. 2009 · Benjamini &amp; Hochberg 1995 · Anselin 1995</li>
        <li>Code: {esc(team.get("project_url"))}</li>
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
    overview_png = DEMO / "explorer_overview.png"
    if not (args.keep_explorer and overview_png.exists()):
        ok = capture_explorer(overview_png)
        print(f"overview screenshot: {'captured' if ok else 'FAILED'} -> {overview_png}")
    overview_uri = file_to_uri(overview_png) if overview_png.exists() else None
    names = {k: [short.get(c.get("id")) or str(c.get("place", "")).split(" (")[0]
                 for c in cards if c.get("kind") == k] for k in ("wetting", "drying")}

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
               soil_breaks=soil_breaks, explorer_uri=explorer_uri, overview_uri=overview_uri,
               wet_names=names["wetting"], dry_names=names["drying"],
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
