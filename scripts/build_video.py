#!/usr/bin/env python
"""Render the 28-second Paradox Earth demo video.

Reads every number from site/data/*.json (meta, tiles, regions, series,
optional cards) and demo/team.json, so it can be re-run after the data are
re-exported.  Frames are drawn with matplotlib (Agg) at 1920x1080 / 30 fps
and piped straight into ffmpeg (H.264, yuv420p, no audio).

    .venv/bin/python scripts/build_video.py            # full render
    .venv/bin/python scripts/build_video.py --stills 1 8 15 21 24.5 27
                                                        # PNG previews only
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "site", "data")
TEAM = os.path.join(ROOT, "demo", "team.json")
OUT = os.path.join(ROOT, "demo", "paradox_earth_30s.mp4")

W, H, FPS, DURATION = 1920, 1080, 30, 28.0

# ---------------------------------------------------------------- palette ---
BG = "#0a1020"
PANEL = "#121a2e"
LAND = np.array([0.20, 0.24, 0.31])
WET = "#3fb6e8"
DRY = "#f08a3c"
INK = "#eef2f8"
MUTED = "#93a0b8"
ALERT = "#ff5d5d"
GOOD = "#5fd39a"
WET_RGB = np.array(matplotlib.colors.to_rgb(WET))
DRY_RGB = np.array(matplotlib.colors.to_rgb(DRY))


def pick_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Avenir Next", "Helvetica Neue", "Helvetica", "DejaVu Sans"):
        if cand in names:
            return cand
    return "DejaVu Sans"


plt.rcParams["font.family"] = [pick_font(), "DejaVu Sans"]


# ------------------------------------------------------------------ data ---
def load_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def coarse_place(lat, lon):
    """Fallback geographic label when cards.json does not supply a place."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    coords = f"{abs(lat):.0f}°{ns}, {abs(lon):.0f}°{ew}"
    if -170 <= lon <= -30:
        cont = "South America" if lat < 12 else "North America"
    elif -30 < lon <= 60 and lat < 37:
        cont = "Africa" if lat < 33 else "Mediterranean"
    elif -30 < lon <= 60:
        cont = "Europe"
    elif lon > 110 and lat < -10:
        cont = "Australia"
    else:
        cont = "Asia"
    return f"{cont} ({coords})"


def load_everything():
    meta = load_json(os.path.join(DATA, "meta.json"))
    tiles = load_json(os.path.join(DATA, "tiles.json"))
    regions = load_json(os.path.join(DATA, "regions.json"))
    cards = load_json(os.path.join(DATA, "cards.json"), default={"cards": []})
    team = load_json(TEAM, default={})

    cols = tiles["cols"]
    ix = {c: i for i, c in enumerate(cols)}
    rows = tiles["rows"]
    deg = float(tiles.get("tile_deg", 1.0))
    ny, nx = int(round(180 / deg)), int(round(360 / deg))

    land = np.zeros((ny, nx), bool)
    sig = np.zeros((ny, nx), np.int8)
    qrank = np.full((ny, nx), np.nan)
    tile_pct, tile_lat = {}, {}
    qs = []
    for r in rows:
        j = int(math.floor((r[ix["south"]] + 90) / deg + 1e-6))
        i = int(math.floor((r[ix["west"]] + 180) / deg + 1e-6))
        if not (0 <= j < ny and 0 <= i < nx):
            continue
        land[j, i] = True
        s = r[ix["sig"]] or 0
        sig[j, i] = s
        q = r[ix["q"]]
        if s != 0 and q is not None:
            qs.append((q, j, i))
        tile_pct[r[ix["id"]]] = r[ix["pct"]]
        tile_lat[r[ix["id"]]] = r[ix["south"]] + deg / 2
    # reveal order: strongest evidence (lowest q) first
    qs.sort()
    for k, (_, j, i) in enumerate(qs):
        qrank[j, i] = k / max(1, len(qs) - 1)

    # ---------------- series: global / wetting / drying weighted means ----
    sig_by_id = {r[ix["id"]]: (r[ix["sig"]] or 0) for r in rows}
    acc = {k: None for k in ("all", "up", "down")}
    wsum = {k: 0.0 for k in acc}
    start = "2000-01"
    for path in sorted(glob.glob(os.path.join(DATA, "series", "*.json"))):
        blk = load_json(path)
        start = blk.get("start", start)
        for tid, vals in blk["tiles"].items():
            if tid not in tile_lat:
                continue
            v = np.asarray([np.nan if x is None else x for x in vals], float)
            w = math.cos(math.radians(tile_lat[tid]))
            groups = ["all"]
            s = sig_by_id.get(tid, 0)
            if s > 0:
                groups.append("up")
            elif s < 0:
                groups.append("down")
            for g in groups:
                vv = np.nan_to_num(v) * w
                acc[g] = vv if acc[g] is None else acc[g] + vv
                wsum[g] += w
    series = {}
    for g in acc:
        if acc[g] is None or wsum[g] == 0:
            series[g] = np.zeros(312)
        else:
            series[g] = acc[g] / wsum[g]
    y0, m0 = (int(x) for x in start.split("-"))
    n = len(series["all"])
    t_years = y0 + (m0 - 1) / 12 + np.arange(n) / 12.0

    def smooth(x, k=12):
        ker = np.ones(k) / k
        return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), ker, "valid")

    for g in list(series):
        series[g + "_s"] = smooth(series[g])

    # ---------------- pick the two featured regions ----------------------
    reg_by_id = {r["id"]: r for r in regions.get("up", []) + regions.get("down", [])}

    def featured(kind, pool):
        card = next((c for c in cards.get("cards", [])
                     if c.get("kind") == kind and c.get("region_id") in reg_by_id), None)
        if card:
            reg = reg_by_id[card["region_id"]]
        elif pool:
            reg = max(pool, key=lambda r: r.get("n_tiles", 0))
        else:
            return None
        pct = None
        if card:
            pct = (card.get("numbers") or {}).get("median_pct_per_decade")
        if pct is None:
            pct = reg.get("median_slope_pct_per_decade")
        if pct is None:
            vals = [tile_pct[t] for t in reg.get("tiles", []) if tile_pct.get(t) is not None]
            pct = float(np.median(vals)) if vals else None
        place = (card or {}).get("place") or coarse_place(reg["centroid_lat"], reg["centroid_lon"])
        user = (card or {}).get("user") or {}
        checks = (card or {}).get("robustness") or []
        return dict(reg=reg, pct=pct, place=place, user=user, checks=checks,
                    slope=reg.get("median_slope_per_decade"), title=(card or {}).get("title"))

    wet = featured("wetting", regions.get("up", []))
    dry = featured("drying", regions.get("down", []))

    return dict(meta=meta, team=team, land=land, sig=sig, qrank=qrank, deg=deg,
                series=series, t_years=t_years, wet=wet, dry=dry, cards=cards)


# --------------------------------------------------------------- helpers ---
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ramp(t, t0, t1):
    return ease((t - t0) / (t1 - t0)) if t1 > t0 else float(t >= t0)


def window(t, t0, t1, fade=0.35):
    """1 inside [t0, t1], easing in/out over `fade` seconds."""
    return min(ramp(t, t0, t0 + fade), 1 - ramp(t, t1 - fade, t1))


def txt(fig, x, y, s, size=30, color=INK, alpha=1.0, weight="normal", ha="left",
        va="center", **kw):
    if alpha <= 0.01:
        return None
    return fig.text(x, y, s, fontsize=size, color=color, alpha=clamp(alpha),
                    fontweight=weight, ha=ha, va=va, **kw)


def fmt_pct(p, digits=1):
    return "n/a" if p is None else f"{p:+.{digits}f}%"


def fmt_share(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def pill(fig, x, y, w, h, color, alpha=1.0, ec=None, lw=0):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        transform=fig.transFigure, fc=color, ec=ec or color, lw=lw, alpha=clamp(alpha),
        mutation_aspect=W / H))


# ------------------------------------------------------------- the scenes ---
class Renderer:
    def __init__(self, D):
        self.D = D
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
        m = D["meta"]
        self.ds = m["dataset"]
        self.agg = m["headline"]["aggregate"]
        self.reg = m["headline"]["regional"]
        self.team = D["team"]
        land = D["land"]
        ny, nx = land.shape
        self.base = np.zeros((ny, nx, 4))
        self.base[..., :3] = LAND
        self.sigcol = np.zeros((ny, nx, 3))
        self.sigcol[D["sig"] > 0] = WET_RGB
        self.sigcol[D["sig"] < 0] = DRY_RGB
        self.full_extent = (-180.0, 180.0, -60.0, 84.0)  # crop Antarctica / empty ocean

    # ---------------------------------------------------------- map ---
    def map_rgba(self, land_a, reveal):
        """reveal in [0,1]: fraction of significant tiles (by q-rank) shown."""
        D = self.D
        img = self.base.copy()
        img[..., 3] = np.where(D["land"], land_a, 0.0)
        qr = D["qrank"]
        has = ~np.isnan(qr)
        # each tile fades in over a short slice of the reveal
        a = np.zeros_like(qr)
        a[has] = np.clip((reveal * 1.15 - qr[has]) / 0.15, 0, 1)
        a *= land_a
        for c in range(3):
            img[..., c] = np.where(has, img[..., c] * (1 - a) + self.sigcol[..., c] * a, img[..., c])
        return img

    def draw_map(self, t, rect, land_a, reveal, extent, highlight=None):
        ax = self.fig.add_axes(rect)
        ax.set_facecolor(BG)
        ax.imshow(self.map_rgba(land_a, reveal), origin="lower",
                  extent=(-180, 180, -90, 90), interpolation="nearest", zorder=1)
        w, e, s, n = extent
        ax.set_xlim(w, e)
        ax.set_ylim(s, n)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        if highlight:
            for reg, color, a in highlight:
                if a <= 0.01:
                    continue
                bw, bs, be, bn = reg["bbox"]
                pad = 1.0
                ax.add_patch(Rectangle((bw - pad, bs - pad), be - bw + 1 + 2 * pad,
                                       bn - bs + 1 + 2 * pad, fill=False, ec=color,
                                       lw=3.5, alpha=a, zorder=3))
        return ax

    def zoom_extent(self, reg, aspect):
        bw, bs, be, bn = reg["bbox"]
        cx, cy = (bw + be + 1) / 2, (bs + bn + 1) / 2
        half_w = max(max((be - bw + 1) * 0.5, (bn - bs + 1) * 0.5 * aspect) * 1.9 + 4, 24.0)
        half_h = half_w / aspect
        return (cx - half_w, cx + half_w, cy - half_h, cy + half_h)

    @staticmethod
    def lerp_extent(a, b, u):
        return tuple(x + (y - x) * u for x, y in zip(a, b))

    # ------------------------------------------------------- chrome ---
    def watermark(self):
        txt(self.fig, 0.985, 0.022,
            f"AI-assisted production · data: NASA GLDAS-2.1 (doi:{self.ds['doi']})",
            size=18, color=INK, ha="right", alpha=0.8)

    def lower_title(self, t, a):
        txt(self.fig, 0.03, 0.955, "PARADOX EARTH", size=20, color=MUTED,
            weight="bold", alpha=a * 0.9)

    # ------------------------------------------------------ scenes ----
    def s_intro(self, t):
        f = self.fig
        a = window(t, 0.0, 2.0, 0.3) if t > 1.0 else ramp(t, 0.0, 0.35)
        team = self.team.get("team_name", "Team")
        members = " · ".join(self.team.get("members", []))
        txt(f, 0.5, 0.60, team, size=64, weight="bold", ha="center", alpha=a)
        txt(f, 0.5, 0.50, members, size=32, color=MUTED, ha="center", alpha=a)
        txt(f, 0.5, 0.38, "presents  PARADOX EARTH", size=30, color=WET,
            ha="center", alpha=a, weight="bold")

    def s_challenge(self, t):
        f = self.fig
        a = window(t, 2.0, 5.0, 0.35)
        ev = self.team.get("event", "NASA International Space Apps Challenge 2026")
        local = self.team.get("local_event")
        txt(f, 0.5, 0.66, ev.upper(), size=28, color=MUTED, ha="center", alpha=a,
            weight="bold")
        txt(f, 0.5, 0.53, "Challenge:", size=34, color=MUTED, ha="center", alpha=a)
        txt(f, 0.5, 0.43, self.team.get("challenge", "Be An Earth System Trend Detective!"),
            size=60, weight="bold", ha="center", alpha=a)
        if local:
            txt(f, 0.5, 0.31, local, size=26, color=MUTED, ha="center", alpha=a * 0.9)

    def global_line_axes(self, a, rect=(0.12, 0.17, 0.76, 0.34)):
        ax = self.fig.add_axes(rect)
        ax.set_facecolor("none")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=16, length=0)
        ax.set_yticks([])
        ax.axhline(0, color=MUTED, lw=1, alpha=0.25 * a)
        for lbl in ax.get_xticklabels():
            lbl.set_alpha(a)
        return ax

    def s_title(self, t):
        f = self.fig
        a = window(t, 5.0, 10.2, 0.4)
        txt(f, 0.5, 0.80, self.D["meta"]["project"], size=110, weight="bold",
            ha="center", alpha=a)
        txt(f, 0.5, 0.68, self.D["meta"]["tagline"], size=38, color=MUTED,
            ha="center", alpha=a)
        # the flat global average line draws in
        S = self.D["series"]
        ty = self.D["t_years"]
        draw_u = ramp(t, 5.4, 8.2)
        ax = self.global_line_axes(a)
        k = max(2, int(len(ty) * draw_u))
        lim = max(abs(S["up_s"]).max(), abs(S["down_s"]).max(), abs(S["all_s"]).max()) * 1.15
        ax.plot(ty[:k], S["all_s"][:k], color=INK, lw=4, alpha=a)
        ax.set_xlim(ty[0], ty[-1] + 1 / 12)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
        ax.set_xticklabels(["2000", "2005", "2010", "2015", "2020", "2025"], alpha=a)
        ag = self.agg
        cap_a = a * ramp(t, 7.6, 8.4)
        txt(f, 0.5, 0.56,
            f"Global land precipitation, {self.ds['period'][0][:4]}–{self.ds['period'][1][:4]}: "
            f"{ag['slope_pct_per_decade']:+.1f}% per decade  (p = {ag['p_value']:.2f})  →  looks flat",
            size=28, ha="center", alpha=cap_a)

    def method_panel(self, t, a):
        """Left-hand 'how it works' panel shown 10-20 s."""
        f = self.fig
        if a <= 0.01:
            return
        x0 = 0.035
        txt(f, x0, 0.88, "HOW IT WORKS", size=22, color=WET, weight="bold", alpha=a)
        n_tiles = self.D["meta"].get("n_tiles", int(self.D["land"].sum()))
        steps = [
            f"Split land into {n_tiles:,} tiles of {self.D['deg']:g}°",
            "Trend-test every tile\n(autocorrelation-robust test)",
            f"Correct for multiple testing\n(Benjamini–Hochberg, q < {self.reg.get('alpha_fdr', 0.05):g})",
            "Find clusters (local Moran's I)",
            "Grade its own evidence",
        ]
        for k, s in enumerate(steps):
            sa = a * ramp(t, 10.3 + k * 0.9, 10.8 + k * 0.9)
            y = 0.80 - k * 0.095
            txt(f, x0, y, f"{k + 1}", size=26, color=WET, weight="bold", alpha=sa)
            txt(f, x0 + 0.025, y, s, size=21, alpha=sa, linespacing=1.15)
        da = a * ramp(t, 10.5, 11.2)
        y = 0.27
        pill(f, x0 - 0.008, y - 0.125, 0.285, 0.165, PANEL, alpha=da * 0.95)
        txt(f, x0, y, "NASA DATA", size=17, color=MUTED, weight="bold", alpha=da)
        txt(f, x0, y - 0.04, f"GLDAS-{self.ds['version']} Noah, monthly 0.25°", size=20,
            alpha=da, weight="bold")
        txt(f, x0, y - 0.075, f"{self.ds['variable']} · {self.ds['provider']}", size=17,
            color=MUTED, alpha=da)
        txt(f, x0, y - 0.105, f"DOI {self.ds['doi']}", size=19, color=WET, alpha=da,
            weight="bold")

    def s_split_and_map(self, t):
        """10 s - 20 s: the flat line splits, becomes the map, tiles reveal."""
        f = self.fig
        S = self.D["series"]
        ty = self.D["t_years"]
        right = (0.33, 0.12, 0.65, 0.74)
        panel_a = window(t, 10.0, 20.0, 0.4)
        self.lower_title(t, panel_a)

        # (a) split: 10.0 - 13.2
        line_a = 1 - ramp(t, 12.6, 13.3)
        if line_a > 0.01:
            u = ramp(t, 10.3, 12.2)
            lim = max(abs(S["up_s"]).max(), abs(S["down_s"]).max(), abs(S["all_s"]).max()) * 1.15
            ax = self.global_line_axes(line_a, rect=(0.36, 0.22, 0.6, 0.5))
            g = S["all_s"]
            ax.plot(ty, g + (S["up_s"] - g) * u, color=WET, lw=4, alpha=line_a)
            ax.plot(ty, g + (S["down_s"] - g) * u, color=DRY, lw=4, alpha=line_a)
            ax.plot(ty, g, color=INK, lw=3, alpha=line_a * (1 - 0.6 * u))
            ax.set_xlim(ty[0], ty[-1] + 1 / 12)
            ax.set_ylim(-lim, lim)
            ax.set_xticks([2000, 2010, 2020])
            ax.set_xticklabels(["2000", "2010", "2020"], alpha=line_a)
            la = line_a * u
            txt(f, 0.965, 0.70, "wetting tiles", size=24, color=WET, ha="right", alpha=la,
                weight="bold")
            txt(f, 0.965, 0.25, "drying tiles", size=24, color=DRY, ha="right", alpha=la,
                weight="bold")
            txt(f, 0.66, 0.80, "Inside the flat average: a few regions, opposite directions",
                size=30, ha="center", alpha=line_a * ramp(t, 10.6, 11.2))

        # (b) map: from 12.6
        land_a = ramp(t, 12.6, 13.4) * panel_a
        if land_a > 0.01:
            reveal = ramp(t, 13.4, 16.8)
            self.draw_map(t, right, land_a, reveal, self.full_extent)
            sa = land_a * ramp(t, 16.6, 17.3)
            r = self.reg
            txt(f, 0.655, 0.905,
                "Each square is one 1° land tile, coloured only if its trend survives correction",
                size=21, color=MUTED, ha="center", alpha=land_a * ramp(t, 13.0, 13.6))
            if sa > 0.01:
                pill(f, 0.37, 0.05, 0.58, 0.075, PANEL, alpha=sa * 0.95)
                for x, col, lab in ((0.39, WET, f"wetting  {fmt_share(r.get('share_area_up'))}"),
                                    (0.57, DRY, f"drying  {fmt_share(r.get('share_area_down'))}")):
                    f.patches.append(Rectangle((x, 0.076), 0.012, 0.012 * W / H,
                                               transform=f.transFigure, fc=col, alpha=sa))
                    txt(f, x + 0.02, 0.088, lab, size=25, color=col, alpha=sa, weight="bold")
                txt(f, 0.935, 0.088, "of land area · FDR q < 0.05", size=19, color=MUTED,
                    ha="right", alpha=sa)

        self.method_panel(t, panel_a)

    def region_card(self, feat, color, label, a, x=0.035, y=0.74):
        import re
        import textwrap

        def wrap(s, width, max_lines):
            lines = textwrap.wrap(s or "", width)
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines[-1] = lines[-1].rstrip(" ,;.") + "…"
            return lines

        f = self.fig
        if a <= 0.01 or not feat:
            return
        pill(f, x - 0.01, y - 0.62, 0.33, 0.70, PANEL, alpha=a * 0.96)
        txt(f, x, y + 0.035, label, size=20, color=color, weight="bold", alpha=a)
        place = wrap(str(feat["place"]).split(" (")[0], 30, 2)
        txt(f, x, y - 0.03, "\n".join(place), size=27, weight="bold", alpha=a, va="top",
            linespacing=1.1)
        y -= 0.045 * len(place)
        txt(f, x, y - 0.10, fmt_pct(feat["pct"]), size=72, color=color, weight="bold", alpha=a)
        txt(f, x, y - 0.175, "median change per decade", size=19, color=MUTED, alpha=a)
        reg = feat["reg"]
        txt(f, x, y - 0.215, f"{reg.get('n_tiles', 0)} tiles · FDR-significant cluster",
            size=19, color=MUTED, alpha=a)
        u = feat.get("user") or {}
        who = re.sub(r"\s*\(e\.g\.[^)]*\)", "", u.get("who") or "").strip()
        dec = u.get("decision")
        if who:
            txt(f, x, y - 0.265, "WHO DECIDES WITH THIS", size=16, color=color, weight="bold", alpha=a)
            wl = wrap(who, 40, 2)
            txt(f, x, y - 0.29, "\n".join(wl), size=19, alpha=a, va="top", linespacing=1.25)
            if dec:
                txt(f, x, y - 0.30 - 0.042 * len(wl), "\n".join(wrap("Decision: " + dec, 50, 3)),
                    size=17, color=MUTED, alpha=a, va="top", linespacing=1.25)
        checks = feat.get("checks") or []
        if not who and not checks:
            txt(f, x, y - 0.33, "Full Discovery Card in the explorer:\nrobustness checklist, literature,\n"
                "who uses it and for which decision", size=18, color=MUTED, alpha=a,
                linespacing=1.4, va="top")
        if not who and checks:
            if True:
                for k, c in enumerate(checks[:4]):
                    mark = {"pass": "✓", "partial": "~", "fail": "✗"}.get(c.get("status"), "·")
                    col = {"pass": GOOD, "partial": "#e8c547", "fail": ALERT}.get(c.get("status"), MUTED)
                    txt(f, x, y - 0.33 - k * 0.04, f"{mark}  {c.get('check', '')}", size=18,
                        color=col, alpha=a)

    def s_regions(self, t):
        """20 s - 23.6 s: zoom to a wetting then a drying region."""
        f = self.fig
        a = window(t, 20.0, 23.6, 0.25)
        wet, dry = self.D["wet"], self.D["dry"]
        rect = (0.37, 0.08, 0.61, 0.82)
        aspect = (rect[2] * W) / (rect[3] * H)
        full = self.full_extent
        z_wet = self.zoom_extent(wet["reg"], aspect) if wet else full
        z_dry = self.zoom_extent(dry["reg"], aspect) if dry else full
        if t < 21.8:
            ext = self.lerp_extent(full, z_wet, ramp(t, 20.0, 20.8))
        else:
            ext = self.lerp_extent(z_wet, z_dry, ramp(t, 21.8, 22.5))
        hw = ramp(t, 20.5, 20.9) * (1 - ramp(t, 21.7, 21.9))
        hd = ramp(t, 22.3, 22.7)
        hl = []
        if wet:
            hl.append((wet["reg"], WET, hw))
        if dry:
            hl.append((dry["reg"], DRY, hd))
        self.draw_map(t, rect, a, 1.0, ext, highlight=hl)
        self.lower_title(t, a)
        txt(f, 0.035, 0.885, "THE RESULT", size=22, color=WET, weight="bold", alpha=a)
        self.region_card(wet, WET, "WETTING HOTSPOT", a * hw, y=0.74)
        self.region_card(dry, DRY, "DRYING HOTSPOT", a * window(t, 22.2, 23.6, 0.3), y=0.74)

    def s_retraction(self, t):
        f = self.fig
        a = window(t, 23.6, 25.9, 0.3)
        txt(f, 0.5, 0.86, "THE DETECTIVE ALSO GRADES ITSELF", size=24, color=MUTED,
            ha="center", weight="bold", alpha=a)
        sa = ramp(t, 23.8, 24.2) * a
        if sa > 0.01:
            scale = 1.0 + 0.25 * (1 - ramp(t, 23.8, 24.2))
            f.text(0.5, 0.66, "FALSE TREND CAUGHT", fontsize=78 * scale, color=ALERT,
                   ha="center", va="center", fontweight="bold", alpha=sa, rotation=-3,
                   bbox=dict(boxstyle="round,pad=0.35", fc="none", ec=ALERT, lw=5, alpha=sa))
        txt(f, 0.5, 0.45, "Paradox Earth caught a false trend that lines up with a change",
            size=36, ha="center", alpha=a * ramp(t, 24.0, 24.4))
        txt(f, 0.5, 0.39, "in the model's input data.",
            size=36, ha="center", alpha=a * ramp(t, 24.0, 24.4))
        txt(f, 0.5, 0.27,
            "Soil water looked like a 26-year decline. It was two step changes matching documented\n"
            "GLDAS-2.1 radiation-forcing changes, so the tool withdrew it instead of reporting it.",
            size=23, color=MUTED, ha="center", alpha=a * ramp(t, 24.3, 24.7), linespacing=1.4)

    def s_end(self, t):
        f = self.fig
        a = ramp(t, 25.9, 26.3)
        txt(f, 0.5, 0.84, self.D["meta"]["project"], size=84, weight="bold", ha="center",
            alpha=a)
        txt(f, 0.5, 0.75, "An automated trend detective that grades its own evidence",
            size=32, color=MUTED, ha="center", alpha=a)
        txt(f, 0.5, 0.665, "EVERY DISCOVERY CARD IS GRADED ON", size=18, color=MUTED,
            weight="bold", ha="center", alpha=a)
        checks = ["FDR-corrected significance", "Stability across time windows",
                  "Neighbour-definition agreement", "Independent literature"]
        for k, c in enumerate(checks):
            ca = a * ramp(t, 26.0 + 0.08 * k, 26.25 + 0.08 * k)
            x = 0.0575 + k * 0.225
            pill(f, x, 0.575, 0.21, 0.06, PANEL, alpha=ca)
            txt(f, x + 0.105, 0.605, c, size=18, alpha=ca, ha="center")
        va = a * ramp(t, 26.15, 26.45)
        txt(f, 0.5, 0.48,
            "Next: more NASA variables · any region on demand · a card for every community",
            size=26, color=WET, ha="center", alpha=va)
        la = a * ramp(t, 26.25, 26.55)
        site = self.team.get("site_url", "")
        repo = self.team.get("project_url", "")
        txt(f, 0.5, 0.34, site, size=38, weight="bold", ha="center", alpha=la)
        txt(f, 0.5, 0.27, repo, size=24, color=MUTED, ha="center", alpha=la)
        txt(f, 0.5, 0.16,
            f"{self.team.get('team_name', '')} · {self.team.get('event', '')}",
            size=20, color=MUTED, ha="center", alpha=la)

    # --------------------------------------------------------- frame ---
    def frame(self, t):
        f = self.fig
        f.clf()
        f.patches.clear()
        f.set_facecolor(BG)
        if t < 2.0:
            self.s_intro(t)
        elif t < 5.0:
            self.s_challenge(t)
        elif t < 10.0:
            self.s_title(t)
        elif t < 20.0:
            self.s_split_and_map(t)
        elif t < 23.6:
            self.s_regions(t)
        elif t < 25.9:
            self.s_retraction(t)
        else:
            self.s_end(t)
        self.watermark()
        f.canvas.draw()
        return np.asarray(f.canvas.buffer_rgba())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", nargs="*", type=float,
                    help="render only these times (s) to PNGs in demo/stills/")
    ap.add_argument("--stills-dir", default=os.path.join(ROOT, "demo", "stills"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    t0 = time.time()
    D = load_everything()
    print(f"data loaded in {time.time() - t0:.1f}s; featured: "
          f"{D['wet'] and D['wet']['reg']['id']} / {D['dry'] and D['dry']['reg']['id']}")
    R = Renderer(D)

    if args.stills is not None:
        outdir = args.stills_dir
        os.makedirs(outdir, exist_ok=True)
        for s in args.stills:
            buf = R.frame(s)
            p = os.path.join(outdir, f"still_{s:05.2f}s.png")
            plt.imsave(p, buf)
            print("wrote", p)
        return

    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    n_frames = int(round(DURATION * FPS))
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(n_frames):
        buf = R.frame(k / FPS)
        proc.stdin.write(buf.tobytes())
        if k % 60 == 0:
            print(f"frame {k}/{n_frames}  ({time.time() - t0:.0f}s)", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"wrote {args.out} ({n_frames} frames, {n_frames / FPS:.2f}s) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
