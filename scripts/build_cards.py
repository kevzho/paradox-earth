#!/usr/bin/env python3
"""Build Paradox Earth Discovery Cards (site/data/cards.json).

Reads the exported site data (meta.json, regions.json, tiles.json, series/*.json)
and the human-curated scripts/cards_curation.json, and writes site/data/cards.json.

Every number on a card is computed here from site/data at build time; the curation
file only holds words (place names, user + decision, literature). Curation entries
are matched to regions by trend sign and centroid proximity, so they survive a
re-export that renumbers region ids.

Each card carries a robustness checklist that the detective computes for itself:
  1. FDR-corrected significance  - share of region tiles with q < alpha and matching sign
  2. Stability across time windows - median %/decade re-estimated from the monthly
     anomaly series over each sensitivity window (OLS on annual means) vs. the full period
  3. Neighbour-definition agreement - queen LISA cluster share; kNN agreement if exported
  4. Agreement with an independent precipitation product (CHIRPS via NASA FLDAS)
  5. Agreement with independent literature - from the curated citations

The soil-water retraction card is graded on meta.soil_water (homogeneity, GLDAS-2.0 control, FLDAS, GRACE);
observation cards (kind "observation") report a GRACE/GRACE-FO hotspot from meta.soil_water.grace.

Usage:  python scripts/build_cards.py [--site site] [--curation scripts/cards_curation.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KM_PER_DEG = 111.32
DEFAULT_WINDOWS = {"2003-2025": ["2003-01", "2025-12"], "2000-2020": ["2000-01", "2020-12"]}
STATUSES = ("pass", "partial", "fail", "not_run")


# ----------------------------------------------------------------------------- helpers
def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def num(x, nd=None):
    """JSON-safe float (None for missing / non-finite)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return round(x, nd) if nd is not None else x


def median(values):
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.median(v)) if v else None


def block_key(south: float, west: float) -> str:
    return f"b_{int(math.floor(south / 10) * 10):+04d}_{int(math.floor(west / 10) * 10):+05d}"


def tile_area_km2(south: float, tile_deg: float) -> float:
    lat_c = math.radians(south + tile_deg / 2)
    return (KM_PER_DEG * tile_deg) ** 2 * math.cos(lat_c)


def ang_dist(lat1, lon1, lat2, lon2) -> float:
    """Approximate angular distance in degrees (cos-lat scaled, dateline-aware)."""
    dlon = abs(lon1 - lon2) % 360
    dlon = min(dlon, 360 - dlon) * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(lat1 - lat2, dlon)


def fmt_area(km2) -> str:
    if km2 is None:
        return "unknown area"
    if km2 >= 1e6:
        return f"about {km2 / 1e6:.1f} million km²"
    if km2 >= 1e4:
        return f"about {round(km2, -3):,.0f} km²"
    return f"about {km2:,.0f} km²"


def fmt_pct(x) -> str:
    return "n/a" if x is None else f"{x:+.1f}%"


def year_of(s: str) -> int:
    return int(str(s)[:4])


# ----------------------------------------------------------------------------- data access
class SiteData:
    def __init__(self, site: Path):
        self.site = site
        data = site / "data"
        self.meta = load_json(data / "meta.json")
        self.regions = load_json(data / "regions.json")
        xc = data / "crosscheck.json"
        self.crosscheck = load_json(xc) if xc.exists() else {}
        tiles = load_json(data / "tiles.json")
        cols = tiles["cols"]
        self.tile_deg = float(tiles.get("tile_deg") or self.meta.get("tile_deg") or 1.0)
        self.tiles = {r[cols.index("id")]: dict(zip(cols, r)) for r in tiles["rows"]}
        self.series_dir = data / "series"
        self._blocks: dict[str, dict] = {}
        self._fallback_index: dict[str, str] | None = None
        self.series_start = None
        alpha = (self.meta.get("headline", {}).get("regional", {}) or {}).get("alpha_fdr")
        self.alpha = float(alpha) if alpha is not None else 0.05
        period = (self.meta.get("dataset", {}) or {}).get("period") or ["2000-01", "2025-12"]
        self.period = (str(period[0]), str(period[-1]))
        self.knn_labels, self.knn_source = self._load_knn_labels()

    def _load_knn_labels(self):
        """Per-tile kNN-weights LISA labels from the spatial-statistics output (read-only).

        Looks for <outputs dir>/spatial/<variable>__trend_slope.csv with a `label_knn`
        column. Returns ({tile_id: "HH"/"LL"/...}, path) or ({}, None) if absent."""
        var = (self.meta.get("dataset", {}) or {}).get("variable") or "Rainf_f_tavg"
        code = {"HH_cluster": "HH", "LL_cluster": "LL", "HL_outlier": "HL", "LH_outlier": "LH",
                "inconclusive": "ns", "no_neighbors": "nn"}
        root = self.site.resolve().parent
        # the same LISA field the export used for tiles.json "lisa" (and the regions), so queen and kNN labels refer
        # to the same map; newest run first; the tile-overlap test below rejects a run that does not match
        # use exactly the run this export came from (meta.source_run); fall back to the newest run otherwise
        run = self.meta.get("source_run")
        pattern = f"{run}/spatial" if run else "outputs*/spatial"
        cands = []
        for field in ("trend_slope_pct", "trend_slope"):
            cands = sorted(root.glob(f"{pattern}/{var}__{field}.csv"), key=lambda q: -q.stat().st_mtime)
            if cands:
                break
        import csv
        for path in cands:
            try:
                with open(path, newline="", encoding="utf-8") as f:
                    rd = csv.DictReader(f)
                    if "label_knn" not in (rd.fieldnames or []):
                        continue
                    idc = rd.fieldnames[0]
                    labels = {r[idc]: code.get(r["label_knn"], r["label_knn"]) for r in rd}
                if labels and len(set(labels) & set(self.tiles)) > 0.9 * len(self.tiles):
                    return labels, str(path.relative_to(root))
            except (OSError, KeyError, ValueError):
                continue
        return {}, None

    # series ---------------------------------------------------------------
    def _load_block(self, key: str):
        if key not in self._blocks:
            p = self.series_dir / f"{key}.json"
            self._blocks[key] = load_json(p) if p.exists() else {"tiles": {}}
            if self.series_start is None and self._blocks[key].get("start"):
                self.series_start = self._blocks[key]["start"]
        return self._blocks[key]

    def _build_fallback(self):
        self._fallback_index = {}
        for p in sorted(self.series_dir.glob("*.json")):
            blk = self._load_block(p.stem)
            for tid in blk.get("tiles", {}):
                self._fallback_index[tid] = p.stem

    def series(self, tid: str):
        t = self.tiles.get(tid)
        if t is None:
            return None, None
        blk = self._load_block(block_key(t["south"], t["west"]))
        s = blk.get("tiles", {}).get(tid)
        if s is None:  # block naming changed? scan once
            if self._fallback_index is None:
                self._build_fallback()
            key = self._fallback_index.get(tid)
            if key:
                blk = self._blocks[key]
                s = blk["tiles"].get(tid)
        if s is None:
            return None, None
        return np.array([np.nan if v is None else v for v in s], dtype=float), blk.get("start") or self.period[0]

    # windows ----------------------------------------------------------------
    def windows(self) -> dict[str, tuple[int, int]]:
        out = {}
        wins = (self.meta.get("headline", {}) or {}).get("windows") or {}
        for name, w in wins.items():
            rng = w.get("window") if isinstance(w, dict) else None
            if rng and len(rng) == 2:
                out[name] = (year_of(rng[0]), year_of(rng[1]))
        if not out:
            out = {k: (year_of(a), year_of(b)) for k, (a, b) in DEFAULT_WINDOWS.items()}
        return out


def window_slope_per_decade(anom: np.ndarray, start: str, y0: int, y1: int):
    """OLS slope (mm/month per decade) of annual-mean anomalies for years y0..y1."""
    sy, sm = int(start[:4]), int(start[5:7])
    n = len(anom)
    years = sy + (sm - 1 + np.arange(n)) // 12
    xs, ys = [], []
    for y in range(y0, y1 + 1):
        v = anom[years == y]
        v = v[np.isfinite(v)]
        if len(v) >= 10:
            xs.append(y)
            ys.append(v.mean())
    if len(xs) < 8:
        return None
    return float(np.polyfit(np.array(xs, float), np.array(ys, float), 1)[0] * 10.0)


# ----------------------------------------------------------------------------- kNN lookup
KNN_SHARE_KEYS = ("agreement", "share", "overlap", "jaccard", "agree", "fraction", "share_agree")


def find_knn(obj, path=""):
    """Look for an exported kNN-vs-queen agreement figure anywhere in a dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if "knn" in k.lower():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1:
                    return float(v), p
                if isinstance(v, dict):
                    for kk in KNN_SHARE_KEYS:
                        vv = v.get(kk)
                        if isinstance(vv, (int, float)) and not isinstance(vv, bool) and 0 <= vv <= 1:
                            return float(vv), f"{p}.{kk}"
            r = find_knn(v, p)
            if r:
                return r
    return None


# ----------------------------------------------------------------------------- checks
def check_fdr(tiles, sign, alpha):
    n = len(tiles)
    if n == 0:
        return {"check": "FDR-corrected significance", "status": "not_run", "detail": "No tiles."}, None
    ok = sum(1 for t in tiles if t.get("q") is not None and t["q"] < alpha and t.get("sig") == sign)
    share = ok / n
    status = "pass" if share >= 0.9 else "partial" if share >= 0.5 else "fail"
    detail = (f"Selection criterion (regions are built from these tiles, so this passes by design): {ok} of {n} "
              f"tiles ({share:.0%}) significant with the matching sign after Benjamini-Hochberg false-discovery "
              f"correction (q < {alpha:g}).")
    return {"check": "FDR-corrected significance", "status": status, "detail": detail, "role": "selection"}, share


def check_windows(data: SiteData, tiles, sign):
    """Median %/decade over each sensitivity window vs. the full period, from the series files."""
    wins = data.windows()
    full = (year_of(data.period[0]), year_of(data.period[1]))
    per_win = {name: [] for name in wins}
    full_vals = []
    used = 0
    for t in tiles:
        pct, slope = t.get("pct"), t.get("slope")
        if pct is None or slope is None or abs(pct) < 0.05:
            continue
        mean = slope * 100.0 / pct  # tile mean precipitation (mm/month) implied by the export
        if not math.isfinite(mean) or mean <= 0:
            continue
        anom, start = data.series(t["id"])
        if anom is None:
            continue
        f = window_slope_per_decade(anom, start, *full)
        if f is None:
            continue
        used += 1
        full_vals.append(f / mean * 100)
        for name, (y0, y1) in wins.items():
            w = window_slope_per_decade(anom, start, y0, y1)
            if w is not None:
                per_win[name].append(w / mean * 100)
    name = "Stability across time windows"
    if used == 0:
        return {"check": name, "status": "not_run", "detail": "No monthly series available for these tiles."}, {}
    full_med = median(full_vals)
    meds = {k: median(v) for k, v in per_win.items()}
    statuses = []
    parts = [f"{full[0]}-{full[1]}: {fmt_pct(full_med)}/decade"]
    for k, m in meds.items():
        parts.append(f"{k}: {fmt_pct(m)}/decade")
        if m is None or full_med is None or full_med == 0:
            statuses.append("not_run")
            continue
        ratio = m / full_med
        if ratio <= 0 or (np.sign(m) != sign):
            statuses.append("fail")
        elif 0.5 <= ratio <= 2.0:
            statuses.append("pass")
        else:
            statuses.append("partial")
    if "fail" in statuses:
        status = "fail"
    elif "partial" in statuses:
        status = "partial"
    elif statuses and all(s == "pass" for s in statuses):
        status = "pass"
    elif "pass" in statuses:
        status = "partial"
    else:
        status = "not_run"
    verdict = {"pass": "same sign, within a factor of 2",
               "partial": "same sign but magnitude changes by more than a factor of 2",
               "fail": "sign flips or vanishes in at least one window",
               "not_run": "windows could not be evaluated"}[status]
    detail = (f"Region trend (median over {used} tiles of OLS on annual means; overlapping windows) - "
              + "; ".join(parts) + f" -> {verdict}.")
    gw = (data.meta.get("headline") or {}).get("windows") or {}
    if status == "pass" and gw:
        shares = "; ".join(f"{k}: {100 * v['share_area_up']:.1f}% wetting / {100 * v['share_area_down']:.1f}% drying"
                           for k, v in gw.items() if "share_area_up" in v)
        detail += (f" Global shares of significant land shrink in these windows ({shares}), but this region's trend "
                   "keeps its sign and size.")
    return {"check": name, "status": status, "detail": detail}, {"full": full_med, **meds}


def check_neighbours(tiles, sign, knn, knn_labels=None, knn_source=None):
    want = "HH" if sign > 0 else "LL"
    n = len(tiles)
    name = "Agreement between neighbour definitions (queen vs kNN LISA)"
    if n == 0:
        return {"check": name, "status": "not_run", "detail": "No tiles."}, None
    share = sum(1 for t in tiles if t.get("lisa") == want) / n
    queen = f"Queen-contiguity LISA labels {share:.0%} of tiles as {want} clusters"
    if knn_labels:
        have = [t for t in tiles if t["id"] in knn_labels]
        if have:
            k_share = sum(1 for t in have if knn_labels[t["id"]] == want) / len(have)
            q_tiles = [t for t in have if t.get("lisa") == want]
            agree = (sum(1 for t in q_tiles if knn_labels[t["id"]] == want) / len(q_tiles)) if q_tiles else 0.0
            if share >= 0.8 and agree >= 0.8:
                status = "pass"
            elif share >= 0.5 and agree >= 0.5:
                status = "partial"
            else:
                status = "fail"
            return {"check": name, "status": status,
                    "detail": (f"{queen}; k-nearest-neighbour weights label {k_share:.0%} as {want}, "
                               f"and {agree:.0%} of the queen {want} tiles keep their label under kNN.")}, share
    if knn is None:
        return {"check": name, "status": "not_run",
                "detail": f"{queen}; the kNN-weights comparison is not in this export, so agreement is not graded."}, share
    val, path = knn
    if share >= 0.9 and val >= 0.7:
        status = "pass"
    elif val < 0.3 or share < 0.5:
        status = "fail"
    else:
        status = "partial"
    return {"check": name, "status": status,
            "detail": f"{queen}; kNN weights agree on {val:.0%} of clustered tiles ({path})."}, share


def check_independent_product(data, region_id, sign):
    """Same region, independent precipitation product: CHIRPS (via NASA FLDAS) over the common 2003-2024 window."""
    name = "Agreement with an independent precipitation product (CHIRPS, via NASA FLDAS)"
    x = (data.crosscheck.get("regions") or {}).get(region_id)
    if not x:
        return {"check": name, "status": "not_run", "detail": "No CHIRPS/FLDAS comparison for this region."}
    same, sig = x["fldas_share_tiles_same_sign"], x["fldas_share_tiles_raw_p05_same_sign"]
    status = "pass" if same >= 0.9 and sig >= 0.5 else "partial" if same >= 0.7 else "fail"
    word = "wetting" if sign > 0 else "drying"
    detail = (f"Over 2003-2024, CHIRPS-based precipitation shows {word} in {same:.0%} of this region's {x['n_tiles']} "
              f"tiles ({sig:.0%} individually significant at p < 0.05); median trend "
              f"{x['fldas_chirps_2003_2024_median_slope']:+.1f} mm/month per decade vs "
              f"{x['gldas_2003_2024_median_slope']:+.1f} in GLDAS over the same years.")
    return {"check": name, "status": status, "detail": detail}


def check_literature(lit):
    name = "Agreement with independent literature"
    if not lit:
        return {"check": name, "status": "not_run", "detail": "No curated peer-reviewed source yet."}
    yes = sum(1 for x in lit if x.get("agrees") is True)
    part = sum(1 for x in lit if x.get("agrees") == "partly")
    no = sum(1 for x in lit if x.get("agrees") is False)
    if yes and not no:
        status = "pass"
    elif yes or part:
        status = "partial"
    else:
        status = "fail"
    detail = f"{len(lit)} source(s): {yes} agree, {part} partly agree, {no} disagree."
    return {"check": name, "status": status, "detail": detail}


INDEPENDENT = ("Stability across time windows", "Agreement between neighbour definitions (queen vs kNN LISA)",
               "Agreement with an independent precipitation product (CHIRPS, via NASA FLDAS)",
               "Agreement with independent literature")


def evidence_level(checks):
    """Grade on the INDEPENDENT checks only; FDR significance (and the queen cluster label) are how regions are
    selected, so they cannot count as corroboration."""
    st = {c["check"]: c["status"] for c in checks}
    if st.get("FDR-corrected significance") != "pass":
        return "observed pattern"
    if st.get("Stability across time windows") == "fail":
        return "statistically supported (window-sensitive)"
    ran = [k for k in INDEPENDENT if k in st and st[k] != "not_run"]
    n_pass = sum(st.get(k) == "pass" for k in ran)
    if ran and n_pass == len(ran):
        return f"robust pattern: passes all {len(ran)} extra checks (windows, neighbours, independent data, literature)"
    return f"supported: passes {n_pass} of {len(ran)} extra checks"


# ----------------------------------------------------------------------------- cards
def literature_out(lit):
    return [{"citation": x["citation"], "url_or_doi": x["url_or_doi"], "agrees": x["agrees"],
             **({"note": x["note"]} if x.get("note") else {})} for x in lit]


def build_stats(data: SiteData, tiles, sign, region=None, knn=None):
    region = region or {}
    n = len(tiles)
    geo_area = sum(tile_area_km2(t["south"], data.tile_deg) for t in tiles)
    area = region.get("area_km2")
    area_note = None
    if area is None or not (0.2 * geo_area <= area <= 1.5 * geo_area):
        if area is not None:
            area_note = ("Area shown is estimated from 1-degree tile geometry; the exported region area "
                         "looked like a placeholder.")
        area = geo_area
    med_pct = region.get("median_slope_pct_per_decade")
    if med_pct is None:
        med_pct = median(t.get("pct") for t in tiles)
    med_slope = region.get("median_slope_per_decade")
    if med_slope is None:
        med_slope = median(t.get("slope") for t in tiles)
    fdr, share = check_fdr(tiles, sign, data.alpha)
    win, win_meds = check_windows(data, tiles, sign)
    rk = region.get("knn_agreement")
    knn_here = (float(rk), "regions.json knn_agreement") if isinstance(rk, (int, float)) else knn
    nb, _ = check_neighbours(tiles, sign, knn_here, data.knn_labels, data.knn_source)
    numbers = {
        "median_pct_per_decade": num(med_pct, 2),
        "median_slope_per_decade": num(med_slope, 3),
        "area_km2": num(area, 0),
        "n_tiles": n,
        "share_tiles_fdr_significant": num(share, 3),
    }
    return numbers, [fdr, win, nb], area_note


def headline_for(numbers, place, sign, period, tile_deg, detectable=True):
    word = "wetting" if sign > 0 else "drying"
    share = numbers["share_tiles_fdr_significant"]
    share_txt = f"{share:.0%}" if share is not None else "n/a"
    slope = numbers["median_slope_per_decade"]
    slope_txt = f"{slope:+.2f} mm/month" if slope is not None else "n/a"
    deg = "one-degree" if tile_deg == 1 else f"{tile_deg:g}-degree"
    what = f"a detectable {word} trend" if detectable else f"a {word} tendency (not a robust trend)"
    return (f"Across {numbers['n_tiles']} {deg} land tiles ({fmt_area(numbers['area_km2'])}) in {place}, "
            f"GLDAS-2.1 precipitation shows {what} of "
            f"{fmt_pct(numbers['median_pct_per_decade'])} per decade ({slope_txt} per decade, median tile, "
            f"{period[0][:4]}-{period[1][:4]}), with {share_txt} of tiles significant after "
            f"false-discovery correction.")


def match_regions(regions_by_sign, entries, default_dist):
    """One-to-one greedy assignment of curation entries to regions by centroid distance."""
    pairs = []
    for e in entries:
        sign_key = "up" if e["sign"] == "wetting" else "down"
        m = e["match"]
        maxd = float(m.get("max_dist_deg", default_dist))
        for r in regions_by_sign.get(sign_key, []):
            if r.get("centroid_lat") is None or r.get("centroid_lon") is None:
                continue
            d = ang_dist(m["centroid_lat"], m["centroid_lon"], r["centroid_lat"], r["centroid_lon"])
            if d <= maxd:
                pairs.append((d, e["key"], r["id"]))
    pairs.sort()
    used_e, used_r, out = set(), set(), {}
    for d, ek, rid in pairs:
        if ek in used_e or rid in used_r:
            continue
        used_e.add(ek)
        used_r.add(rid)
        out[ek] = (rid, d)
    return out


def region_card(data, entry, region, knn, global_caveats):
    sign = 1 if entry["sign"] == "wetting" else -1
    ids = region.get("tiles") or [tid for tid, t in data.tiles.items() if t.get("region") == region["id"]]
    tiles = [data.tiles[i] for i in ids if i in data.tiles]
    numbers, checks, area_note = build_stats(data, tiles, sign, region, knn)
    checks.append(check_independent_product(data, region.get("id"), sign))
    checks.append(check_literature(entry.get("literature", [])))
    caveats = list(entry.get("caveats", [])) + ([area_note] if area_note else []) + global_caveats
    return {
        "id": f"{entry['sign']}-{entry['key']}",
        "kind": entry["sign"],
        "region_id": region["id"],
        "title": entry["title"],
        "place": entry["place"],
        "headline": headline_for(numbers, entry["place"], sign, data.period, data.tile_deg),
        "numbers": numbers,
        "robustness": checks,
        "user": entry["user"],
        "evidence_level": evidence_level(checks),
        "literature": literature_out(entry.get("literature", [])),
        "caveats": caveats,
        "centroid": [num(region.get("centroid_lat"), 2), num(region.get("centroid_lon"), 2)],
        "bbox": region.get("bbox"),
    }


BREAK_RE = re.compile(r"\b((?:19|20)\d{2}(?:[-/](?:\d{2}|\d{4}))?)\b")


def _fmt_ci(ci, nd=1, unit=""):
    return f"{ci[0]:+.{nd}f} to {ci[1]:+.{nd}f}{unit}" if ci and len(ci) == 2 else "n/a"


def retraction_card(data: SiteData, cur):
    r = data.meta.get("retraction")
    if isinstance(r, dict):
        text = r.get("text") or r.get("summary") or r.get("note") or json.dumps(r)
        breaks = r.get("breakpoints") or r.get("breaks") or BREAK_RE.findall(text)
    else:
        text = str(r or "")
        breaks = BREAK_RE.findall(text)
    breaks = [str(b) for b in breaks]
    breaks_txt = " and ".join(breaks) if breaks else "a documented forcing change"
    sw = data.meta.get("soil_water") or {}
    ac, fl, gr = sw.get("artifact_control") or {}, sw.get("fldas") or {}, sw.get("grace") or {}
    doc_lit = cur.get("literature", [])

    homog = {"check": "Homogeneity (break-point) check", "status": "fail",
             "detail": (f"Level shift detected around {breaks_txt}. " if breaks else "Level shift detected. ") + text
             + (" README_GLDAS2 documents an apparent shift in the AGRMET radiation climatology (section 1.5.3) and "
                "degraded WWMCA radiation from mid-2024 (section 1.2.3); the detected steps line up with both."
                if doc_lit else "")}

    # Same model, different forcing (GLDAS-2.0 control)
    name = "Same model, different forcing (GLDAS-2.0 control)"
    h = ac.get("headline") or {}
    if ac.get("swdown_step_21_W_m2") is not None:
        cw = h.get("common_window") or ["?", "?"]
        detail = (f"{ac.get('what', 'GLDAS-2.1 vs GLDAS-2.0')}. At {ac.get('step_date', '?')} incoming solar radiation "
                  f"in the 2.1 forcing steps by {ac['swdown_step_21_W_m2']:+.1f} W/m² (95% CI "
                  f"{_fmt_ci(ac.get('swdown_step_21_ci95'))}); in 2.0 the step is {ac['swdown_step_20_W_m2']:+.1f} W/m² "
                  f"(95% CI {_fmt_ci(ac.get('swdown_step_20_ci95'))}). " + (ac.get("reading") or ""))
        if h.get("soil_trend_2.1_common_mm_per_decade") is not None and h.get("soil_trend_2.0_common_mm_per_decade") is not None:
            detail += (f" Soil-water trend over {cw[0][:4]}-{cw[-1][:4]}: {h['soil_trend_2.1_common_mm_per_decade']:+.1f} "
                       f"mm/decade in 2.1 (p = {h['soil_trend_2.1_common_p']:.2f}) vs "
                       f"{h['soil_trend_2.0_common_mm_per_decade']:+.1f} in 2.0 (p = {h['soil_trend_2.0_common_p']:.2f}).")
        control = {"check": name, "status": "fail", "detail": detail}
    else:
        control = {"check": name, "status": "not_run", "detail": "GLDAS-2.0 control not in this export."}

    # Consistently forced product (FLDAS)
    fw = fl.get("window") or ["?", "?"]
    name = f"Consistently forced product (NASA FLDAS, {str(fw[0])[:4]}-{str(fw[-1])[:4]})"
    if fl.get("aggregate_slope_mm_per_decade") is not None:
        fp = fl.get("aggregate_p")
        ci = fl.get("aggregate_ci95")
        detectable = fp is not None and fp < 0.05 and fl["aggregate_slope_mm_per_decade"] < 0
        detail = (f"FLDAS (Noah driven by MERRA-2 + CHIRPS, no radiation-forcing switch): global land soil water "
                  f"{fl['aggregate_slope_mm_per_decade']:+.1f} mm/decade (95% CI {_fmt_ci(ci)}, p = {fp:.2f}) - "
                  + ("a detectable decline." if detectable else "no detectable global trend, and no decline.")
                  + f" After FDR correction {100 * fl.get('share_area_wetting', 0):.1f}% of land is detectably "
                  f"wetting and {100 * fl.get('share_area_drying', 0):.1f}% drying.")
        fl_check = {"check": name, "status": "pass" if detectable else "fail", "detail": detail}
    else:
        fl_check = {"check": name, "status": "not_run", "detail": "FLDAS soil water not in this export."}

    # Observations (GRACE)
    name = "Observations (GRACE/GRACE-FO)"
    vs = gr.get("vs_fldas_soil_water") or {}
    if vs.get("sign_match_area_sig_either") is not None:
        gw = gr.get("window") or ["?", "?"]
        hot = vs.get("hotspots") or {}
        hot_txt = "; ".join(f"{k} GRACE {v['grace_mean_slope_mm_decade']:+.0f} vs model "
                            f"{v['model_mean_slope_mm_decade']:+.0f} mm/decade"
                            for k, v in list(hot.items())[:3])
        detail = (f"{gr.get('product', 'GRACE mascons')}, {gw[0][:4]}-{gw[-1][:4]}: FDR-significant water-storage trends "
                  f"over {100 * gr.get('land_area_share_fdr_significant', 0):.0f}% of land. GRACE weighs all water "
                  f"(groundwater, lakes, ice), so it cannot confirm or refute a soil-only trend. Where either is "
                  f"significant, GRACE and FLDAS soil water agree in sign over {100 * vs['sign_match_area_sig_either']:.0f}% "
                  f"of the area (Spearman {vs.get('spearman_slopes', float('nan')):.2f}). The largest GRACE declines "
                  f"are groundwater and irrigation hotspots the land model does not represent ({hot_txt}).")
        gr_check = {"check": name, "status": "partial", "detail": detail}
    else:
        gr_check = {"check": name, "status": "not_run", "detail": "GRACE comparison not in this export."}

    fmt = {"breaks": breaks_txt, "retraction_text": text,
           "fldas_window": f"{str(fw[0])[:4]}-{str(fw[-1])[:4]}",
           "fldas_slope": f"{fl['aggregate_slope_mm_per_decade']:+.1f}" if fl.get("aggregate_slope_mm_per_decade") is not None else "n/a",
           "fldas_p": f"{fl['aggregate_p']:.2f}" if fl.get("aggregate_p") is not None else "n/a",
           "fldas_verdict": ("a detectable decline" if fl_check["status"] == "pass" else "no detectable trend"
                             if fl_check["status"] == "fail" else "not evaluated")}
    headline = cur.get("headline_template", "{breaks}").format(**fmt)
    numbers = {"median_pct_per_decade": None, "median_slope_per_decade": None, "area_km2": None,
               "n_tiles": None, "share_tiles_fdr_significant": None,
               "fldas_soil_water_mm_per_decade": num(fl.get("aggregate_slope_mm_per_decade"), 2),
               "fldas_soil_water_p": num(fl.get("aggregate_p"), 3),
               "swdown_step_gldas21_W_m2": num(ac.get("swdown_step_21_W_m2"), 2),
               "swdown_step_gldas20_W_m2": num(ac.get("swdown_step_20_W_m2"), 2)}
    return {
        "id": "retraction-soil-water",
        "kind": "retraction",
        "region_id": None,
        "title": cur.get("title", "Paradox Earth caught a false trend"),
        "place": cur.get("place", "Global land"),
        "headline": headline,
        "numbers": numbers,
        "robustness": [homog, control, fl_check, gr_check],
        "user": cur["user"],
        "evidence_level": "withdrawn (data artefact caught by the detective)",
        "literature": literature_out(doc_lit),
        "caveats": list(cur.get("caveats", [])),
        "breakpoints": breaks,
        **({"figure": ac["figure"]} if ac.get("figure") else {}),
    }


# ----------------------------------------------------------------------------- observation cards (GRACE)
def grace_hotspot_boxes() -> dict:
    """DEPLETION_HOTSPOTS from scripts/grace_check.py, parsed (not imported) so this script needs no GRACE deps.
    Returns {name: (lat_min, lat_max, lon_min, lon_max)}."""
    import ast
    src = (ROOT / "scripts" / "grace_check.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DEPLETION_HOTSPOTS" for t in node.targets):
            return ast.literal_eval(node.value)
    return {}


def grace_box_rows(hotspot: str, window) -> list[dict]:
    """Per-box GRACE vs FLDAS rows for one hotspot from outputs_grace (optional enrichment; [] if absent)."""
    import csv
    tag = f"{str(window[0]).replace('-', '')[:6]}_{str(window[-1]).replace('-', '')[:6]}"
    cands = sorted((ROOT / "outputs_grace").glob(f"compare_outputs_fldas_soil_water_0_200cm_{tag}_boxes.csv"))
    if not cands:
        return []
    with open(cands[0], newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if hotspot in (r.get("hotspot") or "").split(";")]


def observation_card(data: SiteData, cur):
    gr = (data.meta.get("soil_water") or {}).get("grace") or {}
    vs = gr.get("vs_fldas_soil_water") or {}
    key = cur["hotspot"]
    h = (vs.get("hotspots") or {}).get(key)
    if not h:
        return None
    window = vs.get("window") or gr.get("window") or ["?", "?"]
    yrs = f"{str(window[0])[:4]}-{str(window[-1])[:4]}"
    g, m = h["grace_mean_slope_mm_decade"], h["model_mean_slope_mm_decade"]
    n_sig, n_box = int(h["n_grace_sig"]), int(h["n_boxes"])
    hb = grace_hotspot_boxes().get(key)
    bbox = [hb[2], hb[0], hb[3], hb[1]] if hb else None  # [west, south, east, north]
    rows = grace_box_rows(key, window)
    sig_rows = [r for r in rows if r.get("grace_sig_fdr") == "True"]
    n_sig_neg = sum(float(r["grace_slope_mm_decade"]) < 0 for r in sig_rows)
    n_glacier = sum(bool(r.get("glacier_influenced")) for r in sig_rows)
    n_model_pos = sum(float(r["model_slope_mm_decade"]) > 0 for r in rows)
    n_model_sig = sum(r.get("model_sig_fdr") == "True" for r in rows)
    area = sum(float(r["model_land_area_km2"]) for r in rows) if rows else None

    share = n_sig / n_box if n_box else 0.0
    status = "pass" if share >= 0.9 else "partial" if share >= 0.5 else "fail"
    detail = (f"{n_sig} of {n_box} 3-degree boxes ({share:.0%}) have GRACE/GRACE-FO water-storage trends that are "
              f"significant after Benjamini-Hochberg FDR correction, {yrs}")
    if rows:
        flags = sorted({r["glacier_influenced"] for r in sig_rows if r.get("glacier_influenced")})
        ns_lats = sorted({float(r["lat"]) for r in rows if r.get("grace_sig_fdr") != "True"})
        detail += (f"; {n_sig_neg} of those {len(sig_rows)} are declines. {n_glacier} of the significant boxes lie in "
                   f"the glacier-influenced zone ({', '.join(flags) or 'none'}), so part of their signal may be "
                   f"glacier mass loss leaking into the mascons rather than groundwater. Not significant: "
                   f"{n_box - n_sig} boxes, centred at {', '.join(f'{x:g}' for x in ns_lats)}°N.")
    else:
        detail += "."
    c_sig = {"check": "GRACE significance (FDR, per 3-degree box)", "status": status, "detail": detail}

    opposite = (g < 0 < m) or (m < 0 < g)
    d_status = "pass" if opposite and abs(g) >= 3 * abs(m) else "partial" if abs(g) >= 3 * abs(m) else "fail"
    detail = (f"Box mean: GRACE {g:+.0f} mm/decade vs NASA FLDAS soil water (0-200 cm) {m:+.0f} mm/decade - "
              + ("opposite signs." if opposite else "same sign.")
              + (f" The model shows a rise in {n_model_pos} of {len(rows)} boxes ({n_model_sig} FDR-significant)."
                 if rows else "")
              + " The land model has no groundwater pumping or irrigation, so a decline GRACE sees and the soil model "
              "does not points to storage the model does not track (groundwater, or glacier ice near the Himalaya). Globally, GRACE and FLDAS soil water agree in sign over "
              f"{100 * vs.get('sign_match_area_sig_either', float('nan')):.0f}% of the area where either is significant.")
    c_model = {"check": "Disagreement with the land model (NASA FLDAS soil water)", "status": d_status, "detail": detail}
    c_lit = check_literature(cur.get("literature", []))
    c_second = {"check": "Second GRACE solution (e.g. CSR or GSFC mascons)", "status": "not_run",
                "detail": f"Only the {gr.get('product', 'JPL mascon')} solution was analysed."}
    fmt = {"years": yrs, "grace": f"{g:+.0f}", "model": f"{m:+.0f}", "grace_abs": f"{abs(g):.0f}",
           "n_sig": n_sig, "n_boxes": n_box}
    return {
        "id": f"observation-{cur['key']}",
        "kind": "observation",
        "region_id": None,
        "title": cur["title"],
        "place": cur["place"],
        "headline": cur["headline_template"].format(**fmt),
        "numbers": {"median_pct_per_decade": None, "median_slope_per_decade": None, "area_km2": num(area, 0),
                    "n_tiles": None, "share_tiles_fdr_significant": num(share, 3),
                    "grace_mean_slope_mm_per_decade": num(g, 1), "model_mean_slope_mm_per_decade": num(m, 1),
                    "n_boxes": n_box, "n_boxes_grace_fdr_significant": n_sig, "box_deg": 3},
        "robustness": [c_sig, c_model, c_lit, c_second],
        "user": cur["user"],
        "evidence_level": "observed (satellite gravity); cause not identified by this analysis",
        "literature": literature_out(cur.get("literature", [])),
        "caveats": list(cur.get("caveats", [])),
        "centroid": cur.get("centroid"),
        "bbox": bbox,
        "source": {"grace_product": gr.get("product"), "window": window, "model": "NASA FLDAS soil water 0-200 cm"},
    }


def local_card(data: SiteData, cur, knn, global_caveats):
    bbox = cur.get("bbox")
    base = {
        "id": "local-placeholder" if not bbox else "local-region",
        "kind": "local",
        "region_id": None,
        "title": cur.get("title", "TODO: your local region"),
        "place": cur.get("place", "TODO"),
        "user": cur["user"],
        "literature": literature_out(cur.get("literature", [])),
        "todo": bool(cur.get("todo", True)),
        "bbox": bbox,
    }
    if not bbox:
        base.update({
            "headline": "TODO: choose the team's local region; this card fills itself from the data once a bounding box is set.",
            "numbers": {"median_pct_per_decade": None, "median_slope_per_decade": None, "area_km2": None,
                        "n_tiles": None, "share_tiles_fdr_significant": None},
            "robustness": [
                {"check": c, "status": "not_run", "detail": "Waiting for the team to choose a local region."}
                for c in ("FDR-corrected significance", "Stability across time windows",
                          "Agreement between neighbour definitions (queen vs kNN LISA)",
                          "Agreement with an independent precipitation product (CHIRPS, via NASA FLDAS)",
                          "Agreement with independent literature")],
            "evidence_level": "not yet evaluated",
            "caveats": list(cur.get("caveats", [])),
        })
        return base
    w, s, e, n = map(float, bbox)
    tiles = [t for t in data.tiles.values()
             if w <= t["west"] + data.tile_deg / 2 <= e and s <= t["south"] + data.tile_deg / 2 <= n]
    med = median(t.get("slope") for t in tiles)
    sign = 1 if (med or 0) >= 0 else -1
    numbers, checks, area_note = build_stats(data, tiles, sign, None, knn)
    checks.append(check_independent_product(data, None, sign))
    checks.append(check_literature(cur.get("literature", [])))
    base.update({
        "headline": headline_for(numbers, cur.get("place", "the local region"), sign, data.period, data.tile_deg,
                                 detectable=checks[0]["status"] == "pass"),
        "numbers": numbers,
        "robustness": checks,
        "evidence_level": evidence_level(checks),
        "caveats": list(cur.get("caveats", [])) + ([area_note] if area_note else []) + global_caveats,
    })
    return base


# ----------------------------------------------------------------------------- validation
def validate(doc):
    errs = []
    req = ["id", "kind", "region_id", "title", "place", "headline", "numbers", "robustness", "user",
           "evidence_level", "literature", "caveats"]
    nkeys = ["median_pct_per_decade", "median_slope_per_decade", "area_km2", "n_tiles",
             "share_tiles_fdr_significant"]
    if not isinstance(doc.get("cards"), list) or not doc["cards"]:
        return ["cards must be a non-empty list"]
    ids = set()
    for c in doc["cards"]:
        cid = c.get("id", "?")
        for k in req:
            if k not in c:
                errs.append(f"{cid}: missing {k}")
        if cid in ids:
            errs.append(f"duplicate id {cid}")
        ids.add(cid)
        if c.get("kind") not in ("wetting", "drying", "retraction", "local", "observation"):
            errs.append(f"{cid}: bad kind")
        if c.get("kind") == "observation":
            if not c["literature"]:
                errs.append(f"{cid}: no literature")
            for k in ("centroid", "bbox"):
                v = c.get(k)
                if not (isinstance(v, list) and len(v) == (2 if k == "centroid" else 4)):
                    errs.append(f"{cid}: {k} malformed")
            if " TWS" in c.get("headline", "") or "terrestrial water storage" in c.get("headline", "").lower():
                errs.append(f"{cid}: use 'water storage' wording, not TWS, in the headline")
        for k in nkeys:
            if k not in c.get("numbers", {}):
                errs.append(f"{cid}: numbers.{k} missing")
        for r in c.get("robustness", []):
            if set(("check", "status", "detail")) - set(r):
                errs.append(f"{cid}: robustness item missing keys")
            if r.get("status") not in STATUSES:
                errs.append(f"{cid}: bad status {r.get('status')}")
        for k in ("who", "decision", "why_it_matters"):
            if not c.get("user", {}).get(k):
                errs.append(f"{cid}: user.{k} empty")
        for l in c.get("literature", []):
            if set(("citation", "url_or_doi", "agrees")) - set(l):
                errs.append(f"{cid}: literature item missing keys")
            if l.get("agrees") not in (True, False, "partly"):
                errs.append(f"{cid}: bad agrees value")
        if c.get("kind") in ("wetting", "drying", "observation"):
            if c.get("kind") != "observation" and not c["literature"]:
                errs.append(f"{cid}: no literature")
            for bad in (" causes ", " caused by ", " due to ", " because of "):
                if bad in c.get("headline", ""):
                    errs.append(f"{cid}: causal wording in headline ({bad.strip()})")
    try:
        json.dumps(doc, allow_nan=False)
    except ValueError as e:
        errs.append(f"non-finite number: {e}")
    return errs


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", default=str(ROOT / "site"))
    ap.add_argument("--curation", default=str(ROOT / "scripts" / "cards_curation.json"))
    ap.add_argument("--out", default=None, help="default: <site>/data/cards.json")
    args = ap.parse_args(argv)

    data = SiteData(Path(args.site))
    cur = load_json(Path(args.curation))
    sel = cur.get("selection", {})
    default_dist = float(sel.get("default_max_dist_deg", 10.0))
    knn = find_knn(data.meta)
    global_caveats = list(data.meta.get("caveats", []))

    regions_by_id = {r["id"]: r for k in ("up", "down") for r in data.regions.get(k, [])}
    matches = match_regions(data.regions, cur["regions"], default_dist)

    cards, log = [], []
    for sign, cap in (("wetting", int(sel.get("max_wetting", 3))), ("drying", int(sel.get("max_drying", 3)))):
        entries = sorted((e for e in cur["regions"] if e["sign"] == sign), key=lambda e: e.get("priority", 99))
        taken = 0
        for e in entries:
            if e["key"] not in matches:
                log.append(f"  skip {e['key']}: no {sign} region within {e['match'].get('max_dist_deg', default_dist)} deg")
                continue
            if taken >= cap:
                log.append(f"  reserve {e['key']} -> {matches[e['key']][0]} (cap {cap} reached)")
                continue
            rid, d = matches[e["key"]]
            cards.append(region_card(data, e, regions_by_id[rid], knn, global_caveats))
            log.append(f"  card {e['key']} -> {rid} (centroid distance {d:.1f} deg)")
            taken += 1
    for k in ("up", "down"):
        for r in data.regions.get(k, []):
            if r["id"] not in {rid for rid, _ in matches.values()}:
                log.append(f"  uncurated region {r['id']} at ({r.get('centroid_lat'):.1f}, {r.get('centroid_lon'):.1f})")

    if data.meta.get("retraction"):
        cards.append(retraction_card(data, cur.get("retraction", {})))
    for o in cur.get("observations", []):
        oc = observation_card(data, o)
        if oc:
            cards.append(oc)
        else:
            log.append(f"  skip observation {o.get('key')}: hotspot {o.get('hotspot')!r} not in meta.soil_water.grace")
    cards.append(local_card(data, cur.get("local", {}), knn, global_caveats))

    doc = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/build_cards.py",
        "dataset": {k: data.meta.get("dataset", {}).get(k) for k in ("short_name", "version", "doi", "variable", "period")},
        "status_legend": {"pass": "check supports the trend", "partial": "mixed or weaker support",
                          "fail": "check contradicts the trend", "not_run": "not evaluated in this export"},
        "cards": cards,
    }
    errs = validate(doc)
    if errs:
        print("cards.json validation FAILED:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else Path(args.site) / "data" / "cards.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print("\n".join(log))
    print(f"kNN per-tile labels: {data.knn_source or 'not found'}; "
          f"kNN summary figure: {'found at ' + knn[1] if knn else 'not in export'}")
    for c in cards:
        st = " ".join(f"{r['status']}" for r in c["robustness"])
        print(f"  [{c['kind']:10s}] {c['id']:32s} {c['evidence_level']:45s} checks: {st}")
    print(f"wrote {out} ({len(cards)} cards)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
