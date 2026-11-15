"""Flagship summary (§11 evidence orchestration, §13 claims): the aggregate-vs-regional contrast for one variable.

For a variable it reports, from an existing run's outputs:
  aggregate   land-area-weighted mean series and its trend (what "the average" says)
  regional    per-tile trends after Benjamini-Hochberg FDR across tiles, as shares of tiles AND of land area
  regions     connected groups of tiles that are both spatial clusters (LISA HH/LL on the chosen field) and
              individually FDR-significant with the matching sign, largest first
  windows     the same aggregate trend and FDR shares for alternative analysis windows (sensitivity, §12 Exp. B)
Everything is structured numbers with method strings, so Discovery Card text can be generated without free-text claims.
Levels (§13): observed pattern / estimated relationship / statistically supported; never causal.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from . import spatial, trend
from .contract import write_json

WINDOWS = {"full": None, "2003-2025": ("2003-01", "2025-12"), "2000-2020": ("2000-01", "2020-12")}


def tile_weights(tiles: pd.DataFrame, var: str, ids) -> tuple[np.ndarray, str]:
    """Land-area weights for `ids`; falls back to cos(latitude) for runs made before areas were stored."""
    t = tiles.set_index("region_id").loc[list(ids)]
    col = f"area_km2__{var}"
    if col in t:
        return t[col].to_numpy(float), f"true land area of valid cells ({col})"
    lat = np.deg2rad((t["south"] + t["north"]) / 2)
    return np.cos(lat).to_numpy(), "cos(latitude) per tile (land fraction unknown)"


def land_aggregate(df: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Weighted mean across tiles per month, renormalised over tiles present that month (missing stays missing)."""
    x = df.to_numpy(float)
    ok = np.isfinite(x)
    num = np.where(ok, x, 0.0) @ w
    den = ok @ w
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=df.index)


def _trend_job(job):
    y, units = job
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = trend.seasonal_trend(y, units)
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "reason": f"{type(e).__name__}: {e}"}
    return {k: r.get(k) for k in ("status", "slope", "p_value")}


def fdr_shares(slope: pd.Series, p: pd.Series, area: pd.Series, alpha: float = 0.05) -> dict:
    """BH across tiles; shares of tested tiles and of their land area that are significantly up / down."""
    ok = slope.notna() & p.notna()
    q = pd.Series(np.nan, index=p.index)
    q[ok] = multipletests(p[ok], alpha=alpha, method="fdr_bh")[1]
    up, down = ok & (q < alpha) & (slope > 0), ok & (q < alpha) & (slope < 0)
    a = area[ok].sum()
    return {"n_tiles_tested": int(ok.sum()), "land_area_tested_km2": float(a), "alpha_fdr": alpha,
            "share_tiles_up": float(up.sum() / ok.sum()), "share_tiles_down": float(down.sum() / ok.sum()),
            "share_area_up": float(area[up].sum() / a), "share_area_down": float(area[down].sum() / a),
            "area_up_km2": float(area[up].sum()), "area_down_km2": float(area[down].sum()),
            "share_tiles_raw_p_below_alpha": float((p[ok] < alpha).mean()), "_q": q}


def connected_regions(ids, tiles: pd.DataFrame) -> list[list[str]]:
    """8-neighbour connected components (dateline-aware) among `ids`."""
    ids = list(ids)
    if not ids:
        return []
    nb = spatial.lattice_neighbors(tiles, ids)
    seen, comps = set(), []
    for start in ids:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            r = stack.pop()
            comp.append(r)
            for n in nb[r]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        comps.append(sorted(comp))
    return sorted(comps, key=len, reverse=True)


def region_summary(comp, tiles, area, slope, slope_pct) -> dict:
    t = tiles.set_index("region_id").loc[comp]
    a = area.loc[comp]
    lat_c, lon_c = (t["south"] + t["north"]) / 2, (t["west"] + t["east"]) / 2
    return {"n_tiles": len(comp), "area_km2": float(a.sum()),
            "centroid_lat": float(np.average(lat_c, weights=a)), "centroid_lon": float(np.average(lon_c, weights=a)),
            "bbox": [float(t["west"].min()), float(t["south"].min()), float(t["east"].max()), float(t["north"].max())],
            "median_slope_per_decade": float(slope.loc[comp].median()),
            "median_slope_pct_per_decade": float(slope_pct.loc[comp].median()),
            "tiles": comp}


def build(out_dir="outputs", var="Rainf_f_tavg", field="trend_slope_pct", alpha=0.05, windows=None,
          workers=1, top=8, min_region_tiles=5) -> dict:
    out_dir = Path(out_dir)
    tiles = pd.read_csv(out_dir / "tiles.csv")
    df = pd.read_csv(out_dir / f"regional_{var}.csv", index_col=0, parse_dates=True)
    stats = pd.read_csv(out_dir / "stats_summary.csv")
    stats = stats[stats["variable"] == var].set_index("region_id")
    w, wdesc = tile_weights(tiles, var, df.columns)
    area = pd.Series(w, index=df.columns)

    agg = land_aggregate(df, w)
    agg_trend = trend.seasonal_trend(agg)
    agg_mean = float(agg.mean())
    fdr = fdr_shares(stats["slope_per_decade"].reindex(df.columns), stats["trend_p"].reindex(df.columns), area, alpha)
    q = fdr.pop("_q")

    out = {"variable": var, "weights": wdesc,
           "aggregate": {**{k: agg_trend.get(k) for k in ("slope", "ci95", "p_value", "n", "window", "method")},
                         "mean": agg_mean,
                         "slope_pct_per_decade": 100 * agg_trend["slope"] / agg_mean if agg_mean > 0 else None},
           "regional": fdr, "regions": {}, "windows": {}}

    lab_path = out_dir / "spatial" / f"{var}__{field}.csv"
    if lab_path.exists():
        lab = pd.read_csv(lab_path, index_col=0)["label"].reindex(df.columns)
        slope = stats["slope_per_decade"].reindex(df.columns)
        slope_pct = stats.get("slope_pct_per_decade", slope * np.nan).reindex(df.columns)
        sig = q < alpha
        for kind, label, sign in (("up", "HH_cluster", 1), ("down", "LL_cluster", -1)):
            ids = df.columns[(lab == label) & sig & (np.sign(slope) == sign)]
            comps = [c for c in connected_regions(ids, tiles) if len(c) >= min_region_tiles][:top]
            out["regions"][kind] = [region_summary(c, tiles, area, slope, slope_pct) for c in comps]
        out["regions"]["rule"] = (f"connected (8-neighbour, dateline-aware) tiles labelled {field} HH/LL by LISA AND "
                                  f"BH-significant (q<{alpha}) with the matching sign; >= {min_region_tiles} tiles")

    for name, win in (windows or WINDOWS).items():
        if win is None:
            continue
        sub = df.loc[win[0]:win[1]]
        jobs = [(sub[c], "") for c in sub.columns]
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                res = list(ex.map(_trend_job, jobs, chunksize=max(1, len(jobs) // (workers * 8))))
        else:
            res = [_trend_job(j) for j in jobs]
        r = pd.DataFrame(res, index=sub.columns)
        at = trend.seasonal_trend(land_aggregate(sub, w))
        f = fdr_shares(r["slope"], r["p_value"], area, alpha)
        f.pop("_q")
        out["windows"][name] = {"window": list(win), "aggregate_slope": at.get("slope"),
                                "aggregate_ci95": at.get("ci95"), "aggregate_p": at.get("p_value"), **f}
    out["claim_level"] = "statistically supported pattern in a model/forcing dataset; not causal (§13 levels 1-3)"
    write_json(out_dir / f"headline_{var}.json", out)
    return out
