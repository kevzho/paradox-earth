"""Export a pipeline run into compact static JSON for the explorer page, Discovery Cards, slides and video.

  python scripts/export_site.py --out outputs_global --site site

Writes (all strict JSON, numbers rounded for size):
  site/data/meta.json        dataset, period, units, method strings, flagship headline (aggregate, FDR shares,
                             window sensitivity), caveats, soil-water retraction note
  site/data/tiles.json       {"cols": [...], "rows": [[...], ...]} one row per flagship tile:
                             id, west, south, slope (mm/month per decade), pct (%/decade), p, q (BH across tiles),
                             sig (+1 wetting / -1 drying / 0 not significant at q<0.05),
                             lisa (HH/LL/HL/LH/ns/nn), region
  site/data/regions.json     coherent wetting/drying regions (from headline_<var>.json) with ids "up1", "down1", ...
  site/data/series/<block>.json  monthly anomalies per tile, grouped in 10x10 degree blocks (loaded on click):
                             {"start": "2000-01", "tiles": {id: [anomaly mm/month, ...]}}
The page computes nothing statistical; every number shown comes from these files.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paradox.contract import write_json  # noqa: E402

VAR = "Rainf_f_tavg"
LISA_CODE = {"HH_cluster": "HH", "LL_cluster": "LL", "HL_outlier": "HL", "LH_outlier": "LH",
             "inconclusive": "ns", "no_neighbors": "nn"}


def block_key(south: float, west: float) -> str:
    return f"b_{int(math.floor(south / 10) * 10):+04d}_{int(math.floor(west / 10) * 10):+05d}"


def r(x, nd):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_global")
    ap.add_argument("--site", default="site")
    ap.add_argument("--var", default=VAR)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    out, site, var = Path(args.out), Path(args.site), args.var
    data = site / "data"
    (data / "series").mkdir(parents=True, exist_ok=True)

    tiles = pd.read_csv(out / "tiles.csv").set_index("region_id")
    stats = pd.read_csv(out / "stats_summary.csv")
    stats = stats[stats["variable"] == var].set_index("region_id")
    df = pd.read_csv(out / f"regional_{var}.csv", index_col=0, parse_dates=True)
    head = json.loads((out / f"headline_{var}.json").read_text())
    run_cfg = json.loads((out / "run_config.json").read_text())
    qc = json.loads((out / "qc_report.json").read_text())

    ids = [c for c in df.columns if c in stats.index]
    st = stats.loc[ids]
    mean = df[ids].mean()
    pct = st["slope_pct_per_decade"] if "slope_pct_per_decade" in st else 100 * st["slope_per_decade"] / mean
    ok = st["trend_p"].notna() & st["slope_per_decade"].notna()
    q = pd.Series(np.nan, index=ids)
    q[ok] = multipletests(st.loc[ok, "trend_p"], alpha=args.alpha, method="fdr_bh")[1]
    sig = np.where(q < args.alpha, np.sign(st["slope_per_decade"]), 0).astype(int)

    lisa = pd.Series("ns", index=ids)
    for field in ("trend_slope_pct", "trend_slope"):
        p = out / "spatial" / f"{var}__{field}.csv"
        if p.exists():
            lisa = pd.read_csv(p, index_col=0)["label"].reindex(ids).map(LISA_CODE).fillna("ns")
            lisa_field = field
            break
    else:
        lisa_field = None

    region_of, regions = {}, {"up": [], "down": [], "rule": head.get("regions", {}).get("rule")}
    for kind in ("up", "down"):
        for i, g in enumerate(head.get("regions", {}).get(kind, []), 1):
            rid = f"{kind}{i}"
            for t in g["tiles"]:
                region_of[t] = rid
            regions[kind].append({"id": rid, **{k: v for k, v in g.items() if k != "tiles"}, "tiles": g["tiles"]})

    cols = ["id", "west", "south", "slope", "pct", "p", "q", "sig", "lisa", "region"]
    rows = [[t, r(tiles.at[t, "west"], 2), r(tiles.at[t, "south"], 2), r(st.at[t, "slope_per_decade"], 3),
             r(pct[t], 2), r(st.at[t, "trend_p"], 5), r(q[t], 5), int(sig[ids.index(t)]), lisa[t],
             region_of.get(t)] for t in ids]
    write_json(data / "tiles.json", {"variable": var, "tile_deg": run_cfg.get("tile_deg"), "cols": cols, "rows": rows},
               compact=True)
    write_json(data / "regions.json", regions)

    clim = df[ids].groupby(df.index.month).transform("mean")
    anom = (df[ids] - clim).round(1)
    blocks = {}
    for t in ids:
        blocks.setdefault(block_key(tiles.at[t, "south"], tiles.at[t, "west"]), []).append(t)
    for key, members in blocks.items():
        write_json(data / "series" / f"{key}.json",
                   {"start": f"{df.index[0]:%Y-%m}", "tiles": {t: anom[t].tolist() for t in members}}, compact=True)

    a, g = head["aggregate"], head["regional"]
    meta = {
        "project": "Paradox Earth", "tagline": "The planet is changing. The averages are hiding the story.",
        "challenge": "Be An Earth System Trend Detective! (NASA Space Apps Challenge 2026)",
        "dataset": {"name": "GLDAS-2.1 Noah Land Surface Model L4 monthly 0.25 x 0.25 degree",
                    "short_name": "GLDAS_NOAH025_M", "version": "2.1", "doi": "10.5067/SXAVCZFAQLNO",
                    "provider": "NASA GES DISC", "variable": var,
                    "variable_label": "Precipitation (rain + snow), GLDAS-2.1 forcing (GPCP-based)",
                    "units": "mm/month", "period": [f"{df.index[0]:%Y-%m}", f"{df.index[-1]:%Y-%m}"]},
        "tile_deg": run_cfg.get("tile_deg"), "n_tiles": len(ids), "qc_passed": qc.get("passed"),
        "headline": {"aggregate": a, "regional": g, "windows": head.get("windows", {}), "weights": head.get("weights")},
        "methods": {
            "trend": a.get("method"),
            "multiple_testing": f"Benjamini-Hochberg FDR across tiles, q < {args.alpha}",
            "spatial": f"local Moran's I (LISA) on {lisa_field}, 9,999 conditional permutations, BH-FDR; "
                       "queen weights with dateline wrap; kNN and Getis-Ord Gi* robustness",
            "regions": regions["rule"]},
        "caveats": [
            "GLDAS-2.1 precipitation is model forcing derived from GPCP; it is not an independent rain-gauge record.",
            "Trends are statistical over 2000-2025; they shrink in shorter windows and do not identify causes.",
            "Tile shares are after false-discovery correction; individual tiles can still be false positives."],
        "retraction": ("Soil water was the original headline: a 26-year global decline. Paradox Earth's own checks "
                       "showed it is mostly two level shifts that follow documented changes in the radiation data "
                       "driving the GLDAS model (a step in 2010-08 and a degraded period from 2024-07; GES DISC "
                       "README_GLDAS2). Within the single consistent period 2012-01 to 2024-06 the global decline is "
                       "not significant, so the trend was withdrawn and soil water is analysed only inside that "
                       "window."),
    }
    write_json(data / "meta.json", meta)
    print(f"exported {len(ids)} tiles, {len(blocks)} series blocks, "
          f"{len(regions['up'])}+{len(regions['down'])} regions -> {data}")


if __name__ == "__main__":
    main()
