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


def soil_water_story(artifact: Path, fldas: Path, grace: Path) -> dict:
    """The soil-water investigation for the explorer: why GLDAS-2.1's decline was withdrawn and what a consistently
    forced product (NASA FLDAS) and observations (GRACE/GRACE-FO) show instead. Missing pieces are left out."""
    story = {}
    if artifact.exists():
        a = json.loads(artifact.read_text())
        h = a.get("headline", {})
        story["artifact_control"] = {
            "what": "GLDAS-2.1 vs GLDAS-2.0: same Noah model and grid, different forcing, 2000-2014",
            "step_date": a.get("step_date"), "forcing": a.get("forcing"), "source": a.get("forcing_source"),
            "swdown_step_21_W_m2": h.get("swdown_step_2.1_W_m2"), "swdown_step_21_ci95": h.get("swdown_step_2.1_ci95"),
            "swdown_step_20_W_m2": h.get("swdown_step_2.0_W_m2"), "swdown_step_20_ci95": h.get("swdown_step_2.0_ci95"),
            "swdown_step_difference_W_m2": h.get("swdown_step_difference_W_m2"),
            "headline": {k: v for k, v in h.items() if isinstance(v, (int, float, str, list))},
            "figure": "../demo/artifact_control.png",
            "reading": ("Incoming solar radiation steps up only in GLDAS-2.1's forcing at 2010-08, and evaporation responds "
                        "only in 2.1. The 2011-2013 soil-water dip appears in both versions, deeper in 2.1: the forcing "
                        "change exaggerated a real dip into an apparent long-term decline.")}
    fh = fldas / "headline_soil_water_0_200cm.json"
    if fh.exists():
        f = json.loads(fh.read_text())
        ag, rg = f["aggregate"], f["regional"]
        story["fldas"] = {
            "dataset": "FLDAS_NOAH01_C_GL_M v001 (Noah 3.6.1, MERRA-2 + CHIRPS forcing), doi:10.5067/5NHC22T9375G",
            "window": ag.get("window"), "aggregate_slope_mm_per_decade": ag.get("slope"), "aggregate_ci95": ag.get("ci95"),
            "aggregate_p": ag.get("p_value"), "share_area_wetting": rg.get("share_area_up"),
            "share_area_drying": rg.get("share_area_down"), "weights": f.get("weights"),
            "regions": {k: [{kk: r[kk] for kk in ("n_tiles", "area_km2", "centroid_lat", "centroid_lon", "bbox",
                                                   "median_slope_per_decade", "median_slope_pct_per_decade")}
                            for r in f["regions"].get(k, [])[:5]] for k in ("up", "down")}}
    gs, gc = grace / "grace_trends_summary.json", sorted(grace.glob("compare_outputs_fldas_*.json"))
    if gs.exists():
        g = json.loads(gs.read_text())
        story["grace"] = {"product": (g.get("grace") or {}).get("title"), "window": g.get("window"),
                          "land_area_share_fdr_significant": g.get("land_area_share_fdr_significant"),
                          "land_area_share_negative_slope": g.get("land_area_share_negative_slope")}
        if gc:
            c = json.loads(gc[-1].read_text())
            story["grace"]["vs_fldas_soil_water"] = {
                "window": c.get("window"), "sign_match_area_sig_either": (c.get("sign_match_share_area_weighted") or {}).get("sig_in_either"),
                "spearman_slopes": (c.get("spearman_slopes") or {}).get("all_boxes"),
                "hotspots": {k: {kk: v.get(kk) for kk in ("grace_mean_slope_mm_decade", "model_mean_slope_mm_decade",
                                                          "n_grace_sig", "n_boxes")}
                             for k, v in (c.get("hotspots") or {}).items()},
                "note": "GRACE measures total water storage (incl. groundwater, surface water, ice); compare signs, not sizes"}
    return story


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_global")
    ap.add_argument("--site", default="site")
    ap.add_argument("--var", default=VAR)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--artifact", default="outputs_artifact/artifact_control.json")
    ap.add_argument("--fldas", default="outputs_fldas")
    ap.add_argument("--grace", default="outputs_grace")
    ap.add_argument("--cross-check", default="outputs_fldas/precip_cross_check.json",
                    help="CHIRPS/FLDAS precipitation cross-check (scripts/precip_cross_check.py); skipped if absent")
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
        "project": "Paradox Earth", "tagline": "The global average looks flat. A few regions are not.",
        "challenge": "Be An Earth System Trend Detective! (NASA Space Apps Challenge 2026)",
        "dataset": {"name": "GLDAS-2.1 Noah Land Surface Model L4 monthly 0.25 x 0.25 degree",
                    "short_name": "GLDAS_NOAH025_M", "version": "2.1", "doi": "10.5067/SXAVCZFAQLNO",
                    "provider": "NASA GES DISC", "variable": var,
                    "variable_label": "Precipitation (rain + snow), GLDAS-2.1 forcing (GPCP-based)",
                    "units": "mm/month", "period": [f"{df.index[0]:%Y-%m}", f"{df.index[-1]:%Y-%m}"]},
        "tile_deg": run_cfg.get("tile_deg"), "n_tiles": len(ids), "qc_passed": qc.get("passed"),
        "source_run": out.resolve().name,
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
        "retraction": {"breakpoints": ["2010-08", "2024-07"], "window": ["2012-01", "2024-06"], "text": (
                       "Soil water was the original headline: a 26-year global decline. Paradox Earth's own checks "
                       "showed it is mostly two level shifts that follow documented changes in the radiation data "
                       "driving the GLDAS model (a step in 2010-08 and a degraded period from 2024-07; GES DISC "
                       "README_GLDAS2). Within the single consistent period 2012-01 to 2024-06 the global decline is "
                       "not significant, so the trend was withdrawn and soil water is analysed only inside that "
                       "window.")},
    }
    meta["soil_water"] = soil_water_story(Path(args.artifact), Path(args.fldas), Path(args.grace))
    xc = Path(args.cross_check)
    if xc.exists():
        cross = json.loads(xc.read_text())
        write_json(data / "crosscheck.json", cross)
        meta["cross_checks"] = {"precip_chirps_fldas": {k: v for k, v in cross.items() if k != "regions"}}
    write_json(data / "meta.json", meta)
    print(f"exported {len(ids)} tiles, {len(blocks)} series blocks, "
          f"{len(regions['up'])}+{len(regions['down'])} regions -> {data}")


if __name__ == "__main__":
    main()
