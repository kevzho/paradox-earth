"""Cross-check the flagship precipitation trends against an independent product.

  python scripts/precip_cross_check.py --gldas outputs_global_v2 --fldas outputs_fldas --site site

GLDAS-2.1 precipitation is GPCP-based; FLDAS (NASA, Noah forced by CHIRPS + MERRA-2) uses CHIRPS, a separate
satellite-gauge product. Over their common window (2003-01..2024-12) this compares per-tile trend signs (FDR within
each product) and, for every coherent region in site/data/regions.json, the share of the region's tiles whose CHIRPS
trend has the same sign (and is individually significant). Writes <fldas>/precip_cross_check.json.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
from statsmodels.stats.multitest import multipletests
from paradox import trend
def job(y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); r = trend.seasonal_trend(y)
    return r.get("slope"), r.get("p_value")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gldas", default="outputs_global_v2")
    ap.add_argument("--fldas", default="outputs_fldas")
    ap.add_argument("--site", default="site")
    a = ap.parse_args()
    g = pd.read_csv(f'{a.gldas}/regional_Rainf_f_tavg.csv', index_col=0, parse_dates=True).loc['2003-01':'2024-12']
    fs = pd.read_csv(f'{a.fldas}/stats_summary.csv')
    fs = fs[fs.variable == 'Rainf_f_tavg'].set_index('region_id')
    ids = [c for c in g.columns if c in fs.index]
    with ProcessPoolExecutor(8) as ex: res = list(ex.map(job, [g[c] for c in ids], chunksize=64))
    gs = pd.DataFrame(res, index=ids, columns=['slope', 'p'])
    tiles = pd.read_csv(f'{a.gldas}/tiles.csv').set_index('region_id').loc[ids]
    area = tiles['area_km2__Rainf_f_tavg']
    qg = multipletests(gs.p, method='fdr_bh')[1]; qf = multipletests(fs.loc[ids, 'trend_p'], method='fdr_bh')[1]
    sg, sf = np.sign(gs.slope.values), np.sign(fs.loc[ids, 'slope_per_decade'].values)
    either, both = (qg < .05) | (qf < .05), (qg < .05) & (qf < .05)
    out = {"window": ["2003-01", "2024-12"], "n_tiles": len(ids),
           "gldas_fdr_sig_area_share": float(area[qg < .05].sum() / area.sum()),
           "fldas_fdr_sig_area_share": float(area[qf < .05].sum() / area.sum()),
           "sign_match_area_sig_either": float(area[either & (sg == sf)].sum() / area[either].sum()),
           "sign_match_area_sig_both": float(area[both & (sg == sf)].sum() / max(area[both].sum(), 1)),
           "n_sig_both": int(both.sum()),
           "spearman_slopes_all": float(stats.spearmanr(gs.slope, fs.loc[ids, 'slope_per_decade']).statistic),
           "sign_match_area_all": float(area[sg == sf].sum() / area.sum())}
    # every coherent region from the GLDAS headline: does CHIRPS (via FLDAS) agree in sign over 2003-2024?
    regions = json.load(open(f'{a.site}/data/regions.json'))
    out["regions"] = {}
    for kind in ("up", "down"):
        for r in regions[kind]:
            t = [x for x in r['tiles'] if x in fs.index and x in gs.index]
            if not t:
                continue
            fl, sign = fs.loc[t], (1 if kind == "up" else -1)
            out["regions"][r["id"]] = {
                "n_tiles": len(t), "gldas_2003_2024_median_slope": float(gs.loc[t, 'slope'].median()),
                "fldas_chirps_2003_2024_median_slope": float(fl.slope_per_decade.median()),
                "fldas_share_tiles_same_sign": float((np.sign(fl.slope_per_decade) == sign).mean()),
                "fldas_share_tiles_raw_p05_same_sign": float(((fl.trend_p < .05) & (np.sign(fl.slope_per_decade) == sign)).mean())}
    json.dump(out, open(f'{a.fldas}/precip_cross_check.json', 'w'), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != 'regions'}, indent=1))
