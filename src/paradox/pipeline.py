"""End-to-end orchestration: ingest/QC -> regional stats (S2-S4) -> physics-informed forecasting (§10).
Spatial statistics (§7), ML discovery (§8) and cross-lens analysis (§9) plug in between `stats` and `pinn`
by reading `outputs/records.json` and the regional CSVs."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import gldas, qc, seasonal, trend, variability
from .contract import SourceManifest, analysis_record, file_checksum
from .pinn import PINNConfig, ablation


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def ingest(raw_dir, out_dir, bbox=None):
    out_dir = Path(out_dir)
    files = sorted(Path(raw_dir).glob("*.nc4")) + sorted(Path(raw_dir).glob("*.nc"))
    ds = gldas.add_derived(gldas.to_monthly_water_units(gldas.open_gldas(files, bbox=bbox)))
    report = qc.qc_dataset(ds, [v for v in gldas.ALL_VARS if v in ds])
    t = report["chronology"]
    manifest = SourceManifest(
        provider="NASA GES DISC", dataset=gldas.DATASET, variable=",".join(v for v in gldas.ALL_VARS if v in ds),
        version=gldas.VERSION, units=ds.attrs["harmonised_units"], temporal_resolution="monthly",
        spatial_resolution="0.25 deg", effective_spatial_support="0.25 deg model grid (land-surface model output)",
        coverage={"time": [t["start"], t["end"]],
                  "bbox": [float(ds.lon.min()), float(ds.lat.min()), float(ds.lon.max()), float(ds.lat.max())]},
        quality_flags=[] if report["passed"] else ["QC_FAILED"],
        missing_fraction=float(np.mean([v["missing_fraction_land"] for v in report["variables"].values()])),
        provenance={"files": len(files), "first_file_sha256_1MB": file_checksum(files[0]) if files else None,
                    "title": ds.attrs.get("title", ""), "steps": ["open_gldas", "to_monthly_water_units", "add_derived"]},
    )
    mpath = manifest.write(out_dir / "manifests" / f"{gldas.DATASET}.json")
    (out_dir / "qc_report.json").write_text(json.dumps(report, indent=2, default=_json_default))
    return ds, report, mpath


def run_stats(ds, out_dir, variables=("soil_water_0_200cm", "Rainf_f_tavg"), tile_deg=1.0,
              ref_period=None, manifest_path=None, egarch=True):
    out_dir = Path(out_dir)
    regions = gldas.tile_regions(ds, tile_deg)
    records, rows = [], []
    for var in variables:
        units = ds[var].attrs.get("units", "")
        series = gldas.region_series(ds, var, regions)
        series.to_csv(out_dir / f"regional_{var}.csv")
        for rid in series:
            a = seasonal.anomalies(series[rid], ref_period)
            tr = trend.seasonal_trend(a["raw"], units)
            emp = variability.empirical_variability(a["anomaly"])
            eg = variability.arima_egarch(a["anomaly"]) if egarch else {"egarch_status": "not_evaluated"}
            var_rec = {
                "baseline_change": {k: emp.get(k) for k in
                                    ["variance_ratio_late_over_early", "variance_ratio_ci95", "brown_forsythe_p",
                                     "mad_ratio_late_over_early", "split", "method"]},
                "egarch_status": eg["egarch_status"],
                "egarch": {k: eg.get(k) for k in ["arima", "arch_lm", "params", "persistence_beta",
                                                  "asymmetry_gamma", "diagnostics", "gates", "reason"]},
            }
            if "conditional_volatility" in eg:
                eg["conditional_volatility"].rename(rid).to_csv(out_dir / "volatility" / f"{var}__{rid}.csv")
            records.append(analysis_record(rid, gldas.DATASET, var, tr.get("window", [None, None]), trend=tr,
                                           variability=var_rec, manifest_path=str(manifest_path)))
            rows.append({"variable": var, "region_id": rid, "slope_per_decade": tr.get("slope"),
                         "trend_p": tr.get("p_value"), "var_ratio": emp.get("variance_ratio_late_over_early"),
                         "bf_p": emp.get("brown_forsythe_p"), "egarch_status": eg["egarch_status"]})
    (out_dir / "records.json").write_text(json.dumps(records, indent=2, default=_json_default))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "stats_summary.csv", index=False)
    return records, summary


def run_pinn(ds, out_dir, cfg: PINNConfig, lams=(0.0, 0.1)):
    import torch

    res = ablation(ds, cfg, lams)
    for k, run in res["runs"].items():
        torch.save(run.pop("_model").state_dict(), Path(out_dir) / "models" / f"pinn_{k}.pt")
    (Path(out_dir) / "pinn_ablation.json").write_text(json.dumps(res, indent=2, default=_json_default))
    return res


def run_all(raw_dir, out_dir, bbox=None, pinn_cfg: PINNConfig | None = None, stats_kwargs=None, lams=(0.0, 0.1)):
    out_dir = Path(out_dir)
    for sub in ["manifests", "volatility", "models"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    ds, report, mpath = ingest(raw_dir, out_dir, bbox)
    records, summary = run_stats(ds, out_dir, manifest_path=mpath, **(stats_kwargs or {}))
    pinn_res = run_pinn(ds, out_dir, pinn_cfg or PINNConfig(), lams)
    (out_dir / "run_config.json").write_text(json.dumps(
        {"raw_dir": str(raw_dir), "bbox": bbox, "pinn": asdict(pinn_cfg or PINNConfig()),
         "stats": stats_kwargs or {}}, indent=2, default=_json_default))
    return {"qc": report, "summary": summary, "pinn": pinn_res}
