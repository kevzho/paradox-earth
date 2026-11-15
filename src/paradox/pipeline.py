"""End-to-end orchestration: streaming ingest/QC -> regional stats (S2-S4) -> spatial statistics (§7) ->
physics-informed forecasting (§10). ML discovery (§8) and cross-lens analysis (§9) plug in after `spatial`
by reading `outputs/records.json`, `outputs/tiles.csv` and the regional CSVs.

Ingest reads one file at a time, so memory is set by the grid size, not the record length: a global 26-year run
holds per-tile sums, per-cell QC counters and (optionally) the PINN grid, never the full 0.25° cube."""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import gldas, qc, seasonal, spatial, trend, variability
from .contract import SourceManifest, analysis_record, file_checksum, write_json

if TYPE_CHECKING:  # torch is imported only when the PINN runs, never in the stats worker processes
    from .pinn import PINNConfig

STATS_VARS = ("soil_water_0_200cm", "Rainf_f_tavg")
PINN_VARS = ["land_water_store", "Rainf_f_tavg", "Evap_tavg", "runoff", "Tair_f_inst"]  # pinn.CHANNELS order


class PinnGridAccumulator:
    """Collects the PINN channels month by month, block-averaged by `coarsen` (1 = native 0.25°)."""

    def __init__(self, lat, lon, coarsen: int = 1, capacity: int = 0):
        self.lat, self.coarsen, self.capacity = np.asarray(lat), coarsen, capacity
        self.lat_c, self.lon_c = gldas.block_coords(lat, coarsen), gldas.block_coords(lon, coarsen)
        self.arr, self.n, self.months, self.mask = None, 0, [], None

    def update(self, ds):
        for i, t in enumerate(ds["time"].values):
            stack = np.stack([ds[v].isel(time=i).values for v in PINN_VARS]).astype(np.float32)
            stack = gldas.block_mean(stack, self.lat, self.coarsen)
            if self.arr is None:
                self.arr = np.empty((max(self.capacity, 1), *stack.shape), np.float32)
                self.mask = np.ones(stack.shape[-2:], bool)
            if self.n == len(self.arr):
                self.arr = np.concatenate([self.arr, np.empty_like(self.arr)])
            self.mask &= np.isfinite(stack).all(0)
            self.arr[self.n] = np.nan_to_num(stack)
            self.months.append(pd.Timestamp(t).month)
            self.n += 1

    def arrays(self):
        return self.arr[:self.n], self.mask, np.array(self.months)


@dataclass
class Ingested:
    series: dict          # variable -> DataFrame (time x tile)
    units: dict
    report: dict
    manifest_path: Path
    grid: gldas.TileGrid
    pinn: PinnGridAccumulator | None


def ingest(raw_dir, out_dir, bbox=None, tile_deg=1.0, variables=STATS_VARS, min_coverage=0.5,
           pinn=True, pinn_coarsen=1) -> Ingested:
    out_dir = Path(out_dir)
    files = gldas.list_files(raw_dir)
    grid = qacc = tacc = pacc = None
    title, units = "", {}
    for k, f in enumerate(files):
        m = gldas.add_derived(gldas.to_monthly_water_units(gldas.load_file(f, bbox=bbox)))
        if grid is None:
            grid = gldas.TileGrid(m.lat.values, m.lon.values, tile_deg)
            qacc = qc.QCAccumulator([v for v in gldas.ALL_VARS if v in m])
            tacc = gldas.TileAccumulator(grid, variables)
            pacc = PinnGridAccumulator(m.lat.values, m.lon.values, pinn_coarsen, len(files)) if pinn else None
            title, units = m.attrs.get("title", ""), {v: m[v].attrs.get("units", "") for v in variables}
        elif not grid.matches(m.lat.values, m.lon.values):
            raise ValueError(f"{f.name}: grid differs from {files[0].name}")
        qacc.update(m)
        tacc.update(m)
        if pacc is not None:
            pacc.update(m)
        if (k + 1) % 24 == 0:
            print(f"ingest: {k + 1}/{len(files)} files", flush=True)

    report = qacc.report()
    series = tacc.finalize(min_coverage)
    t = report["chronology"]
    manifest = SourceManifest(
        provider="NASA GES DISC", dataset=gldas.DATASET, variable=",".join(qacc.variables),
        version=gldas.VERSION, units="storage mm; flux mm/month totals; T K", temporal_resolution="monthly",
        spatial_resolution="0.25 deg", effective_spatial_support="0.25 deg model grid (land-surface model output)",
        coverage={"time": [t["start"], t["end"]],
                  "bbox": [float(grid.lon.min()), float(grid.lat.min()), float(grid.lon.max()), float(grid.lat.max())]},
        quality_flags=[] if report["passed"] else ["QC_FAILED"],
        missing_fraction=float(np.mean([v["missing_fraction_land"] for v in report["variables"].values()])),
        provenance={"files": len(files), "first_file_sha256_1MB": file_checksum(files[0]), "title": title,
                    "steps": ["load_file", "to_monthly_water_units", "add_derived", "streaming tile aggregation"],
                    "tile_deg": tile_deg, "min_coverage": min_coverage},
    )
    mpath = manifest.write(out_dir / "manifests" / f"{gldas.DATASET}.json")
    write_json(out_dir / "qc_report.json", report)
    for var, df in series.items():
        df.to_csv(out_dir / f"regional_{var}.csv")
    used = sorted(set().union(*(df.columns for df in series.values())))
    pd.DataFrame([(rid, *grid.boxes[rid]) for rid in used], columns=["region_id", "west", "south", "east", "north"]
                 ).to_csv(out_dir / "tiles.csv", index=False)
    return Ingested(series, units, report, mpath, grid, pacc)


def _attempt(fn, **fallback):
    """Run one analysis stage; on failure return `fallback` plus an error reason (one bad tile or stage must not
    kill a global run, nor discard the stages that did succeed)."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {**fallback, "status": "error", "reason": f"{type(e).__name__}: {e}"}


def _analyse(job):
    """Stats for one region x variable. Runs in worker processes; each stage fails independently."""
    var, rid, y, units, ref_period, egarch = job
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = _attempt(lambda: trend.seasonal_trend(y, units))  # trend is fitted on the raw series
        a = _attempt(lambda: seasonal.anomalies(y, ref_period))
        if "anomaly" not in a:
            emp, eg = dict(a), {"egarch_status": "unavailable", "reason": a["reason"]}
        else:
            emp = _attempt(lambda: variability.empirical_variability(a["anomaly"]))
            eg = (_attempt(lambda: variability.arima_egarch(a["anomaly"]), egarch_status="unavailable") if egarch
                  else {"egarch_status": "not_evaluated"})
    var_rec = {
        "baseline_change": {k: emp.get(k) for k in
                            ["variance_ratio_late_over_early", "variance_ratio_ci95", "brown_forsythe_p",
                             "mad_ratio_late_over_early", "split", "method"]},
        "egarch_status": eg["egarch_status"],
        "egarch": {k: eg.get(k) for k in ["arima", "arch_lm", "params", "persistence_beta",
                                          "asymmetry_gamma", "diagnostics", "gates", "reason"]},
    }
    var_rec["baseline_change"]["status"] = emp.get("status")
    row = {"variable": var, "region_id": rid, "slope_per_decade": tr.get("slope"), "trend_p": tr.get("p_value"),
           "var_ratio": emp.get("variance_ratio_late_over_early"), "bf_p": emp.get("brown_forsythe_p"),
           "egarch_status": eg["egarch_status"]}
    # only gate-passing fits are exported as volatility paths (§6 S4: otherwise ship the empirical baseline)
    cv = eg.get("conditional_volatility") if eg["egarch_status"] == "fitted" else None
    return tr, var_rec, row, cv


class _Serial:
    """In-process stand-in for ProcessPoolExecutor (workers=1)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, items, chunksize=1):
        return map(fn, items)


def run_stats(series: dict, units: dict, out_dir, ref_period=None, manifest_path=None, egarch=True, workers=1):
    out_dir = Path(out_dir)
    jobs = [(var, rid, df[rid], units.get(var, ""), ref_period, egarch) for var, df in series.items() for rid in df]
    step = max(1, len(jobs) // 20)
    results = []
    with ProcessPoolExecutor(workers) if workers > 1 else _Serial() as ex:
        for i, r in enumerate(ex.map(_analyse, jobs, chunksize=max(1, len(jobs) // (workers * 8))), 1):
            results.append(r)
            if i % step == 0 or i == len(jobs):
                print(f"stats: {i}/{len(jobs)} series", flush=True)
    records, rows, vol = [], [], {v: {} for v in series}
    for (var, rid, *_), (tr, var_rec, row, cv) in zip(jobs, results):
        records.append(analysis_record(rid, gldas.DATASET, var, tr.get("window", [None, None]), trend=tr,
                                       variability=var_rec, manifest_path=str(manifest_path)))
        rows.append(row)
        if cv is not None:
            vol[var][rid] = cv
    for var, cols in vol.items():
        path = out_dir / "volatility" / f"{var}.csv"  # EGARCH 'fitted' tiles only
        if cols:
            pd.DataFrame(cols).to_csv(path)
        else:
            path.unlink(missing_ok=True)  # never leave a previous run's paths behind
    write_json(out_dir / "records.json", records)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "stats_summary.csv", index=False)
    return records, summary


def run_pinn(pacc: PinnGridAccumulator, out_dir, cfg: PINNConfig, lams=(0.0, 0.1)):
    import torch

    from .pinn import ablation_arrays

    res = ablation_arrays(*pacc.arrays(), cfg, lams)
    res["coarsen"] = pacc.coarsen
    for k, run in res["runs"].items():
        torch.save(run.pop("_model").state_dict(), Path(out_dir) / "models" / f"pinn_{k}.pt")
    write_json(Path(out_dir) / "pinn_ablation.json", res)
    return res


def run_all(raw_dir, out_dir, bbox=None, tile_deg=1.0, variables=STATS_VARS, ref_period=None, workers=None,
            pinn_cfg: PINNConfig | None = None, lams=(0.0, 0.1), pinn=True, pinn_coarsen=1, spatial_perms=9999):
    from .pinn import PINNConfig

    out_dir = Path(out_dir)
    workers = workers or os.cpu_count() or 1
    for sub in ["manifests", "volatility", "models"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    ing = ingest(raw_dir, out_dir, bbox, tile_deg, variables, pinn=pinn, pinn_coarsen=pinn_coarsen)
    print(f"stats: {sum(df.shape[1] for df in ing.series.values())} series on {workers} workers", flush=True)
    records, summary = run_stats(ing.series, ing.units, out_dir, ref_period, ing.manifest_path, workers=workers)
    spatial_summary = None
    if spatial_perms:
        spatial_summary = spatial.run_spatial(records, pd.read_csv(out_dir / "tiles.csv"), out_dir,
                                              permutations=spatial_perms)
        write_json(out_dir / "records.json", records)
    pinn_res = run_pinn(ing.pinn, out_dir, pinn_cfg or PINNConfig(), lams) if pinn else None
    write_json(out_dir / "run_config.json",
               {"raw_dir": str(raw_dir), "bbox": bbox, "tile_deg": tile_deg, "variables": list(variables),
                "ref_period": ref_period, "workers": workers,
                "pinn": asdict(pinn_cfg or PINNConfig()) if pinn else None, "pinn_coarsen": pinn_coarsen,
                "spatial_permutations": spatial_perms})
    return {"qc": ing.report, "summary": summary, "spatial": spatial_summary, "pinn": pinn_res}
