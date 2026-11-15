# Paradox Earth

NASA Space Apps Challenge 2026: *Be An Earth System Trend Detective!*
**The planet is changing. The averages are hiding the story.** (Manifest v3.0)

This repo holds the scientific backend: NASA GLDAS ingestion and QC, seasonal trend inference, variability change
(empirical + ARIMA-EGARCH), and a physics-informed water-balance forecaster. The data sources are listed in
**[DATA_TO_USE.md](DATA_TO_USE.md)**.

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests                  # 21 smoke/equivalence tests, ~30 s
.venv/bin/python run.py --synthetic                  # full pipeline on synthetic GLDAS-like data, ~1 min

# real data (needs Earthdata login, see DATA_TO_USE.md)
.venv/bin/python scripts/download_data.py gldas --start 2000-01 --end 2025-12        # global files, ~7.8 GB
.venv/bin/python run.py --raw data/raw/gldas --tile 1.0 --pinn-coarsen 4 --ref 2000-01 2017-12   # global
.venv/bin/python run.py --raw data/raw/gldas --bbox -125 24 -66 50 --tile 1.0 --out outputs_conus   # CONUS dev loop
```

Global runs stream one file at a time (~0.5 s/file, ~1–2 GB RAM), so a 16 GB laptop is enough for ingest + statistics.
Per-region statistics run on `--workers` processes (default: all cores; ~0.8 s per series, dominated by EGARCH).
`--pinn-coarsen 4` trains the forecaster on a 1° grid; native 0.25° global training needs a GPU and is not wired up
yet (`PINNConfig.device` is unused). `--no-pinn` skips it.

```bash
```

## What is built (Manifest section → module)

| Manifest | Module | What it does |
|---|---|---|
| §4.1, §5 data contract | `gldas.py`, `contract.py` | Opens monthly GLDAS files with an optional bbox. Converts units: storage in mm, fluxes in mm/month totals. Derives soil water 0–200 cm, land-water store and net flux P−ET−R. Streams files one at a time into grid-aligned, area-weighted (cos-lat) tiles with a coverage threshold (one `bincount` per month). Writes the source manifest and the §16 analysis records. |
| S1 QC | `qc.py` | Chronology checks (gaps, duplicates, order), units present, missingness on land cells, physical-range checks. Never modifies the data. |
| S2 seasonal | `seasonal.py` | Monthly climatology fitted on a reference (training-only) window. Keeps raw, climatology and anomaly series side by side. |
| S3 trend | `trend.py` | `y = a + b·t + Σ month dummies`, Newey–West HAC standard errors. Reports slope per decade, 95% CI, p, n, window. A paired-difference test compares slopes between regions directly. |
| S4 variability | `variability.py` | Empirical baseline: rolling variance, early-vs-late variance ratio (Brown–Forsythe test + moving-block bootstrap CI), MAD ratio. Then ARIMA(p,0,q)+trend chosen by AIC, ARCH-LM test, and EGARCH(1,1) via `arch`, reporting persistence β, asymmetry γ, standard errors and conditional volatility. Every fit passes an **eligibility gate**: status is `fitted`, `inconclusive` or `unavailable`, and the record lists which gates failed. |
| §10 PINN | `pinn.py` | Spatiotemporal CNN over the last *k* months of weather and state [S, P, ET, R, T]. It predicts next-month [ΔS, P, ET, R] with loss L = L_pred + λ·L_water-balance (ΔS = P − ET − R, with `centered` alignment for GLDAS monthly means). Normalisation uses training data only. Includes a chronological split, a spatial holdout, persistence / seasonal-naive / linear-AR baselines, and a λ ablation. |
| §15.6 flow | `pipeline.py`, `run.py` | ingest → stats → pinn. Writes everything to `outputs/`. |

Outputs: `qc_report.json`, `manifests/`, `regional_<var>.csv`, `tiles.csv` (tile boxes), `records.json` (§16 records),
`stats_summary.csv`, `volatility/<var>.csv` (one column per tile), `pinn_ablation.json`, `models/pinn_lambda=*.pt`.

## Partner handoff: §7–§9 (between EGARCH and the PINN)

Not built yet: **spatial statistics** (Moran's I, LISA, permutation inference, BH-FDR), **ML discovery** (PCA,
Isolation Forest, 1D-CNN encoder, contrastive matching, ablation) and **cross-lens analysis**.
Plug-in points:
- Input: `outputs/records.json` (one record per region × variable) and `outputs/regional_<var>.csv`.
- Fill `record["spatial"]` (`lisa_label`, `inference_status`) and `record["ml"]` (`ranking_score`, `model_version`).
- Tiles have ids `tile_<south>_<west>`, aligned to multiples of the tile size; boxes are in `outputs/tiles.csv`.
  Build spatial weights from those (queen or kNN). Global grids wrap at ±180°, so link the dateline columns.
- A LISA + BH-FDR implementation you can adapt is in `interpolator-mega/src/interp/lisa.py`.
- Use the regional anomaly series (`seasonal.anomalies`) as ML inputs. Train only on the early period.

## Scientific guardrails (from the manifest)
- GLDAS is land-surface model output, not independent observations.
- The PINN target is soil + snow + canopy storage, **not** terrestrial water storage. GRACE is a complementary
  check, not ground truth.
- A fitted EGARCH volatility path does not by itself prove a persistent change in variance. If the gate fails,
  report the empirical baseline instead.
- Missing data stays missing in every statistical step.
- The forecasting extension ships only if it passes Gate G (λ ablation + baseline comparison documented).

Reviewer agent for code changes: `.claude/agents/code-reviewer.md` (lean, correctness only).
