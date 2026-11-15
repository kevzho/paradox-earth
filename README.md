# Paradox Earth

**The global average looks flat. A few regions are not.**

NASA International Space Apps Challenge 2026, challenge *Be An Earth System Trend Detective!*

Averaged over all of Earth's land, precipitation in NASA's GLDAS-2.1 dataset shows no detectable trend from 2000 to
2025. The land-area-weighted trend is **-0.88 mm/month per decade** (-1.3% per decade of a 69.9 mm/month mean), with
a 95% confidence interval of **[-1.88, +0.12] mm/month per decade** and **p = 0.08**. The trend could be zero, so it
is **not detectable**. Underneath that flat average, we tested 17,284 one-degree land tiles and corrected for false
discoveries (Benjamini-Hochberg FDR, q < 0.05). Most land shows no detectable trend. A small share does, in
**opposite directions**: **4.3% of tested land area (5.7 million km²) shows a detectable wetting trend** and **3.1%
(4.2 million km²) a detectable drying trend** over 2000-2025. These shares are smaller in shorter windows (see
Limits). Examples include wetting in the Central Sahel and northern China, and drying along the southern Andes
(south-central Chile to Patagonia) and in the southwestern Amazon. That is the paradox: a flat global mean can sit on
top of regional trends that point in opposite directions. Paradox Earth finds these regions, grades how strong the
evidence is for each one, and says where the evidence is weak.

- **Live 3D explorer:** https://kevzho.github.io/paradox-earth/site/
- **Slides (7, PDF):** [demo/paradox_earth_slides.pdf](demo/paradox_earth_slides.pdf) (HTML source: [demo/slides.html](demo/slides.html))
- **30-second video:** [demo/paradox_earth_30s.mp4](demo/paradox_earth_30s.mp4) (online link to be uploaded)
- **Code:** https://github.com/kevzho/paradox-earth

## What makes it different

- **A trend detective that grades its own evidence.** Each "Discovery Card" in the explorer comes with a robustness
  checklist that the pipeline computes itself. Regions are selected by FDR-corrected significance, so that row
  passes by construction. The graded checks are three more: stability across overlapping time windows (2000-2025,
  2003-2025, 2000-2020), agreement between two definitions of "neighbouring tile" (same data, different spatial
  weights), and agreement with published studies. Every number on the page is read from the exported data. None is
  typed in by hand.
- **Calibrated statistics.** Monthly climate series are strongly autocorrelated, which makes standard trend tests
  "find" trends far too often. We tested our trend and variance tests on simulated no-trend data. They give false
  positives about 3-6% of the time at a nominal 5%. The Newey-West test we used before gave 29-34% on soil water.
- **It caught its own false trend.** Our first headline was a 26-year global decline in soil water. The pipeline's
  own checks showed that this was mostly two level shifts. Their timing lines up with documented changes in the
  radiation data that drive the GLDAS model: a step in 2010-08 and a degraded period from 2024-07 (GES DISC
  README_GLDAS2). Within the one consistent period, 2012-01 to 2024-06, the global soil-water decline is not
  significant. So we withdrew that trend. Soil water is analysed only inside that window. We switched the
  flagship variable to precipitation, which in GLDAS-2.1 comes from a different forcing source (GPCP-based) than the
  radiation fields that changed.

## How it works

1. **NASA data.** 312 monthly global files (2000-01 to 2025-12) of
   [GLDAS-2.1 Noah, 0.25°, monthly](https://doi.org/10.5067/SXAVCZFAQLNO) (`GLDAS_NOAH025_M`, DOI
   10.5067/SXAVCZFAQLNO) from NASA GES DISC, downloaded with a free Earthdata login.
2. **Quality control.** Checks for gaps, units and physical ranges and a land mask (plus a glacier/ice-sheet mask
   for soil water), then averaging into 17,284 one-degree land tiles weighted by true land area.
3. **Seasonal anomalies.** Monthly climatology from a 2000-2017 reference period gives anomalies, used for the
   variability tests and the maps.
4. **Trend per tile.** OLS slope on the monthly series with month-of-year terms. The p-value is the larger of two
   autocorrelation-aware tests (an intersection-union test): an AR(1) test on monthly data with bias-corrected ρ,
   and an annual-mean test with an AR(1) effective sample size (Santer et al. 2000). Then Benjamini-Hochberg FDR
   across all tiles.
5. **Variance change.** A prewhitened, state-adjusted Brown-Forsythe test (early vs late half), plus ARIMA-EGARCH
   where an eligibility gate allows it.
6. **Regions.** Local Moran's I (LISA, 9,999 permutations) finds spatial clusters. A region is a connected group of
   at least 5 tiles that is a cluster and also FDR-significant with the same sign.
7. **Export.** Results go to static JSON for the explorer, the Discovery Cards, the slides and the video.

## Limits

- **Not causal.** These are statistical trends over 2000-2025. They do not say why precipitation changed, and they
  do not say the trends will continue.
- **A forcing dataset, not rain gauges.** GLDAS-2.1 precipitation is model *forcing* derived from GPCP. It is not an
  independent observation record.
- **Results depend on the window.** In shorter windows, the FDR-significant shares shrink. For 2003-2025 they are
  1.3% wetting and 2.6% drying land area. For 2000-2020 they are 0.7% and 1.0%. The trends of the highlighted
  regions (OLS on the region's annual-mean anomalies) keep their sign and stay within a factor of 2 in every window.
- **The shares are small.** 24% of tiles have raw p < 0.05, against about 5% expected by chance. After FDR
  correction, only the 4.3% and 3.1% of land area above remain, and individual tiles can still be false positives.
- **Percentages in dry places.** Percent-per-decade changes look large where the baseline is small (e.g. the Sahel
  cluster reaches into the Sahara); check the mm/month figure too.
- **Soil water** is analysed only within 2012-01 to 2024-06, for the reason given above.

## NASA data & references

**NASA data**
- Beaudoing, H. and M. Rodell, NASA/GSFC/HSL (2020), *GLDAS Noah Land Surface Model L4 monthly 0.25 x 0.25 degree
  V2.1*, Greenbelt, MD, NASA Goddard Earth Sciences Data and Information Services Center (GES DISC).
  https://doi.org/10.5067/SXAVCZFAQLNO
- GES DISC, *README Document for NASA GLDAS Version 2 Data Products* (README_GLDAS2), the source for the forcing
  changes. https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/doc/README_GLDAS2.pdf
- Rodell, M., et al. (2004). The Global Land Data Assimilation System. *BAMS*, 85(3), 381-394.
  https://doi.org/10.1175/BAMS-85-3-381
- Huffman, G. J., et al. (2001). Global precipitation at one-degree daily resolution from multisatellite
  observations. *J. Hydrometeorology*, 2(1), 36-50 (GPCP 1DD, the source behind the GLDAS-2.1 precipitation forcing).
  https://doi.org/10.1175/1525-7541(2001)002%3C0036:GPAODD%3E2.0.CO;2
- NASA Earthdata (data access), https://urs.earthdata.nasa.gov, via the `earthaccess` library.

**Methods**
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *J. R. Stat. Soc. B*, 57(1), 289-300.
- Santer, B. D., et al. (2000). Statistical significance of trends and trend differences in layer-average
  atmospheric temperature time series. *J. Geophys. Res.*, 105(D6), 7337-7356.
- Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling (intersection-union test).
  *Technometrics*, 24(4), 295-300.
- Brown, M. B. & Forsythe, A. B. (1974). Robust tests for the equality of variances. *JASA*, 69(346), 364-367.
- Anselin, L. (1995). Local indicators of spatial association (LISA). *Geographical Analysis*, 27(2), 93-115.
- Getis, A. & Ord, J. K. (1992). The analysis of spatial association by use of distance statistics.
  *Geographical Analysis*, 24(3), 189-206.
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: a new approach (EGARCH). *Econometrica*,
  59(2), 347-370.

**Literature cited on the Discovery Cards** (for the "agreement with published literature" check; several only
partly match a card's region, and the cards say so)
- Biasutti, M. (2019). Rainfall trends in the African Sahel. *WIREs Climate Change*, 10(4), e591. https://doi.org/10.1002/wcc.591
- Espinoza, J. C., et al. (2019). Contrasting North-South changes in Amazon wet-day and dry-day frequency.
  *Climate Dynamics*, 52, 5413-5430. https://doi.org/10.1007/s00382-018-4462-2
- Garreaud, R. D., et al. (2017). The 2010-2015 megadrought in central Chile. *HESS*, 21, 6307-6327. https://doi.org/10.5194/hess-21-6307-2017
- Garreaud, R. D., et al. (2020). The Central Chile Mega Drought (2010-2018). *Int. J. Climatol.*, 40(1), 421-439. https://doi.org/10.1002/joc.6219
- Hua, W., et al. (2016). Possible causes of the Central Equatorial African long-term drought. *ERL*, 11, 124002. https://doi.org/10.1088/1748-9326/11/12/124002
- Jin, Q. & Wang, C. (2017). A revival of Indian summer monsoon rainfall since 2002. *Nature Climate Change*, 7, 587-594. https://doi.org/10.1038/nclimate3348
- Joseph, L., et al. (2024). Increased summer monsoon rainfall over Northwest India. *GRL*, 51, e2024GL108829. https://doi.org/10.1029/2024GL108829
- Marengo, J. A., et al. (2018). Changes in climate and land use over the Amazon region. *Front. Earth Sci.*, 6, 228. https://doi.org/10.3389/feart.2018.00228
- Marengo, J. A., et al. (2022). Increased climate pressure on the agricultural frontier in the Eastern
  Amazonia-Cerrado transition zone. *Sci. Rep.*, 12, 457. https://doi.org/10.1038/s41598-021-04241-4
- Ullah, W., et al. (2023). Increasing trend in daily monsoon precipitation extremes over Pakistan. *Front. Environ. Sci.*, 11, 1228817. https://doi.org/10.3389/fenvs.2023.1228817
- Wu, K., et al. (2022). The strengthening relationship between summer rainfall over North China and PDO since the
  mid-2000s. *Front. Earth Sci.*, 10, 957940. https://doi.org/10.3389/feart.2022.957940
- Zhou, L., et al. (2014). Widespread decline of Congo rainforest greenness in the past decade. *Nature*, 509, 86-90. https://doi.org/10.1038/nature13265

**Software:** NumPy, pandas, SciPy, xarray, netCDF4, statsmodels, arch, PySAL (esda, libpysal), scikit-learn,
matplotlib, earthaccess, ffmpeg, globe.gl / three.js (explorer), headless Chrome (slides PDF). PyTorch is used only by an
experimental forecaster that did not contribute to any reported result. No NASA logos are used. The globe
texture is drawn from our own results.

## AI assistance

An AI coding assistant was used to help write and review code and to draft explanatory text (README, Discovery
Card text, slide and video copy). Every number is computed by the code from NASA GLDAS-2.1 data. The full
disclosure, including the statement on who conceived the idea, is in [docs/SUBMISSION.md](docs/SUBMISSION.md)
(section 6, to be completed and confirmed by the team before submission).

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests                  # 38 tests
.venv/bin/python run.py --synthetic --out outputs    # whole pipeline on synthetic GLDAS-like data, no download

# 1. NASA data (free Earthdata login, see DATA_TO_USE.md): 312 global monthly files, ~7.8 GB
.venv/bin/python scripts/download_data.py gldas --start 2000-01 --end 2025-12

# 2. global run (as used for the submitted results); run_capped.py kills the job if the process tree exceeds 10 GB
.venv/bin/python scripts/run_capped.py --cap-gb 10 --log data/run_global.log -- \
  .venv/bin/python -u run.py --raw data/raw/gldas --tile 1.0 --ref 2000-01 2017-12 --no-pinn --out outputs_global_v2

# 3. export for the explorer, then build cards, slides and video (every number is read from site/data)
.venv/bin/python scripts/export_site.py --out outputs_global_v2 --site site
.venv/bin/python scripts/build_cards.py      # site/data/cards.json (words from scripts/cards_curation.json)
.venv/bin/python scripts/build_slides.py     # demo/slides.html + demo/paradox_earth_slides.pdf (headless Chrome)
.venv/bin/python scripts/build_video.py      # demo/paradox_earth_30s.mp4 (ffmpeg)

# view the explorer locally
npx http-server . -p 8800    # then open http://localhost:8800/site/
```

---

# For developers

This repo holds the scientific backend: NASA GLDAS ingestion and QC, seasonal anomalies, calibrated trend and
variance-change tests, ARIMA-EGARCH, §7 spatial statistics, the flagship headline summary, and an optional
physics-informed water-balance forecaster (PINN). It also holds the static explorer (`site/`) and the demo builders
(`scripts/build_*.py`). The data sources are listed in **[DATA_TO_USE.md](DATA_TO_USE.md)**.

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests                  # 38 smoke/equivalence/regression/calibration tests
.venv/bin/python run.py --synthetic                  # full pipeline on synthetic GLDAS-like data, ~1 min

# real data (needs Earthdata login, see DATA_TO_USE.md)
.venv/bin/python scripts/download_data.py gldas --start 2000-01 --end 2025-12        # global files, ~7.8 GB
.venv/bin/python run.py --raw data/raw/gldas --tile 1.0 --ref 2000-01 2017-12 --no-pinn --out outputs_global_v2  # global
.venv/bin/python run.py --raw data/raw/gldas --bbox -125 24 -66 50 --tile 1.0 --out outputs_conus   # CONUS dev loop
```

Global runs stream one file at a time (~0.5 s/file, ~1-2 GB RAM), so a 16 GB laptop is enough for ingest and
statistics. Per-region statistics run on `--workers` processes (default: all cores). EGARCH dominates the time per
series. `scripts/run_capped.py --cap-gb N --log FILE -- <cmd>` wraps a long run. It adds up the resident memory of
the whole process tree (macOS ignores `ulimit -v`) and kills the process group if the total goes over the cap.
`--pinn-coarsen 4` trains the forecaster on a 1° grid. Native 0.25° global training needs a GPU and is not wired up
yet (`PINNConfig.device` is unused). `--no-pinn` skips it. The submitted run used `--no-pinn`.

## What is built (Manifest section → module)

| Manifest | Module | What it does |
|---|---|---|
| §4.1, §5 data contract | `gldas.py`, `contract.py` | Opens monthly GLDAS files with an optional bbox. Converts units: storage in mm, fluxes in mm/month totals. Derives soil water 0-200 cm, land-water store and net flux P−ET−R. Builds land and land-ice masks from the first month. Ice cells sit at Noah's constant saturation and are excluded from `ICE_MASKED` variables. Streams files one at a time into grid-aligned tiles with a coverage threshold, and stores each tile's true valid land area (`area_km2__<var>`). Writes the source manifest and the §16 analysis records. |
| S1 QC | `qc.py` | Chronology checks (gaps, duplicates, order), units present, missingness on land cells, physical-range checks. Never modifies the data. |
| S1 homogeneity | `pipeline.py` | Documented GLDAS-2.1 radiation-forcing changes (README_GLDAS2: 2010-08 AGRMET shift, 2024-07 WWMCA degradation, 2025-06 GALWEM). Model states inherit these steps, so `ANALYSIS_WINDOWS` restricts soil water and the land-water store to 2012-01..2024-06. Precipitation (GPCP-based) shows no step and uses the full record. |
| S2 seasonal | `seasonal.py` | Monthly climatology fitted on a reference (training-only) window. Keeps raw, climatology and anomaly series side by side. |
| S3 trend | `trend.py` | `y = a + b·t + Σ month dummies`, OLS slope per decade. Inference is an intersection-union test: p = max(AR(1)-parametric SE with bias-corrected ρ on t(n−k), annual-mean OLS with AR(1) n_eff per Santer et al. 2000). The CI is the envelope of the two CIs. Simulated size is 3-6% on AR(1) nulls (φ 0-0.95) and on real-residual surrogates. The Newey-West HAC test it replaced had 29-34% on soil water. `slope_difference` compares slopes between two regions directly. |
| S4 variability | `variability.py` | The **test** is a prewhitened, state-adjusted Brown-Forsythe test: AR(p) innovations with trend and month dummies, then a regression of absolute deviations on a late-half dummy, month dummies and the lagged level, with an ACF-deflated t. Simulated size is 3.9-6.1%, against 43-64% for raw Brown-Forsythe at high persistence. The descriptive outputs are the raw variance ratio, the MAD ratio and the rolling variance. Then ARIMA(p,0,q)+trend chosen by AIC, ARCH-LM, and EGARCH(1,1) via `arch`, behind an **eligibility gate**: `fitted`, `inconclusive` or `unavailable`, with the failed gates listed. |
| §7 spatial | `spatial.py` | On each per-tile map (trend slope, log variance ratio): global Moran's I; local Moran's I (LISA) with two-sided conditional-permutation p-values (9,999 by default) and Benjamini-Hochberg FDR. Labels: `HH_cluster` / `LL_cluster`, `HL_outlier` / `LH_outlier`, `inconclusive`, `no_neighbors`. Primary weights are queen contiguity on the tile lattice, wrapping at ±180° on global runs. Robustness checks: kNN (k=8, on the sphere) labels and Getis-Ord Gi* z. Fills `record["spatial"]`. |
| §11, §13 headline | `headline.py` | Flagship summary for `FLAGSHIP = "Rainf_f_tavg"`. Computes the land-area-weighted aggregate series and its trend, per-tile BH-FDR shares of tiles and of land area (wetting and drying), and connected regions (8-neighbour, dateline-aware, at least 5 tiles) that are LISA HH/LL and FDR-significant with a matching sign. It repeats this for the sensitivity windows 2003-2025 and 2000-2020. Writes `headline_<var>.json`. Claim level is statistical, never causal. |
| §10 PINN | `pinn.py` | Spatiotemporal CNN over the last *k* months of weather and state [S, P, ET, R, T]. It predicts next-month [ΔS, P, ET, R] with loss L = L_pred + λ·L_water-balance (ΔS = P − ET − R, with `centered` alignment for GLDAS monthly means). Normalisation uses training data only. Includes a chronological split, a spatial holdout, persistence / seasonal-naive / linear-AR baselines, and a λ ablation. Optional; not part of the submitted results. |
| §15.6 flow | `pipeline.py`, `run.py` | ingest → stats → spatial → headline → pinn. Writes everything to `--out`. `--spatial-only` re-runs §7 on an existing `--out`. |

Outputs: `qc_report.json` (including `masks`), `manifests/`, `regional_<var>.csv`, `tiles.csv` (tile boxes, land/ice
fractions, area weights), `records.json` (§16 records), `stats_summary.csv`, `volatility/<var>.csv` (EGARCH-`fitted`
tiles only), `spatial/<var>__<field>.csv`, `spatial_summary.json`, `headline_Rainf_f_tavg.json`, `run_config.json`,
and with the PINN `pinn_ablation.json` and `models/pinn_lambda=*.pt`. All JSON is strict (no NaN). The final
submitted run is `outputs_global_v2/`. All `outputs*/` directories are git-ignored.

## Site and demo builders

| Script | Reads | Writes |
|---|---|---|
| `scripts/export_site.py --out outputs_global_v2 --site site` | a run's `headline_<var>.json`, `stats_summary.csv`, `tiles.csv`, `spatial/`, `regional_<var>.csv`, `run_config.json`, `qc_report.json` | `site/data/meta.json`, `tiles.json`, `regions.json`, `series/<10° block>.json` |
| `scripts/build_cards.py` | `site/data/*`, `scripts/cards_curation.json` (words only: place names, users, literature) | `site/data/cards.json`, with a self-computed robustness checklist per card |
| `scripts/build_slides.py [--no-pdf]` | `site/data/*`, `demo/team.json` | `demo/slides.html`, `demo/paradox_earth_slides.pdf` (headless Chrome) |
| `scripts/build_video.py [--stills t ...]` | `site/data/*`, `demo/team.json` | `demo/paradox_earth_30s.mp4` (28 s, 1920×1080, ffmpeg) |

The explorer (`site/index.html`, `app.js`, `style.css`; the root `index.html` redirects there) computes nothing
statistical. It only displays the exported JSON. `demo/team.json` holds the team name and members, which the team
fills in before submission.

Spatial caveats (§7.4-7.5): the permutation null shuffles values across the map, so it ignores the smooth large-scale
structure of GLDAS forcing. Neighbouring LISA tests share data, so BH q-values are a documented convention, not a
guarantee. Dependence-preserving nulls are Experiment E. With 999 permutations BH has almost no power at these tile
counts (esda's two-sided p floor is ~3/(R+1)), so keep the 9,999 default for any map you report.

## Partner handoff: §8-§9 (after spatial, before the PINN)

Not built yet: **ML discovery** (PCA, Isolation Forest, 1D-CNN encoder, contrastive matching, ablation) and
**cross-lens analysis**. Plug-in points:
- Input: `<out>/records.json` (one record per region × variable, `spatial` already filled),
  `<out>/regional_<var>.csv` and `<out>/spatial/*.csv`.
- Fill `record["ml"]` (`ranking_score`, `model_version`).
- Tiles have ids `tile_<south>_<west>`, aligned to multiples of the tile size. Boxes are in `<out>/tiles.csv`.
  `spatial.lattice_neighbors` / `spatial.knn_neighbors` build neighbour lists from them.
- Use the regional anomaly series (`seasonal.anomalies`) as ML inputs. Train only on the early period.

## Scientific guardrails (from the manifest)
- GLDAS is land-surface model output, not independent observations. Its precipitation is GPCP-based forcing.
- Trends are statistical and noncausal. Never claim that "rainfall is increasing globally": the aggregate trend is not
  detectable, and the regional shares are small and depend on the window.
- Model states (soil water, land-water store) are analysed only within one forcing regime (2012-01..2024-06).
- The PINN target is soil + snow + canopy storage, **not** terrestrial water storage. GRACE is a complementary
  check, not ground truth.
- A fitted EGARCH volatility path does not by itself prove a persistent change in variance. If the gate fails,
  report the empirical baseline instead.
- Missing data stays missing in every statistical step.
- The forecasting extension ships only if it passes Gate G (λ ablation + baseline comparison documented).

Reviewer agent for code changes: `.claude/agents/code-reviewer.md` (lean, correctness only).
