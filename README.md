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
  passes by construction. The graded checks are four more: stability across overlapping time windows (2000-2025,
  2003-2025, 2000-2020), agreement between two definitions of "neighbouring tile" (same data, different spatial
  weights), agreement with an independent precipitation product (CHIRPS, via NASA FLDAS), and agreement with
  published studies. Central Sahel, northern China and northwest India pass all four; the southern Andes and the
  southwestern Amazon pass 3 of 4; the middle Zambezi 2 of 4. Every number on the page is read from the exported
  data. None is typed in by hand.
- **Calibrated statistics.** Monthly climate series are strongly autocorrelated, which makes standard trend tests
  "find" trends far too often. We tested our trend and variance tests on simulated no-trend data. They give false
  positives about 3-6% of the time at a nominal 5%. The Newey-West test we used before gave 29-34% on soil water.
- **It caught its own false trend.** Our first headline was a 26-year global decline in soil water. The pipeline's
  own checks showed that this was mostly two level shifts. Their timing lines up with documented changes in the
  radiation data that drive the GLDAS model: a step in 2010-08 and a degraded period from 2024-07 (GES DISC
  README_GLDAS2). Within the one consistent period, 2012-01 to 2024-06, the global soil-water decline is not
  significant. So we withdrew that trend. Soil water is analysed only inside that window. We switched the
  flagship variable to precipitation, which in GLDAS-2.1 comes from a different forcing source (GPCP-based) than the
  radiation fields that changed. A controlled comparison with GLDAS-2.0 and a second NASA land model (FLDAS) then
  backed this up (see [Soil water: the decline that wasn't](#soil-water-the-decline-that-wasnt)).
- **Checked against independent data.** The precipitation trends are compared with CHIRPS-based precipitation from
  NASA FLDAS, and the soil-water story with GRACE/GRACE-FO satellite gravity (see
  [Independent data checks](#independent-data-checks)).
- **A physics-informed forecast lab.** A small CNN emulates next month's GLDAS land water store with a
  water-budget penalty, tested on held-out years, a held-out region and an out-of-regime period, against a linear
  model with the same inputs (see [Forecast lab](#forecast-lab-a-physics-informed-emulator)). It is an emulator of
  a NASA model, not a forecast of the real world.

## How it works

1. **NASA data.** 312 monthly global files (2000-01 to 2025-12) of
   [GLDAS-2.1 Noah, 0.25°, monthly](https://doi.org/10.5067/SXAVCZFAQLNO) (`GLDAS_NOAH025_M`, DOI
   10.5067/SXAVCZFAQLNO) from NASA GES DISC, downloaded with a free Earthdata login. Checks use GLDAS-2.0, NASA
   FLDAS and GRACE/GRACE-FO (see [NASA data & references](#nasa-data--references)).
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
7. **Independent checks.** The same pipeline runs on NASA FLDAS (CHIRPS precipitation) for a cross-check of the
   precipitation trends; GLDAS-2.0 and GRACE/GRACE-FO test the soil-water story.
8. **Export.** Results go to static JSON for the explorer, the Discovery Cards, the slides and the video.

## Independent data checks

GLDAS-2.1 precipitation is GPCP-based forcing, so a trend in it could be a feature of one product. We re-ran the
same trend pipeline on **NASA FLDAS** (Noah forced by MERRA-2 and **CHIRPS** precipitation, a separate
satellite-plus-gauge product; McNally et al. 2017, Funk et al. 2015) and compared the two over their common years,
2003-2024 (15,690 one-degree tiles).

- **Sign agreement where it matters.** Where either product has an FDR-significant trend, the two agree in sign
  over **87%** of that land area; where both do (40 tiles), over **94%**. Over all tiles, trends agree in sign over
  69% of land area and the tile-by-tile rank correlation of slopes is **Spearman 0.49**.
- **Few detectable tiles in either product over this shorter window.** 2.4% of land area is FDR-significant in
  GLDAS-2.1 and 1.2% in FLDAS for 2003-2024, consistent with the shrinking shares in shorter windows noted under
  Limits.
- **What it means.** Where a trend is detectable, two products built from different precipitation sources mostly
  point the same way. They do not agree on size, and moderate rank correlation means the full maps differ. So this
  check raises confidence in the direction of the highlighted regions, not in their exact magnitude.
- **Per region (on the Discovery Cards).** CHIRPS shows the same sign in all tiles of the Central Sahel, northern
  China, northwest India and southern Andes regions, with a majority of tiles individually significant (pass). The
  southwestern Amazon (92% same sign, 25% significant) and the middle Zambezi (100% same sign, none significant)
  are graded partial; CHIRPS trends there are smaller than GLDAS's.

**GRACE/GRACE-FO** (JPL mascons RL06.3Mv04, 2003-2024) measures total terrestrial water storage: soil, groundwater,
snow, surface water and ice. It can test signs and patterns of the land model's soil water, not its sizes.
FDR-significant storage trends cover 40% of land area. Where either GRACE or FLDAS soil water has a significant
trend, they agree in sign over 61% of the area (Spearman 0.26). The largest disagreements line up with groundwater
and irrigation hotspots that the land model does not represent (next section). A separate Discovery Card shows
northwest India: GRACE -204 mm/decade, FLDAS soil water +34 mm/decade.

## Soil water: the decline that wasn't

GLDAS-2.1 soil water (0-200 cm) appears to decline over 2000-2025. A published study (Qin et al. 2023) reported a
GLDAS-Noah soil-moisture decline over 2000-2020. Our analysis indicates that the post-2010 part of the GLDAS-2.1
decline lines up with documented changes in the model's radiation forcing. Three checks, all noncausal:

1. **Artifact control: same model, different forcing.** GLDAS-2.1 and GLDAS-2.0 run the same Noah model on the same
   grid but with different forcing (2.1: GDAS + GPCP precipitation + AGRMET radiation; 2.0: Princeton reanalysis-based
   forcing). Over their common years, 2000-2014, incoming solar radiation steps up at 2010-08 by **+8.8 W/m²**
   (95% CI 6.0 to 11.5) in 2.1 and by **-1.1 W/m²** (-2.7 to +0.4) in 2.0. Evaporation responds only in 2.1. The
   2011-2013 soil-water dip appears in **both** versions, deeper in 2.1. Our reading: the forcing change exaggerated
   a real dip into an apparent long-term decline. Over 2000-2014 neither version's soil-water trend is detectable
   (2.1: -6.2 mm/decade, p = 0.08; 2.0: -3.5 mm/decade, p = 0.31). Figure: [demo/artifact_control.png](demo/artifact_control.png).
2. **A second NASA land model with steady forcing.** NASA FLDAS (MERRA-2 + CHIRPS forcing), 2003-2024: global land
   soil water **+5.0 mm/decade, 95% CI -1.6 to +11.6, p = 0.14**, so no detectable trend. 2.8% of land area shows a
   detectable wetting trend and 0.7% a detectable drying trend after FDR correction.
3. **GRACE hotspots.** Several of the strongest GRACE declines sit in groundwater and irrigation hotspots, which the
   land model does not simulate (FLDAS soil water, 2003-2024): northwest India GRACE **-204** vs soil model
   **+34 mm/decade**, North China Plain **-108** vs **+9**, California Central Valley **-108** vs **-4**. A real
   water loss in these places is consistent with GRACE studies (Rodell et al. 2009, 2018), but it lines up with
   storage outside the soil layer the model tracks (groundwater; near the Himalaya, possibly glacier ice leaking
   into the mascons).

So soil water stays withdrawn as a 2000-2025 trend and is analysed only inside the single forcing regime
2012-01..2024-06. Studies that compare soil-moisture products (e.g. Hirschi et al. 2025, satellite and reanalysis
products) examine how far long-term soil-drying trends agree between datasets; our contribution is narrower: test a
model record for forcing breaks before trusting a trend in it.

## Forecast lab: a physics-informed emulator

**What it is.** A small convolutional network (CNN) that emulates NASA GLDAS-2.1. From the last 6 months of the land
water store and fluxes, plus next month's rain and temperature, it predicts next month's **change in the GLDAS land
water store** (soil 0-200 cm + snow water + canopy water; this is *not* total water storage) together with
evaporation and runoff. The loss adds a water-budget penalty: L = L_pred + λ·(ΔS − (P − ET − R))², with fluxes
averaged over adjacent months because GLDAS fluxes are monthly means. It runs at 1°
(16,598 land cells) over 2012-01..2025-05, i.e. within one radiation-forcing regime plus the period after it.
Skill is measured against GLDAS model output. **It is emulation skill, not real-world forecast skill.**

**Design.**
- Splits by target month: train 2012-2019, validation 2020-2021, **test 2022-01..2024-06** (same forcing regime).
- **Held-out region:** the eastern quarter of the grid (90°E-180°E) is never in the training loss; a 5-column guard
  band separates it from the training region.
- **Out-of-regime test:** 2024-07..2025-05, when GLDAS-2.1's radiation forcing is degraded (README_GLDAS2). Never
  trained on; compared with the same calendar months one year earlier (2023-07..2024-05).
- **Grid:** 2 input arms (rain + temperature known; plus radiation known) × λ ∈ {0, 0.1, 1, 10} × 5 seeds = 40 runs,
  60 epochs each, model selected by validation loss with the same rule for every λ. Trained on a RunPod RTX 4090
  (about 1 GPU-hour in total).
- **Baselines:** persistence (no change), climatological increment (the training-period mean change for each cell
  and calendar month) and a **linear autoregressive model with the same inputs**, the strongest of the three.
  Skill = MSESS = 1 − MSE/MSE_reference on the storage change (predicting the storage level itself is trivial).
- **Truth floor:** GLDAS's own budget residual under the monthly-mean approximation, ~8.5 mm/month on the test
  months, shows how closely the monthly budget can close at all.

**Results** (mean ± sd over 5 seeds; source `site/data/pinn.json`):

| Inputs | λ | Skill vs linear AR: test, training region | Skill vs linear AR: test, held-out 90°E-180°E | Skill vs linear AR: out-of-regime 2024-07..2025-05 | ΔS RMSE, test (mm) | Budget residual, own fluxes (mm/month) | Budget residual, GLDAS fluxes (mm/month) |
|---|---|---|---|---|---|---|---|
| rain + temperature known | 0 | +0.20 ± 0.01 | +0.20 ± 0.01 | +0.12 ± 0.02 | 14.1 ± 0.1 | 7.5 ± 0.2 | 7.7 ± 0.05 |
| rain + temperature known | 0.1 | +0.22 ± 0.02 | +0.22 ± 0.03 | +0.13 ± 0.03 | 14.0 ± 0.2 | 6.5 ± 0.2 | 7.2 ± 0.09 |
| rain + temperature known | 1 | +0.21 ± 0.02 | +0.19 ± 0.02 | +0.14 ± 0.02 | 14.1 ± 0.2 | 4.2 ± 0.2 | 6.4 ± 0.06 |
| rain + temperature known | 10 | +0.01 ± 0.04 | -0.08 ± 0.07 | -0.04 ± 0.04 | 15.7 ± 0.4 | 2.7 ± 0.2 | 7.3 ± 0.3 |
| + radiation known | 0 | +0.18 ± 0.02 | +0.16 ± 0.03 | +0.10 ± 0.03 | 14.3 ± 0.2 | 7.5 ± 0.2 | 7.7 ± 0.09 |
| + radiation known | 0.1 | +0.19 ± 0.02 | +0.17 ± 0.04 | +0.11 ± 0.03 | 14.2 ± 0.2 | 6.7 ± 0.4 | 7.3 ± 0.08 |
| + radiation known | 1 | +0.19 ± 0.02 | +0.16 ± 0.05 | +0.11 ± 0.02 | 14.2 ± 0.1 | 4.4 ± 0.3 | 6.5 ± 0.2 |
| + radiation known | 10 | -0.01 ± 0.03 | -0.12 ± 0.06 | -0.04 ± 0.03 | 15.9 ± 0.2 | 2.8 ± 0.3 | 7.3 ± 0.1 |
| *linear AR, same inputs* | — | 0 | 0 | 0 | 15.8 | — | 8.4 |
| *GLDAS itself (truth floor)* | — | — | — | — | — | — | 8.5 |

- **Skill.** With rain and temperature known and λ ≤ 1, the CNN beats the linear AR by **+0.20 to +0.22** on the
  test months and **+0.19 to +0.22** in the held-out region it never trained on.
- **The physics weight has a sweet spot.** λ = 1 cuts the model's own water-budget residual from **7.5 to 4.2
  mm/month** with no loss of accuracy, and makes its storage change more consistent with GLDAS's own fluxes
  (**7.7 → 6.4 mm/month**, below GLDAS's own 8.5). λ = 10 closes the budget further (2.7) but skill collapses to
  about zero.
- **Radiation inputs do not help.** That arm is slightly worse in every configuration, and in the out-of-regime
  months (degraded radiation forcing) it loses more: training region +0.10 to +0.11 against +0.12 to +0.14 for the
  base arm; held-out region +0.01 to +0.04 against +0.06 to +0.11. This is consistent with the network leaning on
  an input whose record later changed.
- **Skill drops out of regime for every configuration** (base arm, training region: +0.17 to +0.18 in the same
  calendar months a year earlier, +0.12 to +0.14 after the forcing change).

Figures: [demo/pinn_tradeoff.png](demo/pinn_tradeoff.png) (budget residual against skill for every λ and seed) and
[demo/pinn_skill_map.png](demo/pinn_skill_map.png) (per-cell skill against the linear AR on the test months, base
arm, λ = 0.1).

**Limits.** This emulates a land model; it says nothing about how well real-world water storage can be forecast,
and it inherits every bias of GLDAS-2.1 forcing. The target is soil + snow + canopy, not total water storage, so it
cannot be compared with GRACE. Next month's rain and temperature are given as inputs (taken from GLDAS), so this is
not a weather forecast. The area-weighted skill hides large local differences: the skill map has many cells where
the linear model does better, and in deserts the storage barely changes, so per-cell skill there is unstable. One
model family, one resolution (1°), 5 seeds.

## Limits

- **Not causal.** These are statistical trends over 2000-2025. They do not say why precipitation changed, and they
  do not say the trends will continue.
- **A forcing dataset, not rain gauges.** GLDAS-2.1 precipitation is model *forcing* derived from GPCP. It is not an
  independent observation record. The CHIRPS cross-check (via FLDAS) supports the sign of the highlighted trends
  where either product is significant, but the two products often differ in size.
- **Results depend on the window.** In shorter windows, the FDR-significant shares shrink. For 2003-2025 they are
  1.3% wetting and 2.6% drying land area. For 2000-2020 they are 0.7% and 1.0%. The trends of the highlighted
  regions (OLS on the region's annual-mean anomalies) keep their sign and stay within a factor of 2 in every window.
- **The shares are small.** 24% of tiles have raw p < 0.05, against about 5% expected by chance. After FDR
  correction, only the 4.3% and 3.1% of land area above remain, and individual tiles can still be false positives.
- **Percentages in dry places.** Percent-per-decade changes look large where the baseline is small (e.g. the Sahel
  cluster reaches into the Sahara); check the mm/month figure too.
- **Soil water** is analysed only within 2012-01 to 2024-06, for the reasons given above. GRACE measures total
  water storage, so it can test signs but not confirm or refute a soil-only trend.
- **The forecast lab** reports emulation skill against a NASA model, not real-world forecast skill.

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
- Beaudoing, H. and M. Rodell, NASA/GSFC/HSL, *GLDAS Noah Land Surface Model L4 monthly 0.25 x 0.25 degree V2.0*
  (`GLDAS_NOAH025_M` v2.0, Princeton forcing), NASA GES DISC, used for the artifact control (2000-2014).
  https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH025_M_2.0/summary
- McNally, A., NASA/GSFC/HSL, *FLDAS Noah Land Surface Model L4 Global Monthly 0.1 x 0.1 degree (MERRA-2 and CHIRPS)*
  (`FLDAS_NOAH01_C_GL_M` v001), NASA GES DISC. https://doi.org/10.5067/5NHC22T9375G. CHIRPS precipitation enters
  through the FLDAS forcing; we did not download CHIRPS separately.
- NASA JPL, *GRACE and GRACE-FO JPL mascon RL06.3Mv04 CRI* (`TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4`), NASA
  PO.DAAC. https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4
- McNally, A., et al. (2017). A land data assimilation system for sub-Saharan Africa food and water security
  applications. *Scientific Data*, 4, 170012. https://doi.org/10.1038/sdata.2017.12
- Funk, C., et al. (2015). The climate hazards infrared precipitation with stations: a new environmental record for
  monitoring extremes. *Scientific Data*, 2, 150066. https://doi.org/10.1038/sdata.2015.66
- NASA Earthdata (data access), https://urs.earthdata.nasa.gov, via the `earthaccess` library.

**Soil water and groundwater**
- Qin, T., et al. (2023). Continued decline of global soil moisture content, with obvious soil stratification and
  regional difference. *Science of the Total Environment*, 864, 160982. https://doi.org/10.1016/j.scitotenv.2022.160982
- Hirschi, M., et al. (2025). Potential of long-term satellite observations and reanalysis products
  for characterising soil drying: trends and drought events. *Hydrology and Earth System Sciences*, 29, 397-425. https://doi.org/10.5194/hess-29-397-2025
- Rodell, M., Velicogna, I. & Famiglietti, J. S. (2009). Satellite-based estimates of groundwater depletion in India.
  *Nature*, 460, 999-1002. https://doi.org/10.1038/nature08238
- Rodell, M., et al. (2018). Emerging trends in global freshwater availability. *Nature*, 557, 651-659.
  https://doi.org/10.1038/s41586-018-0123-1

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
matplotlib, earthaccess, ffmpeg, globe.gl / three.js (explorer), headless Chrome (slides PDF), PyTorch (forecast lab
only; the trend results do not use it). GPU training on RunPod. No NASA logos are used. The globe texture is drawn
from our own results.

## AI assistance

An AI coding assistant was used to help write and review code, to design and run the calibration simulations and
the GPU forecast-lab experiment, and to draft explanatory text (README, Discovery Card text, slide and video copy).
Every number is computed by the code from NASA data (GLDAS-2.1, GLDAS-2.0, FLDAS, GRACE/GRACE-FO). The full
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

# 4. independent checks (run before step 3; export_site.py reads outputs_fldas, outputs_artifact, outputs_grace)
.venv/bin/python scripts/download_data.py fldas --start 2003-01 --end 2024-12      # NASA FLDAS (CHIRPS forcing)
.venv/bin/python scripts/shim_fldas.py --check                                      # -> data/raw/fldas_shim (0.5°, GLDAS layout)
.venv/bin/python run.py --raw data/raw/fldas_shim --dataset FLDAS_NOAH01_C_GL_M 001 --no-forcing-window \
  --no-egarch --tile 1.0 --ref 2003-01 2017-12 --no-pinn --out outputs_fldas
.venv/bin/python scripts/precip_cross_check.py --gldas outputs_global_v2 --fldas outputs_fldas --site site
.venv/bin/python scripts/download_data.py gldas20 --start 2000-01 --end 2014-12    # GLDAS-2.0 control
.venv/bin/python scripts/artifact_control.py                                        # -> outputs_artifact/, demo/artifact_control.png
.venv/bin/python scripts/download_data.py grace                                     # JPL mascon RL06.3Mv04 (one file)
.venv/bin/python scripts/grace_check.py
.venv/bin/python scripts/grace_check.py --compare outputs_fldas soil_water_0_200cm --window 2003-01 2024-12

# 5. forecast lab (GPU; ~1 GPU-hour on an RTX 4090 for 40 runs; run before the slide and video builds in step 3)
.venv/bin/python scripts/build_pinn_arrays.py --raw data/raw/gldas --coarsen 4 --out data/pinn/gldas_1deg_2012_2025.npz
.venv/bin/python scripts/run_pinn_experiment.py --arrays data/pinn/gldas_1deg_2012_2025.npz --out outputs_pinn \
  --device cuda --seeds 0 1 2 3 4
.venv/bin/python scripts/pinn_report.py --results outputs_pinn --site site --demo demo   # site/data/pinn.json + figures

# view the explorer locally
npx http-server . -p 8800    # then open http://localhost:8800/site/
```

---

# For developers

This repo holds the scientific backend: NASA GLDAS ingestion and QC, seasonal anomalies, calibrated trend and
variance-change tests, ARIMA-EGARCH, §7 spatial statistics, the flagship headline summary, and an optional
physics-informed water-balance emulator (forecast lab). It also holds the static explorer (`site/`) and the demo builders
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
`--pinn-coarsen 4` trains the emulator on a 1° grid. `--no-pinn` skips it; the submitted trend run used
`--no-pinn`. The reported forecast-lab results come from the standalone `scripts/run_pinn_experiment.py` (GPU, 40
runs), not from `run.py`.

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
| §10 PINN | `pinn.py` | Spatiotemporal CNN over the last *k* months of weather and state [S, P, ET, R, T]. It predicts next-month [ΔS, P, ET, R] with loss L = L_pred + λ·L_water-balance (ΔS = P − ET − R, with `centered` alignment for GLDAS monthly means). Normalisation uses training data only. Includes a chronological split, a spatial holdout, persistence / seasonal-naive / linear-AR baselines, and a λ ablation. Used by `scripts/run_pinn_experiment.py` for the forecast lab (emulation skill against GLDAS, not real-world skill); not used by the trend results. |
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
| `scripts/precip_cross_check.py` | `outputs_global_v2/`, `outputs_fldas/`, `site/data/regions.json` | `outputs_fldas/precip_cross_check.json` (exported to `site/data/crosscheck.json` and `meta.json` `cross_checks`) |
| `scripts/artifact_control.py` | `data/raw/gldas`, `data/raw/gldas20` (2000-2014) | `outputs_artifact/artifact_control.json`, `demo/artifact_control.png` |
| `scripts/grace_check.py [--compare OUT VAR --window A B]` | `data/raw/grace`, a run's outputs | `outputs_grace/*.json`, `*.csv` |
| `scripts/pinn_report.py` | `outputs_pinn/results.json`, `skill_maps.npz` | `site/data/pinn.json`, `demo/pinn_tradeoff.png`, `demo/pinn_skill_map.png` |

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
- The forecasting extension ships only if it passes Gate G (λ ablation + baseline comparison documented). Both
  are in `site/data/pinn.json` and the Forecast lab section above.

Reviewer agent for code changes: `.claude/agents/code-reviewer.md` (lean, correctness only).
