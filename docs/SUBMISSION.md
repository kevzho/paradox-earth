# Paradox Earth: Space Apps 2026 project-page text

Ready-to-paste text for each field of the NASA Space Apps Challenge 2026 project submission form.
Challenge: **Be An Earth System Trend Detective!**

Every number below comes from `site/data/meta.json` and `site/data/cards.json`, which `scripts/export_site.py` and
`scripts/build_cards.py` export from `outputs_global_v2/`. If the pipeline is re-run, re-check the numbers before
pasting. Lines marked **[TEAM]** must be filled in or confirmed by the team, not by a tool.

---

## Before you submit (checklist)

- [ ] **Team details.** Fill `demo/team.json` (team name, members exactly as on the Space Apps team page, local
      event). Re-run `scripts/build_slides.py` and `scripts/build_video.py`, then check that the title slide and the
      video end card show the real names and no "(confirm)" text is left.
- [ ] **Slide 6 bug.** The PDF currently prints a raw Python dict (`{'breakpoints': ['2010-08', ...], 'text': ...}`)
      under "The detective that checks itself". `build_slides.py` expects `meta.json` `retraction` to be a string,
      but it is now an object. Make the builder use `retraction["text"]`, rebuild, and re-check all 7 pages.
- [ ] **Slide 6 fake shifts.** The slide-6 figure draws four "shift" lines (2010-08, 2012-01, 2024-06, 2024-07).
      Only 2010-08 and 2024-07 are breaks; 2012-01..2024-06 is the analysis window. Take the breaks from
      `meta.json` `retraction.breakpoints` instead of a regex, rebuild, re-check.
- [ ] **Chile card names the wrong place.** The `drying-chile_megadrought` cluster spans 34-54°S (centroid ~44°S);
      only 12 of 64 tiles are in the 30-38°S megadrought core and 7 are in Argentina. Rename title/place/user in
      `scripts/cards_curation.json` (e.g. "Drying along the southern Andes", "South-central Chile to southern
      Patagonia") and the slide-3 "Central Chile" label, then rebuild cards, slides and video. This page already
      uses the corrected wording.
- [ ] **Overclaiming copy on other surfaces.** Make the site, slides and video match this page: tagline "The global
      average looks flat. A few regions are not." (not "The planet is changing"); "shows a detectable wetting/drying
      trend (2000-2025)" (not "is getting wetter/drier"); no "opposite trends cancel in the global mean"; the
      soil-water card should say the shifts "line up with" forcing changes, not "caused by".
- [ ] **Check counts.** Cards say "3 of 3 independent checks" while the explorer badge says "4 / 4". The FDR row is
      the selection criterion and passes by construction, and the window/neighbour checks reuse the same data.
      Use one count ("3 of 3 extra checks, plus FDR selection") and drop "independent".
- [ ] **References and AI line on the slides.** Slide 7 needs a short reference list (or "full list in the
      explorer's case files / README") and an AI-acknowledgement line.
- [ ] **AI label on slides.** The video has a visible "AI-assisted production" label and the site has an "AI
      disclosure" section. The slides have no AI-assistance note yet. Add a one-line footer to the slides so the
      SI disclosure below is accurate, or remove the slides from that sentence.
- [ ] **Local card.** The `local-placeholder` card in `cards.json` still says "TODO". Set `local.bbox` in
      `scripts/cards_curation.json` and re-run `scripts/build_cards.py`, or drop the card before publishing.
- [ ] **Video.** Upload `demo/paradox_earth_30s.mp4` (28 s, under the 30 s limit) as **Public** or **Unlisted**
      (YouTube, or a public Google Drive / GitHub link). Paste the URL into the Demo field below.
- [ ] **Repo and site.** Make `github.com/kevzho/paradox-earth` public. Turn on GitHub Pages from branch `main`,
      folder `/ (root)`. Commit `site/`, `demo/`, `index.html` and `docs/` first; they are untracked right now.
- [ ] **Link test.** In a private/incognito window, open the site, the slides PDF link, the repo and the video.
      Click one tile and one Discovery Card on the site.
- [ ] **SI disclosure.** Read the "Use of Super Intelligence" section with the whole team and complete the
      **[TEAM]** line. Do not tick it unless it is true.
- [ ] **Form checkboxes.** Tick both required checkboxes on the submission form (rules and terms acknowledgement).
- [ ] **Deadline.** Submit before **23:59 local time** today. Submit early and edit afterwards if the form allows it.

---

## 1. Project Name

```
Paradox Earth
```

Tagline (if the form has room): *The global average looks flat. A few regions are not.*

---

## 2. High-Level Project Summary

```
Paradox Earth is a trend detective for NASA land data. It takes 26 years (2000–2025) of monthly GLDAS-2.1
precipitation and tests every 1° land tile, 17,284 of them, with an autocorrelation-aware trend test and a
false-discovery correction. The result is a paradox. Averaged over all land, precipitation shows no detectable
trend (−0.88 mm/month per decade, p = 0.08). Underneath, 4.3% of land area shows a detectable wetting trend and
3.1% a detectable drying trend. These are small, window-sensitive shares, so every finding comes as a Discovery
Card. Each card has a robustness checklist, published literature, caveats and a named user, such as a
hydropower planner on the Zambezi or an extension office in the Sahel. The detective also checks itself: it
caught and withdrew a dramatic soil-water decline whose level shifts line up with documented changes in the
model's radiation input. A 3D explorer lets anyone click a place and read its case file.
```

(~155 words.)

---

## 3. Link to Project Demo

```
Video (28 s): [VIDEO URL — paste after uploading demo/paradox_earth_30s.mp4]
Slides (7 slides, PDF): https://github.com/kevzho/paradox-earth/blob/main/demo/paradox_earth_slides.pdf
```

---

## 4. Link to Final Project

```
Live explorer: https://kevzho.github.io/paradox-earth/site/
Code and data pipeline: https://github.com/kevzho/paradox-earth
```

---

## 5. Detailed Project Description

```
WHAT IT DOES

Paradox Earth looks for trends in NASA's GLDAS-2.1 land data and reports only the ones that survive checking.
We started from one question: when land precipitation looks stable on average, are there regions underneath
where a trend is still detectable, and in which direction?

Over 2000–2025, land-area-weighted global land precipitation shows a trend of −0.88 mm/month per decade
(−1.3% per decade of a 69.9 mm/month mean; 95% CI −1.88 to +0.12; p = 0.08). That is NOT a detectable trend.
Tile by tile, we tested 17,284 one-degree land tiles (133 million km²). After Benjamini–Hochberg false-discovery
correction (q < 0.05), 4.3% of land area (5.7 million km²) shows a detectable wetting trend and 3.1%
(4.2 million km²) a detectable drying trend. Without correction, 24% of tiles have p < 0.05, against about 5%
expected by chance. We report the corrected shares.

We do not overstate this. The shares depend on the window. With the same method they shrink to 1.3% wetting and
2.6% drying for 2003–2025, and to 0.7% and 1.0% for 2000–2020. The trends of the highlighted regions (OLS on
each region's annual-mean anomalies) keep their sign and stay within a factor of 2 across those windows. Paradox Earth does not say rainfall is increasing or decreasing
globally, and it does not identify causes. It reports statistical trends in a GPCP-based forcing dataset.

Coherent clusters become Discovery Cards. Each card lists the trend, the area, a robustness checklist, published
literature that agrees or only partly agrees, caveats, and a named user with a concrete decision. FDR
significance is how regions are selected, so it passes by construction; the graded checks are stability across
overlapping time windows, agreement between two neighbour definitions, and agreement with the literature.
Example cards:
 • Central Sahel (Chad, eastern Niger), wetting: an agricultural extension office choosing millet/sorghum
   varieties and sowing windows.
 • Northern and northeastern China and eastern Mongolia (38–51°N), wetting: a water-resources bureau in Inner
   Mongolia or northeast China (e.g. the Liao River basin) setting reservoir rules, groundwater quotas and urban
   flood drainage.
 • Northwest India and Pakistan, wetting: a district disaster-management and irrigation office in Rajasthan or
   Sindh deciding between flood drainage and drought relief.
 • Southern Andes, south-central Chile to southern Patagonia (34–54°S, Chile and western Argentina), drying: a
   hydropower or water planner for the Biobío–Aysén basins. Only 12 of the 64 tiles lie in the 30–38°S core of
   the central-Chile megadrought.
 • Southwestern Amazon (Rondônia and southern Amazonas, Brazil), drying: an agricultural climate-risk zoning analyst
   deciding which second-crop (safrinha) maize plantings are high-risk.
 • Middle Zambezi (Lake Kariba), drying: a hydropower planner balancing generation against storage.
 • Teachers and students: the soil-water card shows why every trend needs a homogeneity check.

THE DETECTIVE CHECKS ITSELF

Our first headline was a 26-year global decline in soil water. Our own break tests showed it was mostly two
level shifts, at 2010-08 and 2024-07. Their timing lines up with documented changes in the radiation forcing
that drives the GLDAS model (GES DISC README_GLDAS2). Within the one consistent period, 2012-01 to 2024-06, the global decline
is not significant. We withdrew the trend and kept the case as a card: a step in the input data is not a trend.

HOW IT WORKS

1. Ingest: a Python pipeline streams 312 monthly GLDAS-2.1 files one at a time under a memory cap. It converts
   units, applies a land mask (plus a glacier/ice-sheet mask for soil water), and aggregates the 0.25° grid to
   1° land tiles weighted by true land area.
   Quality control covers chronology (no missing or duplicate months) and value ranges.
2. Trend test: an OLS regression of each monthly series on a linear trend plus month-of-year terms. The
   p-value comes from an intersection–union test. It takes the larger of (a) the p-value from an AR(1)
   parametric standard error with bias-corrected autocorrelation and (b) the p-value from a regression on
   annual means with an AR(1) effective sample size (Santer et al. 2000). We calibrated it by simulation on
   AR(1) null series and on surrogates built from real GLDAS residuals. Its false-positive rate was 3–6% at a
   nominal 5% (the Newey-West HAC test we used first gave 29–34% on soil water).
3. Multiple testing: Benjamini–Hochberg FDR across all tiles. Shares are reported by tile count and by land area.
4. Spatial clusters: local Moran's I (LISA) with 9,999 conditional permutations and BH-FDR, queen weights that
   wrap the dateline, plus k-nearest-neighbour and Getis–Ord Gi* robustness checks. A region is at least 5
   connected tiles that are both a LISA hot/cold spot and individually FDR-significant with the same sign.
5. Variability change: early-versus-late variance with a prewhitened, state-adjusted Brown–Forsythe test
   (simulated false-positive rate about 4–6%).
6. Export: JSON for the web explorer, Discovery Cards, slides and video, all generated from the same outputs, so
   every number on every surface comes from the same run.

BENEFITS

 • Honest statistics: autocorrelation-aware, calibrated tests and false-discovery control, so the map does not
   fill up with chance "trends".
 • Every claim carries its evidence grade, its caveats and the literature, so a user can judge how far to trust it.
 • It catches data artefacts. The soil-water retraction shows the tool rejecting its own most exciting result.
 • Open and reproducible: free NASA data, open-source code, one command per stage.

INTENDED IMPACT

Planners often rely on long-term climate averages. Paradox Earth points them to where a statistical trend in
the 2000–2025 record is detectable, how confident to be, and what published research says; it does not explain
causes or forecast. The explorer also teaches students and the public that a flat global average can sit on top
of regional trends in opposite directions, and that a trend line needs checking before anyone believes it.

TOOLS, LANGUAGES, LIBRARIES

Python 3.12: NumPy, pandas, SciPy, xarray, netCDF4, statsmodels, arch, PySAL (esda, libpysal), scikit-learn,
matplotlib, pytest, earthaccess (NASA Earthdata download), pypdf. Web: HTML, CSS and JavaScript with globe.gl
(three.js) for the 3D globe; fonts from Google Fonts. Build: headless Google Chrome (slides to PDF), FFmpeg
(video), GitHub Pages (hosting). PyTorch is used by an experimental water-balance forecaster in the repo; it
did not contribute to any result shown here.
```

---

## 6. Use of Super Intelligence (SI): disclosure template

> **[TEAM] Read this together and edit it until every sentence is true. Then paste it. This file states only
> what is visible in the repository. It does not say who had the idea; that line is yours to complete.**

```
USE OF SUPER INTELLIGENCE (AI) — DISCLOSURE

AI coding assistants were used in this project, as follows:

 • Code: AI coding assistants helped write, debug and review Python code for the data pipeline, statistical
   tests, spatial statistics and export scripts, and the HTML/CSS/JavaScript for the web explorer.
   [TEAM: confirm, or rewrite to what is true] The team reviewed, ran and tested the code.
 • Statistical calibration and validation: AI assistants helped write and run Monte Carlo simulations used to
   check the false-positive rate of the trend and variance tests, and helped run validation checks (quality
   control, break tests, window stability).
 • Demo materials: AI assistants helped generate the presentation slides, the 30-second video and the project
   website. The video carries a visible "AI-assisted production" label. The website has an "AI disclosure"
   section. [TEAM: confirm the slides carry an AI-assistance note before submitting.]
 • Writing: AI assistants helped draft and edit text, including this project page, the README, the Discovery
   Card text (user stories, literature notes, caveats) and the slide and video copy.

What AI did NOT produce: every map, chart and number is computed from NASA GLDAS-2.1 data by our code. No
AI-generated photographs or illustrations are used; the figures are data plots. The citations on the cards
were checked against their DOI records (authors, journal, volume, pages). [TEAM: confirm that you have read
the cited papers' abstracts and agree with each card's "agrees / partly agrees" label.]

Idea and design:
[TEAM — complete and confirm, or rewrite truthfully]
"The project idea and design, set out in the Paradox Earth project manifest
(docs/Paradox_Earth_Project_Manifest_v3.pdf), were conceived by our team members:
______________________________. AI was not used to come up with the idea of how to solve the challenge."
Confirmed by (names): ______________________  Date: __________
```

---

## 7. NASA Data

> **[TEAM]** Before pasting, open README_GLDAS2.pdf and check that the two radiation-forcing changes below are
> described as written (AGRMET radiation climatology shift; degraded WWMCA radiation from mid-2024). No tool in
> this session could read the PDF.

```
GLDAS-2.1 Noah Land Surface Model L4, monthly, 0.25° x 0.25°
 • Short name: GLDAS_NOAH025_M, version 2.1
 • DOI: 10.5067/SXAVCZFAQLNO
 • Provider: NASA Goddard Earth Sciences Data and Information Services Center (GES DISC), Hydrology Data and
   Information Services Center; distributed via NASA Earthdata
 • Landing page: https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH025_M_2.1/summary
 • Period used: 2000-01 to 2025-12 (312 months), global land; downloaded with earthaccess

Variables used
 • Rainf_f_tavg: total precipitation rate (rain + snow) from the GLDAS-2.1 forcing, which is derived from GPCP.
   This is the headline variable. It is converted from kg m⁻² s⁻¹ to mm/month.
 • SoilMoi0_10cm_inst, SoilMoi10_40cm_inst, SoilMoi40_100cm_inst, SoilMoi100_200cm_inst: summed to soil water
   0–200 cm. Withdrawn as a 2000–2025 trend (see below); analysed only in 2012-01 to 2024-06.
 • Also ingested and quality-checked but not reported as results: SWE_inst, CanopInt_inst, Evap_tavg, Qs_acc,
   Qsb_acc, Tair_f_inst.

How it is used
GLDAS-2.1 gives the same gap-free monthly grid over every land surface. Because of that, a trend in Chile and a
trend in Siberia are measured the same way. We aggregate the 0.25° grid to 17,284 one-degree land tiles, test
each tile for a trend with an autocorrelation-aware test, correct for multiple testing, find spatial clusters,
and turn each cluster into a Discovery Card. The explorer, slides and video are all built from these outputs.

README_GLDAS2 document
NASA GES DISC, "README Document for NASA GLDAS Version 2 Data Products"
(https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/doc/README_GLDAS2.pdf). It documents the
GLDAS-2.1 forcing sources and changes in the radiation forcing: an apparent shift in the AGRMET radiation
climatology and degraded WWMCA-based radiation from mid-2024. Our break tests found soil-water level shifts at
2010-08 and 2024-07 whose timing lines up with these documented forcing changes, which is why that trend was
withdrawn.

Caveat: GLDAS-2.1 is model output. Its precipitation is GPCP-based forcing, not an independent rain-gauge record.
```

---

## 8. Space Agency Partner Data

```
None. This project uses only NASA data (GLDAS-2.1). No data from Space Apps partner agencies was used.
```

---

## 9. References

```
DATA
 1. Beaudoing, H. and M. Rodell, NASA/GSFC/HSL (2020). GLDAS Noah Land Surface Model L4 monthly 0.25 x 0.25
    degree V2.1. Greenbelt, Maryland, USA: Goddard Earth Sciences Data and Information Services Center
    (GES DISC). doi:10.5067/SXAVCZFAQLNO. Accessed 2026.
 2. Rodell, M., Houser, P. R., Jambor, U., Gottschalck, J., Mitchell, K., Meng, C.-J., Arsenault, K.,
    Cosgrove, B., Radakovich, J., Bosilovich, M., Entin, J. K., Walker, J. P., Lohmann, D., & Toll, D. (2004).
    The Global Land Data Assimilation System. Bulletin of the American Meteorological Society, 85(3), 381–394.
    doi:10.1175/BAMS-85-3-381
 3. NASA GES DISC. README Document for NASA GLDAS Version 2 Data Products.
    https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/doc/README_GLDAS2.pdf
 4. Huffman, G. J., Adler, R. F., Morrissey, M. M., Bolvin, D. T., Curtis, S., Joyce, R., McGavock, B., &
    Susskind, J. (2001). Global precipitation at one-degree daily resolution from multisatellite observations.
    Journal of Hydrometeorology, 2(1), 36–50. doi:10.1175/1525-7541(2001)002<0036:GPAODD>2.0.CO;2
    (GPCP 1DD, the precipitation source behind the GLDAS-2.1 forcing.)

STATISTICAL METHODS
 5. Santer, B. D., Wigley, T. M. L., Boyle, J. S., Gaffen, D. J., Hnilo, J. J., Nychka, D., Parker, D. E., &
    Taylor, K. E. (2000). Statistical significance of trends and trend differences in layer-average atmospheric
    temperature time series. Journal of Geophysical Research, 105(D6), 7337–7356. doi:10.1029/1999JD901105
 6. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful
    approach to multiple testing. Journal of the Royal Statistical Society: Series B, 57(1), 289–300.
    doi:10.1111/j.2517-6161.1995.tb02031.x
 7. Anselin, L. (1995). Local indicators of spatial association — LISA. Geographical Analysis, 27(2), 93–115.
    doi:10.1111/j.1538-4632.1995.tb00338.x
 8. Ord, J. K., & Getis, A. (1995). Local spatial autocorrelation statistics: distributional issues and an
    application. Geographical Analysis, 27(4), 286–306. doi:10.1111/j.1538-4632.1995.tb00912.x
 9. Brown, M. B., & Forsythe, A. B. (1974). Robust tests for the equality of variances. Journal of the American
    Statistical Association, 69(346), 364–367. doi:10.1080/01621459.1974.10482955
10. Berger, R. L. (1982). Multiparameter hypothesis testing and acceptance sampling. Technometrics, 24(4),
    295–300. doi:10.1080/00401706.1982.10487790 (intersection–union test)
11. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: a new approach. Econometrica, 59(2),
    347–370. doi:10.2307/2938260 (EGARCH; exploratory variability model, not used for headline results)

LITERATURE ON THE DISCOVERY CARDS
12. Biasutti, M. (2019). Rainfall trends in the African Sahel: Characteristics, processes, and causes. WIREs
    Climate Change, 10(4), e591. doi:10.1002/wcc.591
13. Wu, K., Liu, H., Zhao, L., Lin, Y., & Yang, Z. (2022). The strengthening relationship between summer
    rainfall over North China and PDO since the mid-2000s. Frontiers in Earth Science, 10, 957940.
    doi:10.3389/feart.2022.957940
14. Joseph, L., Skliris, N., Dey, D., Marsh, R., Hirschi, J., et al. (2024). Increased summer monsoon rainfall
    over Northwest India caused by Hadley cell expansion and Indian Ocean warming. Geophysical Research Letters,
    51, e2024GL108829. doi:10.1029/2024GL108829
15. Jin, Q., & Wang, C. (2017). A revival of Indian summer monsoon rainfall since 2002. Nature Climate Change,
    7, 587–594. doi:10.1038/nclimate3348
16. Ullah, W., Karim, A., Ullah, S., et al. (2023). An increasing trend in daily monsoon precipitation extreme
    indices over Pakistan and its relationship with atmospheric circulations. Frontiers in Environmental
    Science, 11, 1228817. doi:10.3389/fenvs.2023.1228817
17. Garreaud, R. D., Boisier, J. P., Rondanelli, R., Montecinos, A., Sepulveda, H. H., & Veloso-Aguila, D.
    (2020). The Central Chile Mega Drought (2010–2018): A climate dynamics perspective. International Journal
    of Climatology, 40(1), 421–439. doi:10.1002/joc.6219
18. Garreaud, R. D., Alvarez-Garreton, C., Barichivich, J., et al. (2017). The 2010–2015 megadrought in central
    Chile: impacts on regional hydroclimate and vegetation. Hydrology and Earth System Sciences, 21,
    6307–6327. doi:10.5194/hess-21-6307-2017
19. Espinoza, J. C., Ronchail, J., Marengo, J. A., & Segura, H. (2019). Contrasting North–South changes in
    Amazon wet-day and dry-day frequency and related atmospheric features (1981–2017). Climate Dynamics, 52,
    5413–5430. doi:10.1007/s00382-018-4462-2
20. Marengo, J. A., Jimenez, J. C., Espinoza, J. C., Cunha, A. P., & Aragão, L. E. O. (2022). Increased climate
    pressure on the agricultural frontier in the Eastern Amazonia–Cerrado transition zone. Scientific Reports,
    12, 457. doi:10.1038/s41598-021-04241-4
21. Marengo, J. A., Souza, C. M., Thonicke, K., et al. (2018). Changes in climate and land use over the Amazon
    region: current and future variability and trends. Frontiers in Earth Science, 6, 228.
    doi:10.3389/feart.2018.00228
22. Hua, W., Zhou, L., Chen, H., Nicholson, S. E., Raghavendra, A., & Jiang, Y. (2016). Possible causes of the
    Central Equatorial African long-term drought. Environmental Research Letters, 11, 124002.
    doi:10.1088/1748-9326/11/12/124002
23. Zhou, L., Tian, Y., Myneni, R. B., et al. (2014). Widespread decline of Congo rainforest greenness in the
    past decade. Nature, 509, 86–90. doi:10.1038/nature13265
(The soil-water card also cites references 2 and 3 above.)

SOFTWARE (licence)
Python libraries (requirements.txt and builder scripts):
 • NumPy (BSD-3-Clause) — https://numpy.org
 • pandas (BSD-3-Clause) — https://pandas.pydata.org
 • SciPy (BSD-3-Clause) — https://scipy.org
 • xarray (Apache-2.0) — https://xarray.dev
 • netCDF4-python (MIT) — https://unidata.github.io/netcdf4-python/
 • statsmodels (BSD-3-Clause) — https://www.statsmodels.org
 • arch (NCSA/University of Illinois Open Source License) — https://github.com/bashtage/arch
 • PySAL esda and libpysal (BSD-3-Clause) — https://pysal.org
 • scikit-learn (BSD-3-Clause) — https://scikit-learn.org
 • matplotlib (Matplotlib License, PSF-based) — https://matplotlib.org
 • pypdf (BSD-3-Clause) — https://github.com/py-pdf/pypdf
 • earthaccess (MIT) — https://github.com/nsidc/earthaccess
 • pytest (MIT) — https://pytest.org
 • PyTorch (BSD-3-Clause) — https://pytorch.org (experimental forecaster only; not used for reported results)
 • PyYAML (MIT) — listed in requirements.txt; not imported by the current code
Web explorer:
 • globe.gl 2.46.2 (MIT), loaded from cdn.jsdelivr.net; bundles three.js and three-globe (MIT)
   — https://github.com/vasturiano/globe.gl
 • The globe texture is drawn in the browser from our own tile data; no third-party imagery is used.
Tools:
 • Google Chrome, headless (slides rendered to PDF)
 • FFmpeg (LGPL/GPL) — video encoding, https://ffmpeg.org
 • http-server (MIT, via npx) — local preview only
 • GitHub and GitHub Pages — code hosting and website hosting
Fonts (SIL Open Font License 1.1, served by Google Fonts):
 • IBM Plex Sans and IBM Plex Mono (site and slides)
 • Instrument Serif (site)
 • Bricolage Grotesque (slides)
 • The video uses the system font available on the build machine (Avenir Next / Helvetica Neue, falling back
   to DejaVu Sans, Bitstream Vera licence).

AI ASSISTANCE
 • AI coding assistants were used as described in the "Use of Super Intelligence" section.

NASA credit and affiliation: data courtesy of NASA GES DISC. No NASA logos are used. This project is not
affiliated with or endorsed by NASA.
```
