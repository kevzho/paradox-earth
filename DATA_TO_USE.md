# Data to use

Everything the Paradox Earth pipeline needs, on standby. **Nothing here is downloaded yet** — all sources need a free
NASA Earthdata login (https://urs.earthdata.nasa.gov). Fetch with `scripts/download_data.py` (uses `earthaccess`).
Raw files go in `data/raw/<key>/` (git-ignored). Every version/short name marked *verify* must be confirmed in
Earthdata Search at retrieval time and recorded in the source manifest (`data/manifests/`).

## Priority order (follows Manifest §18–20)

| # | Key | Lens | Needed by | Tier |
|---|---|---|---|---|
| 1 | `gldas` | A — Weather & climate | S1–S4 stats, ARIMA-EGARCH, PINN | **0 (required)** |
| 2 | `viirs_ntl` | B — Human encroachment | cross-lens A (partner) | 1 |
| 3 | `modis_lc` | B — Human encroachment | land-cover transitions (partner) | 1–2 |
| 4 | `grace` | C — Planetary resources | GRACE storage check for PINN, Lens C | 2–3 |
| 5 | `modis_vi` | C — Planetary resources | vegetation instability (cross-lens B) | 2 |
| 6 | `population` | B context (optional) | context only | 2 |

---

### 1. GLDAS-2.1 Noah monthly 0.25° — **flagship, start here**
- **Short name:** `GLDAS_NOAH025_M`  **Version:** `2.1`  **DOI:** 10.5067/SXAVCZFAQLNO (*verify*)
- **Provider:** NASA GES DISC. **Period:** 2000-01 → present (≈1–2 month latency). **Grid:** 0.25°, 1440×600, global land.
- **Files:** one NetCDF4 per month, `GLDAS_NOAH025_M.A{YYYYMM}.021.nc4` (~25 MB each; ~7.5 GB for 2000–2025 globally).
- **Variables used** (native units → harmonised by `gldas.to_monthly_water_units`):

| Variable | Native unit | Use |
|---|---|---|
| `SoilMoi0_10cm_inst`, `SoilMoi10_40cm_inst`, `SoilMoi40_100cm_inst`, `SoilMoi100_200cm_inst` | kg m⁻² | soil-water storage (primary demo variable + PINN target) |
| `SWE_inst`, `CanopInt_inst` | kg m⁻² | added to PINN storage target |
| `Rainf_f_tavg` | kg m⁻² s⁻¹ | total precipitation P (weather driver; alt. demo variable) |
| `Evap_tavg` | kg m⁻² s⁻¹ | evapotranspiration ET |
| `Qs_acc`, `Qsb_acc` | kg m⁻² per 3 h | surface + subsurface runoff R |
| `Tair_f_inst` | K | air temperature (weather driver) |

- **Recommended pull for the hackathon:** CONUS or one continent first (`--bbox -125 24 -66 50`), 2000-01 → 2025-12
  (26 full years = 312 months; ≥120 months is the EGARCH data gate). Go global only after Gate E.
- **Caveats:** model output driven by forcing, not independent observations. Monthly storages are monthly *means*,
  so the water budget uses `alignment="centered"`. Do **not** splice GLDAS-2.0 (1948–2014) onto 2.1: different forcing.

### 2. VIIRS Black Marble nighttime lights
- **Monthly:** `VNP46A3` (*verify version*, LAADS DAAC). **Yearly:** `VNP46A4`. 15″ (~500 m) tiles, 2012-01 → present.
- **Use:** radiance trends (not population). Aggregate to the 0.25° GLDAS grid (or coarser) before any cross-lens stat.
- Use the gap-filled / quality-flagged radiance layer; keep the quality flags.

### 3. MODIS land cover
- **Short name:** `MCD12C1` v061 (0.05° CMG, yearly, 2001 → ~2023) — easiest to harmonise. `MCD12Q1` v061 (500 m) if needed.
- **Use:** annual land-cover transitions, analysed as **annual** data (no monthly EGARCH, no interpolation to monthly).

### 4. GRACE / GRACE-FO terrestrial water storage
- **Short name:** `TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4` (JPL mascons, PO.DAAC; *verify* release).
- **Period:** 2002-04 → present, gap 2017-07 → 2018-05 between missions (keep missing; never interpolate into stats).
- **Grid:** 0.5° display grid but ~3° (~300 km) effective resolution — use GRACE-scale regions/weights.
- **Use:** independent total-storage check on PINN/GLDAS storage anomalies (GRACE − GLDAS = exploratory residual,
  not groundwater). Apply the land mask and scale factors shipped with the product.

### 5. MODIS vegetation indices
- **Short name:** `MOD13C2` v061 (monthly 0.05° CMG NDVI/EVI, 2000-02 → present). Use EVI; keep pixel-reliability.
- **Use:** vegetation instability vs soil-moisture instability (cross-lens B).

### 6. Gridded population (optional context)
- **GPWv4.11** population density (NASA SEDAC, now distributed via Earthdata; *verify* location). 5-yearly 2000–2020.
- Context only; never a causal variable.

---

## Not needed for the MVP
GLDAS-2.0 / 2.2, MERRA-2, IMERG, SMAP — only as later robustness checks (e.g. IMERG precipitation vs GLDAS forcing).

## Synthetic stand-in
Until real data is pulled: `python run.py --synthetic` generates GLDAS-shaped files in `data/raw/synthetic/`
(clearly labelled synthetic; never use for findings).
