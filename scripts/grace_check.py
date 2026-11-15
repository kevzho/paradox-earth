"""Observational sign check of model storage trends against GRACE/GRACE-FO (JPL mascon RL06.3Mv04 CRI).

GRACE measures total terrestrial water storage (TWS: soil + groundwater + snow + surface water + ice). GLDAS
soil_water_0_200cm is one component of it, so this is a *consistency* check of trend sign and spatial pattern, not a
validation of magnitudes. Where they disagree, the difference is co-located with (not attributed to) known
groundwater-depletion or glacier regions; nothing here is a causal statement.

Usage
  python scripts/grace_check.py                                 # GRACE 3-degree box trends 2003-01..2024-12
  python scripts/grace_check.py --compare outputs_global_v2 soil_water_0_200cm --window 2012-01 2024-06

GRACE processing
  * lwe_thickness (cm) x 10 -> mm; land_mask == 1 cells only; multiplied by the gridded CRI scale_factor (file:
    "to be used with mascon solution that has the CRI filter applied", derived from CLM land hydrology). The unscaled
    variant is carried as a sensitivity column (slope_mm_decade_unscaled). Land cells with no gain (NaN over
    Antarctica/Greenland, 0 over the Caspian Sea) are left out of the scaled box mean rather than counted as zero.
  * lon rolled to -180..180; each solution is assigned to the calendar month containing its time_bounds midpoint;
    months with no solution stay NaN (incl. the 2017-07..2018-05 GRACE / GRACE-FO gap). On a midpoint collision the
    solution whose bounds also overlap an empty neighbouring month moves there (2015-04-12..05-11 -> 2015-05, matching
    the file's months_missing list). Any remaining double month is
    averaged (counted in the summary).
  * 3-degree boxes on a regular grid anchored at (-90, -180): cos(lat)-weighted mean over the box's GRACE land cells.
  * Excluded: Antarctica (south of 60S), Greenland (59-84N, 74-10W, which also removes eastern Baffin Island), boxes
    with land fraction < MIN_LAND, boxes whose GLDAS-2.1 land-ice share (Noah soil > ICE_SOIL_MM) is >= ICE_MAX,
    boxes adjacent to a box with ice share >= ICE_NEIGHBOR (GRACE leakage), and named glacier regions (Gulf of
    Alaska, Canadian Arctic, Iceland, Svalbard, Russian Arctic islands, Patagonian icefields).
  * Trends: paradox.trend.seasonal_trend (calibrated). GRACE has missing months, so its p-value is the annual-mean leg
    over complete calendar years only. BH-FDR (q = 0.05) across boxes, per product.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paradox import gldas  # noqa: E402
from paradox.trend import seasonal_trend  # noqa: E402

GRACE_DIR = ROOT / "data" / "raw" / "grace"
GLDAS_DIR = ROOT / "data" / "raw" / "gldas"
OUT = ROOT / "outputs_grace"
BOX = 3.0
NLAT, NLON = int(180 / BOX), int(360 / BOX)
MIN_LAND = 0.30          # minimum GRACE land fraction of a box
ICE_MAX = 0.02           # exclude a box when >= 2% of its land is Noah land ice
ICE_NEIGHBOR = 0.10      # ... or when an 8-neighbour has >= 10% (mascon leakage of glacier mass loss)
FDR_Q = 0.05
WORKERS = 3
DEFAULT_WINDOW = ("2003-01", "2024-12")

# (lat_min, lat_max, lon_min, lon_max) tested against the box centre
GLACIER_REGIONS = {
    "Gulf of Alaska glaciers": (56, 65, -156, -128),
    "Canadian Arctic Archipelago": (72, 84, -125, -60),
    "Iceland": (62, 68, -26, -12),
    "Svalbard": (76, 81, 8, 30),
    "Russian Arctic islands": (70, 82, 45, 110),
    "Patagonian icefields": (-56, -45, -77, -70),
}
GREENLAND = (59, 84, -74, -10)
# Regions with documented GRACE TWS declines co-located with groundwater withdrawal (Rodell et al. 2018, Nature 557)
DEPLETION_HOTSPOTS = {
    "NW India": (24, 34, 68, 80),
    "North China Plain": (33, 42, 111, 121),
    "California Central Valley": (34, 41, -123, -117),
    "Middle East (Tigris-Euphrates / Iran)": (27, 40, 38, 62),
    "Caspian Sea region (also Caspian sea-level decline)": (36, 48, 45, 56),
}
GLACIER_INFLUENCED = {"High Mountain Asia": (27, 40, 70, 100), "Tien Shan": (39, 46, 68, 96),
                      "Cordillera Darwin (Tierra del Fuego)": (-57, -53, -74, -64)}
NOT_COMPARABLE = {"Rainf_f_tavg", "precipitation", "Evap_tavg", "Qs_acc", "Qsb_acc", "runoff", "net_flux",
                  "Tair_f_inst"}


# ---------------------------------------------------------------- box geometry
def box_id(i: int, j: int) -> str:
    return f"box_{-90 + BOX * i:+06.1f}_{-180 + BOX * j:+07.1f}"


def box_center(i, j):
    return -90 + BOX * (np.asarray(i) + 0.5), -180 + BOX * (np.asarray(j) + 0.5)


def in_region(lat, lon, reg):
    a, b, c, d = reg
    return (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d)


def region_labels(lat, lon, regions) -> list[str]:
    return [k for k, r in regions.items() if in_region(lat, lon, r)]


def block_reduce(field: np.ndarray, f: int) -> np.ndarray:
    """Sum a (lat, lon) field over f x f blocks."""
    ny, nx = field.shape
    return field.reshape(ny // f, f, nx // f, f).sum(axis=(1, 3))


# ---------------------------------------------------------------- GLDAS ice mask (3-degree)
def gldas_ice_share() -> np.ndarray:
    """Share of each 3-degree box's Noah land that is land ice (soil water > ICE_SOIL_MM), from one GLDAS-2.1 file.
    Returns (NLAT, NLON) with NaN where GLDAS has no land."""
    f = sorted(GLDAS_DIR.glob("GLDAS_NOAH025_M.A2000??.021.nc4"))[0]
    ds = gldas.add_derived(gldas.to_monthly_water_units(gldas.load_file(f, gldas.STORAGE_VARS[:4])))
    land, ice = gldas.land_ice_masks(ds)
    lat, lon = ds["lat"].values, ds["lon"].values
    assert np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)
    f_cells = int(round(BOX / 0.25))
    i0 = int(round((lat[0] - 0.125 + 90) / BOX))
    j0 = int(round((lon[0] - 0.125 + 180) / BOX))
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, lon.size))
    ln = block_reduce(w * land, f_cells)
    ic = block_reduce(w * ice, f_cells)
    out = np.full((NLAT, NLON), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[i0:i0 + ln.shape[0], j0:j0 + ln.shape[1]] = np.where(ln > 0, ic / ln, np.nan)
    return out


# ---------------------------------------------------------------- GRACE
def grace_file() -> Path:
    fs = sorted(GRACE_DIR.glob("GRCTellus.JPL.*MSCN*CRI*.nc"))
    if not fs:
        raise FileNotFoundError(f"no JPL mascon CRI file in {GRACE_DIR}")
    return fs[-1]


def load_grace_boxes():
    """Stream the mascon file one solution at a time -> monthly 3-degree box means (mm), scaled and unscaled."""
    path = grace_file()
    nc = netCDF4.Dataset(path)
    nc.set_auto_mask(True)
    lat, lon = nc["lat"][:].filled(np.nan), nc["lon"][:].filled(np.nan)
    assert np.all(np.diff(lat) > 0) and lat.size == 360 and lon.size == 720, "unexpected GRACE grid"
    assert nc["lwe_thickness"].units == "cm"
    roll = int(np.argmax(lon > 180.0))                  # 0..360 -> -180..180
    lon = np.where(lon > 180, lon - 360, lon)
    lon = np.roll(lon, -roll)
    assert np.all(np.diff(lon) > 0)
    rolled = lambda a: np.roll(a, -roll, axis=-1)  # noqa: E731
    land = rolled(nc["land_mask"][:].filled(0)) == 1
    sf = rolled(nc["scale_factor"][:].filled(np.nan))
    f_cells = int(round(BOX / 0.5))
    w = np.cos(np.deg2rad(lat))[:, None] * land
    land_frac = block_reduce(land * np.cos(np.deg2rad(lat))[:, None], f_cells) / \
        block_reduce(np.ones_like(w) * np.cos(np.deg2rad(lat))[:, None], f_cells)

    tb = nc["time_bounds"][:].filled(np.nan)
    epoch = pd.Timestamp("2002-01-01")
    mid = epoch + pd.to_timedelta(tb.mean(axis=1), unit="D")
    month = pd.Series(mid.to_period("M").to_timestamp())
    start = epoch + pd.to_timedelta(tb[:, 0], unit="D")
    end = epoch + pd.to_timedelta(tb[:, 1], unit="D")
    reassigned = []
    for t in np.where(month.duplicated(keep=False).values)[0]:  # midpoint collision: move a solution into an
        for nb in (month[t] + pd.offsets.MonthBegin(1), month[t] - pd.offsets.MonthBegin(1)):  # empty neighbour
            if (month == month[t]).sum() > 1 and nb not in set(month) and start[t] <= nb + pd.offsets.MonthEnd(0) \
                    and end[t] >= nb:
                reassigned.append(f"{start[t].date()}..{end[t].date()} -> {str(nb.date())[:7]}")
                month[t] = nb
                break
    month = pd.DatetimeIndex(month)
    months = pd.date_range(month.min(), month.max(), freq="MS")
    pos = {m: k for k, m in enumerate(months)}
    num_s = np.zeros((months.size, NLAT, NLON))
    num_u = np.zeros_like(num_s)
    count = np.zeros(months.size, int)
    wsum = block_reduce(w, f_cells)
    # Scaled product: only land cells with a usable gain. sf is NaN over Antarctica/Greenland and exactly 0 over the
    # Caspian Sea (inside the CRI land mask but outside CLM land); counting those cells as zero-valued land would
    # dilute the box mean, so they are dropped from both numerator and denominator of the scaled mean.
    sf_ok = land & np.isfinite(sf) & (sf != 0)
    w_s = np.where(sf_ok, w, 0.0)
    wsum_s = block_reduce(w_s, f_cells)
    for t in range(tb.shape[0]):
        x = rolled(nc["lwe_thickness"][t].filled(np.nan)) * 10.0       # cm -> mm
        ok = land & np.isfinite(x)
        ws = np.where(ok, w, 0.0)
        assert np.allclose(block_reduce(ws, f_cells), wsum), "missing land cells within a solution"
        xs = np.where(sf_ok & np.isfinite(x), x * sf, 0.0)
        k = pos[month[t]]
        num_s[k] += block_reduce(w_s * xs, f_cells)
        num_u[k] += block_reduce(ws * np.where(ok, x, 0.0), f_cells)
        count[k] += 1
    attr_missing = getattr(nc, "months_missing", "")
    meta = {
        "file": path.name, "title": nc.title, "n_solutions": int(tb.shape[0]),
        "time_mean_removed": getattr(nc, "time_mean_removed", None), "GIA_removed": getattr(nc, "GIA_removed", None),
        "first_month": str(months[0].date())[:7], "last_month": str(months[-1].date())[:7],
        "months_without_solution": [str(m.date())[:7] for m, c in zip(months, count) if c == 0],
        "midpoint_collisions_reassigned": reassigned,
        "months_with_two_solutions_averaged": [str(m.date())[:7] for m, c in zip(months, count) if c > 1],
        "file_attr_months_missing": attr_missing.split(";") if attr_missing else [],
        "land_cells_negative_scale_factor": int((land & (sf < 0)).sum()),
        "land_cells": int(land.sum()),
        "land_cells_scale_factor_missing_or_zero_dropped_from_scaled": int((land & ~sf_ok).sum()),
        "scale_factor_land_median": float(np.nanmedian(sf[land])),
    }
    nc.close()
    with np.errstate(invalid="ignore", divide="ignore"):
        den = wsum[None] * count[:, None, None]
        den_s = wsum_s[None] * count[:, None, None]
        scaled = np.where(den_s > 0, num_s / den_s, np.nan)
        unscaled = np.where(den > 0, num_u / den, np.nan)
    return months, scaled, unscaled, land_frac, meta


def box_table(land_frac, ice_share) -> pd.DataFrame:
    ii, jj = np.meshgrid(np.arange(NLAT), np.arange(NLON), indexing="ij")
    clat, clon = box_center(ii, jj)
    # 8-neighbour max of ice share (longitude wraps)
    ice0 = np.nan_to_num(ice_share)
    nb = np.zeros_like(ice0)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                sh = np.roll(ice0, dj, axis=1)
                sh = np.roll(sh, di, axis=0)
                if di == 1:
                    sh[0] = 0
                if di == -1:
                    sh[-1] = 0
                nb = np.maximum(nb, sh)
    df = pd.DataFrame({
        "box_id": [box_id(i, j) for i, j in zip(ii.ravel(), jj.ravel())],
        "i": ii.ravel(), "j": jj.ravel(), "lat": clat.ravel(), "lon": clon.ravel(),
        "south": clat.ravel() - BOX / 2, "west": clon.ravel() - BOX / 2,
        "grace_land_frac": land_frac.ravel(), "gldas_ice_share": ice_share.ravel(), "neighbor_ice_share": nb.ravel(),
    })
    df["area_km2"] = (gldas.EARTH_RADIUS_KM ** 2 * np.deg2rad(BOX)
                      * (np.sin(np.deg2rad(df.south + BOX)) - np.sin(np.deg2rad(df.south))))
    df["land_area_km2"] = df.area_km2 * df.grace_land_frac
    reason = np.full(len(df), "", dtype=object)
    lat, lon = df.lat.values, df.lon.values
    glac = np.array([";".join(region_labels(a, b, GLACIER_REGIONS)) for a, b in zip(lat, lon)], dtype=object)
    for cond, why in [
        (df.grace_land_frac.values < MIN_LAND, "land_frac<min"),
        (lat < -60, "antarctica"),
        (in_region(lat, lon, GREENLAND), "greenland"),
        (np.nan_to_num(df.gldas_ice_share.values) >= ICE_MAX, "ice_share"),
        (df.neighbor_ice_share.values >= ICE_NEIGHBOR, "neighbor_ice"),
        (glac != "", "glacier_region"),
    ]:
        reason = np.where(cond & (reason == ""), why, reason)
    df["excluded"] = reason
    df["hotspot"] = [";".join(region_labels(a, b, DEPLETION_HOTSPOTS)) for a, b in zip(lat, lon)]
    df["glacier_influenced"] = [";".join(region_labels(a, b, GLACIER_INFLUENCED)) for a, b in zip(lat, lon)]
    return df


# ---------------------------------------------------------------- trends
def _trend(args):
    y, idx = args
    r = seasonal_trend(pd.Series(y, index=idx), "mm")
    if r.get("status") != "ok":
        return np.nan, np.nan, np.nan, np.nan, r.get("n", 0), 0
    return r["slope"], r["p_value"], r["ci95"][0], r["ci95"][1], r["n"], r["gaps"]


def trends(series: np.ndarray, months: pd.DatetimeIndex, window) -> pd.DataFrame:
    """series: (time, nbox). Trend of each column over the window (inclusive months)."""
    sel = (months >= pd.Timestamp(window[0])) & (months <= pd.Timestamp(window[1]))
    idx = months[sel]
    sub = series[sel]
    jobs = [(sub[:, k], idx) for k in range(sub.shape[1])]
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(_trend, jobs, chunksize=32))
    out = pd.DataFrame(res, columns=["slope_mm_decade", "p_value", "ci95_lo", "ci95_hi", "n_months", "gaps"])
    out["q_fdr"] = np.nan
    ok = out.p_value.notna().values
    if ok.any():
        out.loc[ok, "q_fdr"] = multipletests(out.p_value[ok], method="fdr_bh")[1]
    out["sig_fdr"] = out.q_fdr < FDR_Q
    return out


def grace_boxes():
    months, scaled, unscaled, land_frac, meta = load_grace_boxes()
    boxes = box_table(land_frac, gldas_ice_share())
    keep = boxes.excluded.values == ""
    flat = lambda a: a.reshape(a.shape[0], -1)[:, keep]  # noqa: E731
    return months, flat(scaled), flat(unscaled), boxes, keep, meta


def complete_years(series: np.ndarray, months: pd.DatetimeIndex, window) -> list[int]:
    """Calendar years inside the window with 12 GRACE months (the only years seasonal_trend's annual leg uses)."""
    s = pd.Series(np.isfinite(series[:, 0]), index=months)[window[0]:window[1]]
    c = s.groupby(s.index.year).sum()
    return [int(y) for y in c.index[c.values == 12]]


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_grace(window=DEFAULT_WINDOW):
    OUT.mkdir(exist_ok=True)
    months, gs, gu, boxes, keep, meta = grace_boxes()
    b = boxes[keep].reset_index(drop=True)
    ts = trends(gs, months, window)
    tu = trends(gu, months, window)
    df = pd.concat([b[["box_id", "lat", "lon", "south", "west", "land_area_km2", "grace_land_frac",
                       "gldas_ice_share", "hotspot", "glacier_influenced"]], ts], axis=1)
    df["slope_mm_decade_unscaled"] = tu.slope_mm_decade
    df["p_value_unscaled"] = tu.p_value
    df.to_csv(OUT / "grace_trends_3deg.csv", index=False)
    pd.DataFrame(gs, index=months, columns=b.box_id).to_csv(OUT / "grace_boxes_monthly_mm.csv",
                                                             float_format="%.3f", index_label="time")
    ok = df.slope_mm_decade.notna()
    w = df.land_area_km2
    summ = {
        "grace": meta, "window": list(window),
        "exclusions": boxes.excluded.replace("", "kept").value_counts().to_dict(),
        "n_boxes_kept": int(keep.sum()), "n_boxes_with_trend": int(ok.sum()),
        "n_fdr_significant": int(df.sig_fdr.sum()),
        "land_area_share_fdr_significant": float(w[df.sig_fdr].sum() / w[ok].sum()),
        "land_area_share_negative_slope": float(w[ok & (df.slope_mm_decade < 0)].sum() / w[ok].sum()),
        "sign_agreement_scaled_vs_unscaled": float(
            (np.sign(df.slope_mm_decade) == np.sign(df.slope_mm_decade_unscaled))[ok].mean()),
        "note": ("GRACE TWS anomaly (mm water equivalent, vs 2004.0-2009.999 mean). Missing months -> "
                 "seasonal_trend p-value uses the annual-mean leg over complete calendar years only."),
        "grace_complete_years_in_window": complete_years(gs, months, window),
        "settings": {"box_deg": BOX, "min_land_frac": MIN_LAND, "ice_max": ICE_MAX, "ice_neighbor": ICE_NEIGHBOR,
                     "fdr_q": FDR_Q},
    }
    (OUT / "grace_trends_summary.json").write_text(json.dumps(jsonable(summ), indent=2))
    return summ


# ---------------------------------------------------------------- comparison
def model_boxes(outputs: Path, var: str, months_wanted=None):
    """1-degree tile series -> 3-degree land-area-weighted box means; box needs >= 50% of its area each month."""
    tiles = pd.read_csv(outputs / "tiles.csv")
    acol = f"area_km2__{var}"
    if acol not in tiles:
        raise SystemExit(f"{outputs}/tiles.csv has no {acol} column (older run); re-run the pipeline")
    reg = pd.read_csv(outputs / f"regional_{var}.csv", index_col="time", parse_dates=["time"])
    tiles = tiles.set_index("region_id").reindex(reg.columns)
    a = tiles[acol].fillna(0).values
    i = np.floor((tiles.south.values + 90) / BOX).astype(int)
    j = np.floor((tiles.west.values + 180) / BOX).astype(int)
    k = i * NLON + j
    S = sparse.csr_matrix((a, (np.arange(len(a)), k)), shape=(len(a), NLAT * NLON))
    M = reg.values
    fin = np.isfinite(M)
    num = sparse.csr_matrix(np.where(fin, M, 0.0)).dot(S).toarray()
    den = sparse.csr_matrix(fin.astype(float)).dot(S).toarray()
    tot = np.asarray(S.sum(axis=0)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where((den >= 0.5 * tot[None]) & (tot[None] > 0), num / den, np.nan)
    return pd.DatetimeIndex(reg.index), val, tot


def run_compare(outputs: Path, var: str, window=None):
    if var in NOT_COMPARABLE:
        msg = {"status": "skipped", "variable": var,
               "reason": "GRACE observes water storage (TWS); fluxes such as precipitation are not comparable."}
        return msg
    OUT.mkdir(exist_ok=True)
    gm, gs, gu, boxes, keep, meta = grace_boxes()
    mm, mval, mtot = model_boxes(outputs, var)
    lo = max(gm[0], mm[0], pd.Timestamp(window[0]) if window else gm[0])
    hi = min(gm[-1], mm[-1], pd.Timestamp(window[1]) if window else gm[-1])
    win = (str(lo.date())[:7], str(hi.date())[:7])
    b = boxes[keep].reset_index(drop=True)
    flat_idx = (b.i * NLON + b.j).values
    mv = mval[:, flat_idx]
    tot = mtot[flat_idx]
    common = (tot > 0) & np.isfinite(mv).any(axis=0)
    b, gs, gu, mv, tot = b[common].reset_index(drop=True), gs[:, common], gu[:, common], mv[:, common], tot[common]
    # GLDAS sampled on GRACE months (same missing-month pattern) as a sampling sensitivity
    g_on_m = pd.DataFrame(gs, index=gm).reindex(mm).values
    mv_matched = np.where(np.isfinite(g_on_m), mv, np.nan)

    tg = trends(gs, gm, win)
    tgu = trends(gu, gm, win)
    tm = trends(mv, mm, win)
    tmm = trends(mv_matched, mm, win)
    df = b[["box_id", "lat", "lon", "hotspot", "glacier_influenced", "gldas_ice_share"]].copy()
    df["model_land_area_km2"] = tot
    for name, t in [("grace", tg), ("model", tm)]:
        for c in ["slope_mm_decade", "p_value", "q_fdr", "sig_fdr", "n_months"]:
            df[f"{name}_{c}"] = t[c].values
    df["grace_slope_unscaled"] = tgu.slope_mm_decade.values
    df["model_slope_on_grace_months"] = tmm.slope_mm_decade.values
    ok = df.grace_slope_mm_decade.notna() & df.model_slope_mm_decade.notna()
    df = df[ok].reset_index(drop=True)
    w = df.model_land_area_km2
    same = np.sign(df.grace_slope_mm_decade) == np.sign(df.model_slope_mm_decade)
    same_u = np.sign(df.grace_slope_unscaled) == np.sign(df.model_slope_mm_decade)
    either = df.grace_sig_fdr | df.model_sig_fdr
    both = df.grace_sig_fdr & df.model_sig_fdr
    share = lambda m, s=same: float(w[m & s].sum() / w[m].sum()) if m.any() else None  # noqa: E731
    rho, prho = stats.spearmanr(df.grace_slope_mm_decade, df.model_slope_mm_decade)
    rho_e, _ = stats.spearmanr(df.grace_slope_mm_decade[either], df.model_slope_mm_decade[either]) \
        if either.sum() > 2 else (np.nan, None)
    rho_m, _ = stats.spearmanr(df.grace_slope_mm_decade, df.model_slope_on_grace_months, nan_policy="omit")
    rho_u, _ = stats.spearmanr(df.grace_slope_unscaled, df.model_slope_mm_decade)

    opp = ~same & either
    dis = df[opp].assign(abs_diff=(df.grace_slope_mm_decade - df.model_slope_mm_decade).abs())
    dis = dis.sort_values("abs_diff", ascending=False).head(25)
    disagreements = [{
        "box_id": r.box_id, "lat": r.lat, "lon": r.lon,
        "grace_slope_mm_decade": r.grace_slope_mm_decade, "grace_q": r.grace_q_fdr, "grace_sig": r.grace_sig_fdr,
        "model_slope_mm_decade": r.model_slope_mm_decade, "model_q": r.model_q_fdr, "model_sig": r.model_sig_fdr,
        "flag": "; ".join(x for x in [r.hotspot, f"glacier-influenced: {r.glacier_influenced}"
                                      if r.glacier_influenced else ""] if x),
    } for r in dis.itertuples()]
    hot = df.hotspot != ""
    hot_tab = {}
    for name in DEPLETION_HOTSPOTS:
        m = df.hotspot.str.contains(name, regex=False)
        if m.any():
            hot_tab[name] = {
                "n_boxes": int(m.sum()),
                "grace_mean_slope_mm_decade": float(np.average(df.grace_slope_mm_decade[m], weights=w[m])),
                "model_mean_slope_mm_decade": float(np.average(df.model_slope_mm_decade[m], weights=w[m])),
                "sign_match_share": share(m), "n_grace_sig": int(df.grace_sig_fdr[m].sum()),
                "n_model_sig": int(df.model_sig_fdr[m].sum())}
    res = {
        "status": "ok", "variable": var, "outputs_dir": str(outputs), "window": list(win),
        "grace_file": meta["file"], "grace_scaling": "CRI scale_factor applied (unscaled = sensitivity)",
        "n_boxes": int(len(df)), "n_grace_fdr_sig": int(df.grace_sig_fdr.sum()),
        "n_model_fdr_sig": int(df.model_sig_fdr.sum()), "n_either_sig": int(either.sum()),
        "n_both_sig": int(both.sum()),
        "sign_match_share_area_weighted": {
            "sig_in_either": share(either), "sig_in_both": share(both), "sig_in_grace": share(df.grace_sig_fdr),
            "sig_in_model": share(df.model_sig_fdr), "all_boxes": share(pd.Series(True, index=df.index)),
            "sig_in_either_excluding_hotspots": share(either & ~hot),
            "sig_in_either_grace_unscaled": share(either, same_u),
        },
        "spearman_slopes": {"all_boxes": float(rho), "p": float(prho), "sig_in_either": float(rho_e),
                            "model_sampled_on_grace_months": float(rho_m), "grace_unscaled": float(rho_u)},
        "grace_complete_years_in_window": int(len(cy := complete_years(gs, gm, win))),
        "grace_p_value_basis": f"annual means of complete calendar years only: {cy}; GRACE slopes use every "
                               "available month, so a GRACE 'significant' flag tests the trend across these years",
        "hotspots": hot_tab,
        "strongest_disagreements": disagreements,
        "notes": [
            "GRACE = total water storage; model = soil water 0-200 cm only. Agreement in sign is a consistency check, "
            "not a validation of magnitude.",
            "Disagreements in flagged regions are co-located with documented groundwater-depletion areas (GRACE TWS "
            "includes groundwater and surface-water storage that the 0-200 cm soil column does not); they are "
            "flagged, not treated as model failure. The Caspian boxes also border the Caspian Sea, whose level "
            "decline is surface water. No causal attribution is made.",
            "GRACE has missing months, so its p-values use only the annual-mean leg over complete calendar years "
            "(see grace_complete_years_in_window); model p-values use the full calibrated test.",
        ],
    }
    tag = f"{var}_{win[0]}_{win[1]}".replace("-", "")
    (OUT / f"compare_{Path(outputs).name}_{tag}.json").write_text(json.dumps(jsonable(res), indent=2))
    df.to_csv(OUT / f"compare_{Path(outputs).name}_{tag}_boxes.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, metavar=("OUTPUTS_DIR", "VARIABLE"))
    ap.add_argument("--window", nargs=2, metavar=("START", "END"), help="YYYY-MM YYYY-MM (inclusive)")
    a = ap.parse_args()
    if a.compare:
        r = run_compare(Path(a.compare[0]).resolve(), a.compare[1], a.window)
        keys = ["window", "n_boxes", "n_grace_fdr_sig", "n_model_fdr_sig", "n_either_sig", "n_both_sig",
                "sign_match_share_area_weighted", "spearman_slopes", "grace_complete_years_in_window", "hotspots"]
        print(json.dumps(jsonable({k: r.get(k) for k in keys} if r.get("status") == "ok" else r), indent=1))
        if r.get("status") == "ok":
            for d in r["strongest_disagreements"][:12]:
                print(f"{d['box_id']}  grace {d['grace_slope_mm_decade']:+8.1f} (sig {d['grace_sig']})  "
                      f"model {d['model_slope_mm_decade']:+8.1f} (sig {d['model_sig']})  {d['flag']}")
    else:
        s = run_grace(tuple(a.window) if a.window else DEFAULT_WINDOW)
        print(json.dumps(jsonable({k: v for k, v in s.items() if k != "grace"}), indent=1))
        print(json.dumps(jsonable({k: v for k, v in s["grace"].items() if "months" in k or "scale" in k}), indent=1))


if __name__ == "__main__":
    main()
