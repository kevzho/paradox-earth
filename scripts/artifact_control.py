"""Controlled comparison: GLDAS-2.1 vs GLDAS-2.0 over their common years (2000-01..2014-12).

Both products run the same Noah land model on the same 0.25 deg grid; they differ in the meteorological forcing
(2.1: GDAS/AGRMET radiation + GPCP precipitation; 2.0: Princeton reanalysis-based forcing). The question is whether
features of the GLDAS-2.1 record line up with the documented 2010-08 AGRMET radiation change and are absent from
GLDAS-2.0. A match is consistent with a forcing-record artefact; it does not by itself establish one.

A step test at one pre-chosen date is not enough for persistent series such as soil water: the AR(1) step p-value is
optimistic when residual autocorrelation is ~0.9. The script therefore also scans every candidate break date and reports
where 2010-08 ranks and how many arbitrary dates would also look "significant" (break_scan).

What it does
  1. Streams the monthly files one at a time and reduces each to a land-area-weighted mean on a COMMON static mask:
     cells with valid soil water + all analysed variables in BOTH versions (reference month = first paired month),
     minus glacier cells (soil water > ICE_SOIL_MM in either version) and high-SWE / ice-sheet cells
     (SWE > 1000 mm in either version). Weights: cos(latitude).
  2. Deseasonalises each series with a monthly climatology computed over the COMMON period (months present in both).
  3. Analyses the DIFFERENCE series (2.1 minus 2.0) and each version on its own: OLS on [const, trend, step at
     2010-08] with an AR(1)-parametric SE for the step (same estimator as paradox.trend), plus a moving-block-bootstrap
     CI of the step and pre/post means.
  4. Soil-water trend of each version with paradox.trend.seasonal_trend (calibrated intersection-union test).

Re-runnable: per-month means are cached in outputs_artifact/monthly_land_means.csv (keyed by a mask signature); a rerun
only reads months not yet cached, so it picks up GLDAS-2.0 files as the download completes. --rebuild ignores the cache.

Outputs: outputs_artifact/artifact_control.json, demo/artifact_control.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paradox import gldas  # noqa: E402
from paradox.trend import METHOD as TREND_METHOD, _ar1_param_se, seasonal_trend  # noqa: E402
from paradox.pipeline import DOCUMENTED_FORCING_EVENTS, FORCING_SOURCE  # noqa: E402

DIRS = {"2.1": ROOT / "data" / "raw" / "gldas", "2.0": ROOT / "data" / "raw" / "gldas20"}
PATTERN = {"2.1": re.compile(r"GLDAS_NOAH025_M\.A(\d{6})\.021\.nc4$"),
           "2.0": re.compile(r"GLDAS_NOAH025_M\.A(\d{6})\.020\.nc4$")}
OUT_DIR = ROOT / "outputs_artifact"
CACHE = OUT_DIR / "monthly_land_means.csv"
OUT_JSON = OUT_DIR / "artifact_control.json"
OUT_PNG = ROOT / "demo" / "artifact_control.png"

START, END = "2000-01", "2014-12"
STEP_DATE = "2010-08"
SWE_MAX_MM = 1000.0
VARS = ["SWdown_f_tavg", "LWdown_f_tavg", "Rainf_f_tavg", "Evap_tavg", "soil_water_0_200cm"]
UNITS = {"SWdown_f_tavg": "W m-2", "LWdown_f_tavg": "W m-2", "Rainf_f_tavg": "mm month-1",
         "Evap_tavg": "mm month-1", "soil_water_0_200cm": "mm"}
READ = ["SWdown_f_tavg", "LWdown_f_tavg", "Rainf_f_tavg", "Evap_tavg", "SWE_inst"] + gldas.STORAGE_VARS[:4]
BLOCK, NBOOT = 12, 2000


# ---------------------------------------------------------------- ingest ---
def list_months(version: str) -> dict[str, Path]:
    out = {}
    for p in sorted(DIRS[version].glob("*.nc4")):
        m = PATTERN[version].search(p.name)
        if m:
            ym = f"{m.group(1)[:4]}-{m.group(1)[4:]}"
            if START <= ym <= END:
                out[ym] = p
    return out


def read_fields(path: Path) -> dict[str, np.ndarray] | None:
    try:
        ds = gldas.add_derived(gldas.to_monthly_water_units(gldas.load_file(path, READ)))
    except Exception as exc:  # a file still being written, or truncated: skip it this run
        print(f"  skip {path.name}: {exc}", flush=True)
        return None
    out = {v: ds[v].isel(time=0).values.astype(float) for v in VARS + ["SWE_inst"]}
    out["lat"] = ds["lat"].values.astype(float)
    ds.close()
    return out


def build_mask(f21: dict, f20: dict):
    valid = np.ones_like(f21["soil_water_0_200cm"], bool)
    for f in (f21, f20):
        for v in VARS:
            valid &= np.isfinite(f[v])
    ice = np.zeros_like(valid)
    for f in (f21, f20):
        ice |= np.nan_to_num(f["soil_water_0_200cm"]) > gldas.ICE_SOIL_MM
        ice |= np.nan_to_num(f["SWE_inst"]) > SWE_MAX_MM
    mask = valid & ~ice
    w = np.broadcast_to(np.cos(np.deg2rad(f21["lat"]))[:, None], mask.shape)
    stats_ = {"cells_valid_both": int(valid.sum()), "cells_excluded_ice_or_high_swe": int((valid & ice).sum()),
              "cells_used": int(mask.sum())}
    sig = hashlib.sha1(np.packbits(mask).tobytes()).hexdigest()[:16]
    return mask, w, stats_, sig


def land_means(f: dict, mask: np.ndarray, w: np.ndarray) -> dict:
    row = {}
    total = float(w[mask].sum())
    for v in VARS:
        x = f[v]
        ok = mask & np.isfinite(x)
        ww = w[ok]
        row[v] = float((ww * x[ok]).sum() / ww.sum()) if ww.sum() > 0 else np.nan
        row[f"cov_{v}"] = float(ww.sum() / total)
    return row


def ingest(rebuild: bool):
    files = {v: list_months(v) for v in DIRS}
    paired = sorted(set(files["2.1"]) & set(files["2.0"]))
    if not paired:
        raise SystemExit("no month has both GLDAS-2.1 and GLDAS-2.0 files yet")
    ref = paired[0]
    f21, f20 = read_fields(files["2.1"][ref]), read_fields(files["2.0"][ref])
    if f21 is None or f20 is None:
        raise SystemExit(f"cannot read reference month {ref}")
    if not np.array_equal(f21["lat"], f20["lat"]):
        raise SystemExit("GLDAS-2.1 and GLDAS-2.0 grids differ")
    mask, w, mstats, sig = build_mask(f21, f20)
    mstats.update(reference_month=ref, signature=sig)

    cache = pd.DataFrame()
    if CACHE.exists() and not rebuild:
        cache = pd.read_csv(CACHE, dtype={"version": str, "month": str, "mask_signature": str})
        cache = cache[cache["mask_signature"] == sig]
    done = set(zip(cache.get("version", []), cache.get("month", [])))
    rows = [] if cache.empty else cache.to_dict("records")
    todo = [(v, ym, p) for v in DIRS for ym, p in files[v].items() if (v, ym) not in done]
    print(f"mask {sig}: {mstats['cells_used']} cells; cached {len(done)} month-versions, reading {len(todo)}",
          flush=True)
    t0 = time.time()
    for i, (v, ym, p) in enumerate(todo):
        f = f21 if (v, ym) == ("2.1", ref) else f20 if (v, ym) == ("2.0", ref) else read_fields(p)
        if f is None:
            continue
        rows.append({"version": v, "month": ym, "mask_signature": sig, **land_means(f, mask, w)})
        if (i + 1) % 24 == 0:
            print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).sort_values(["version", "month"]).to_csv(CACHE, index=False)
    df = pd.DataFrame(rows).sort_values(["version", "month"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df, mstats


# ------------------------------------------------------------- analysis ---
def series(df: pd.DataFrame, version: str, var: str) -> pd.Series:
    d = df[df["version"] == version]
    s = pd.Series(d[var].values, index=pd.to_datetime(d["month"]), name=f"{var}_{version}").sort_index()
    return s.reindex(pd.date_range(f"{START}-01", f"{END}-01", freq="MS"))


def deseasonalise(y: pd.Series, common: pd.DatetimeIndex) -> pd.Series:
    clim = y.loc[common].groupby(common.month).mean()
    return y - clim.reindex(y.index.month).values


def step_test(y: pd.Series, date: str = STEP_DATE, seed: int = 0) -> dict:
    """y_t = a + b t + c 1[t >= date] + e_t. Step c with AR(1)-parametric SE (bias-corrected rho, t(n-3)) when the
    months are consecutive, and a moving-block-bootstrap (block 12) percentile CI of c."""
    y = y.dropna()
    when = pd.Timestamp(f"{date}-01")
    pre, post = y[y.index < when], y[y.index >= when]
    base = {"step_date": date, "n_pre": int(len(pre)), "n_post": int(len(post)),
            "window": [str(y.index[0].date())[:7], str(y.index[-1].date())[:7]] if len(y) else None}
    if len(pre) < 24 or len(post) < 12:
        return {"status": "insufficient_coverage", **base,
                "reason": "needs >= 24 months before and >= 12 months from the step date"}
    idx = pd.DatetimeIndex(y.index)
    t = ((idx.year - idx[0].year) * 12 + (idx.month - idx[0].month)).values.astype(float)
    s = (idx >= when).astype(float)
    X = np.column_stack([np.ones_like(t), t, s])
    n, k = X.shape
    fit = sm.OLS(y.values, X).fit()
    c = float(fit.params[2])
    gaps = int(len(pd.date_range(idx[0], idx[-1], freq="MS")) - n)
    out = {"status": "ok", **base, "step": c, "trend_per_decade": float(fit.params[1] * 120), "gaps": gaps,
           "pre_mean": float(pre.mean()), "post_mean": float(post.mean()),
           "post_minus_pre_mean": float(post.mean() - pre.mean())}
    if gaps == 0:
        z = X[:, :2]
        x_s = s - z @ np.linalg.lstsq(z, s, rcond=None)[0]
        se, rho = _ar1_param_se(np.asarray(fit.resid), x_s, n, k)
        q = stats.t.ppf(0.975, n - k)
        out.update(se_ar1=se, ci95_ar1=[c - q * se, c + q * se], rho_ar1_bc=rho,
                   p_ar1=float(2 * stats.t.sf(abs(c / se), n - k)) if se > 0 else 1.0)
    rng = np.random.default_rng(seed)
    e, yhat = np.asarray(fit.resid), np.asarray(fit.fittedvalues)
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, size=(NBOOT, nb))
    blocks = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(NBOOT, -1)[:, :n]
    ystar = yhat[None, :] + e[blocks]
    coef = np.linalg.lstsq(X, ystar.T, rcond=None)[0]
    lo, hi = np.percentile(coef[2], [2.5, 97.5])
    # two-sided p from the bootstrap distribution's sign (percentile-interval inversion)
    p_boot = 2 * min((coef[2] <= 0).mean(), (coef[2] >= 0).mean())
    out.update(ci95_block_bootstrap=[float(lo), float(hi)], p_block_bootstrap=float(min(1.0, p_boot)))
    return out


def break_scan(y: pd.Series, date: str = STEP_DATE) -> dict:
    """Same OLS + AR(1) step statistic at every admissible break date (>= 24 months before, >= 12 after).
    Reports the date with the largest |t|, the rank of `date`, and the share of dates with p <= 0.05. If many
    arbitrary dates look 'significant', the single-date p-value is not evidence that `date` is special."""
    y = y.dropna()
    idx = pd.DatetimeIndex(y.index)
    if len(pd.date_range(idx[0], idx[-1], freq="MS")) != len(y):
        return {"status": "skipped_month_gaps"}
    n = len(y)
    t = np.arange(n, dtype=float)
    z = np.column_stack([np.ones(n), t])
    rows = []
    for j in range(24, n - 12 + 1):
        s = (np.arange(n) >= j).astype(float)
        X = np.column_stack([z, s])
        fit = sm.OLS(y.values, X).fit()
        x_s = s - z @ np.linalg.lstsq(z, s, rcond=None)[0]
        se, _ = _ar1_param_se(np.asarray(fit.resid), x_s, n, 3)
        c = float(fit.params[2])
        tt = abs(c / se) if se > 0 else 0.0
        rows.append((idx[j].strftime("%Y-%m"), c, tt, float(2 * stats.t.sf(tt, n - 3))))
    if not rows:
        return {"status": "insufficient_coverage"}
    tabs = np.array([r[2] for r in rows])
    best = rows[int(np.argmax(tabs))]
    at = next((r for r in rows if r[0] == date), None)
    return {"status": "ok", "candidate_dates": len(rows), "first": rows[0][0], "last": rows[-1][0],
            "best_date": best[0], "best_step": best[1], "best_p_ar1": best[3],
            "rank_of_step_date_by_abs_t": None if at is None else int((tabs >= at[2]).sum()),
            "share_of_dates_with_p_le_0.05": float(np.mean([r[3] <= 0.05 for r in rows]))}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def analyse(df: pd.DataFrame, mstats: dict):
    raw = {(v, var): series(df, v, var) for v in DIRS for var in VARS}
    have = {v: raw[(v, "soil_water_0_200cm")].dropna().index for v in DIRS}
    common = have["2.1"].intersection(have["2.0"])
    anom = {key: deseasonalise(s, common) for key, s in raw.items()}
    res = {"variables": {}}
    for var in VARS:
        diff = (raw[("2.1", var)] - raw[("2.0", var)])
        diff_anom = deseasonalise(diff, common)
        r = {"units": UNITS[var],
             "difference_2.1_minus_2.0": {"mean_raw": float(diff.mean()),
                                          "step_test": step_test(diff_anom),
                                          "break_scan": break_scan(diff_anom)},
             "step_test_2.1": step_test(anom[("2.1", var)]),
             "step_test_2.0": step_test(anom[("2.0", var)])}
        if var == "soil_water_0_200cm" or var in ("Rainf_f_tavg", "Evap_tavg"):
            r["trend_2.1_full"] = seasonal_trend(raw[("2.1", var)].dropna(), UNITS[var])
            r["trend_2.1_common"] = seasonal_trend(raw[("2.1", var)].loc[common], UNITS[var])
            r["trend_2.0_common"] = seasonal_trend(raw[("2.0", var)].loc[common], UNITS[var])
            r["trend_difference_common"] = seasonal_trend(diff.loc[common].dropna(), UNITS[var])
        res["variables"][var] = r

    def span(ix):
        return [str(ix[0].date())[:7], str(ix[-1].date())[:7]] if len(ix) else None

    expected = len(pd.date_range(f"{START}-01", f"{END}-01", freq="MS"))
    coverage = {"window": [START, END], "months_expected": expected,
                "gldas21_months": int(len(have["2.1"])), "gldas21_span": span(have["2.1"]),
                "gldas20_months": int(len(have["2.0"])), "gldas20_span": span(have["2.0"]),
                "common_months": int(len(common)), "common_span": span(common),
                "gldas20_complete": bool(len(have["2.0"]) == expected),
                "step_date_inside_common_period": bool(len(common) and common[-1] >= pd.Timestamp(f"{STEP_DATE}-01"))}
    res = {
        "title": "Controlled comparison: GLDAS-2.1 vs GLDAS-2.0 (same Noah model and grid, different forcing)",
        "coverage": coverage,
        "mask": {**mstats, "rule": ("static common land mask from the reference month: soil water and all analysed "
                                     "variables finite in BOTH versions; excluded where soil water > "
                                     f"{gldas.ICE_SOIL_MM:.0f} mm (glacier) or SWE > {SWE_MAX_MM:.0f} mm (ice sheet / "
                                     "high-SWE) in either version"),
                 "weights": "cos(latitude) area weights"},
        "methods": {
            "land_mean": "area-weighted mean over the common mask, one file at a time; missing stays NaN",
            "units": "SWdown/LWdown W m-2 (monthly mean); Rainf_f/Evap mm/month (rate x 86400 x days); soil water "
                     "0-200 cm mm (sum of 4 Noah layers)",
            "deseasonalise": "subtract the calendar-month climatology computed over the common period "
                             "(months present in both versions)",
            "step_test": ("OLS on [const, linear trend, step 1[t >= 2010-08]]; step SE AR(1)-parametric with "
                          "bias-corrected rho (paradox.trend estimator), t(n-3); plus moving-block bootstrap "
                          f"(block {BLOCK} months, {NBOOT} resamples of residuals) percentile 95% CI"),
            "break_scan": ("the same OLS + AR(1) step statistic at every admissible date; reports the best date, the "
                           "rank of 2010-08 by |t| and the share of dates with p <= 0.05. A high share means the "
                           "single-date p (and the residual block bootstrap, which keeps the design fixed) is "
                           "optimistic for that series, and 2010-08 is not distinctive"),
            "trend": "paradox.trend.seasonal_trend on the raw monthly series with month dummies: " + TREND_METHOD,
            "trend_2.1_full_note": "the series is restricted to the 2000-01..2014-12 window, so 'full' equals 'common' "
                                   "when GLDAS-2.0 is complete",
            "difference": "difference series = GLDAS-2.1 minus GLDAS-2.0 land mean, same mask, same months",
        },
        "step_date": STEP_DATE,
        "step_event": DOCUMENTED_FORCING_EVENTS.get(STEP_DATE),
        "forcing_source": FORCING_SOURCE,
        "forcing": {"2.1": "GDAS atmospheric analysis + GPCP precipitation + AGRMET radiation",
                    "2.0": "Princeton Global Meteorological Forcing (reanalysis-based)"},
        **res,
    }
    res["headline"] = headline(res)
    return clean(res), raw, anom, common


def headline(res: dict) -> dict:
    v = res["variables"]
    sw, soil = v["SWdown_f_tavg"], v["soil_water_0_200cm"]

    def pick(d, *keys):
        for k in keys:
            d = d.get(k) if isinstance(d, dict) else None
        return d

    return {
        "swdown_step_2.1_W_m2": pick(sw, "step_test_2.1", "step"),
        "swdown_step_2.1_ci95": pick(sw, "step_test_2.1", "ci95_ar1"),
        "swdown_step_2.1_p": pick(sw, "step_test_2.1", "p_ar1"),
        "swdown_step_2.0_W_m2": pick(sw, "step_test_2.0", "step"),
        "swdown_step_2.0_ci95": pick(sw, "step_test_2.0", "ci95_ar1"),
        "swdown_step_2.0_status": pick(sw, "step_test_2.0", "status"),
        "swdown_step_difference_W_m2": pick(sw, "difference_2.1_minus_2.0", "step_test", "step"),
        "swdown_step_difference_ci95": pick(sw, "difference_2.1_minus_2.0", "step_test", "ci95_ar1"),
        "swdown_step_difference_p": pick(sw, "difference_2.1_minus_2.0", "step_test", "p_ar1"),
        "swdown_step_difference_status": pick(sw, "difference_2.1_minus_2.0", "step_test", "status"),
        "soil_step_difference_mm": pick(soil, "difference_2.1_minus_2.0", "step_test", "step"),
        "soil_step_difference_status": pick(soil, "difference_2.1_minus_2.0", "step_test", "status"),
        "swdown_difference_break_scan": pick(sw, "difference_2.1_minus_2.0", "break_scan"),
        "soil_difference_break_scan": pick(soil, "difference_2.1_minus_2.0", "break_scan"),
        "evap_difference_break_scan": pick(v["Evap_tavg"], "difference_2.1_minus_2.0", "break_scan"),
        "soil_trend_2.1_full_mm_per_decade": pick(soil, "trend_2.1_full", "slope"),
        "soil_trend_2.1_full_p": pick(soil, "trend_2.1_full", "p_value"),
        "soil_trend_2.1_common_mm_per_decade": pick(soil, "trend_2.1_common", "slope"),
        "soil_trend_2.1_common_p": pick(soil, "trend_2.1_common", "p_value"),
        "soil_trend_2.0_common_mm_per_decade": pick(soil, "trend_2.0_common", "slope"),
        "soil_trend_2.0_common_p": pick(soil, "trend_2.0_common", "p_value"),
        "soil_trend_difference_common_mm_per_decade": pick(soil, "trend_difference_common", "slope"),
        "soil_trend_difference_common_p": pick(soil, "trend_difference_common", "p_value"),
        "common_window": res["coverage"]["common_span"],
    }


# ---------------------------------------------------------------- figure ---
INK, RULE, TEXT, MUTED = "#0A1322", "#22314A", "#E8EEF6", "#8C9AB0"
C21, C20 = "#E06C22", "#3D8FE0"  # validated (dataviz validate_palette, dark, surface #0A1322)


def fmt(x, nd=1):
    return "n/a" if x is None else f"{x:+.{nd}f}".replace("-", "−")


def fmt_p(p):
    return "n/a" if p is None else ("p < 0.001" if p < 0.001 else f"p = {p:.2f}")


def figure(res: dict, anom: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next((c for c in ("Avenir Next", "Helvetica Neue", "Helvetica") if c in names), "DejaVu Sans")
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "font.size": 15, "axes.edgecolor": RULE,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": TEXT})
    cov, h = res["coverage"], res["headline"]
    fig, axes = plt.subplots(2, 1, figsize=(13.33, 7.5), dpi=150, facecolor=INK, sharex=True)
    fig.subplots_adjust(left=0.085, right=0.865, top=0.83, bottom=0.10, hspace=0.36)
    step = pd.Timestamp(f"{STEP_DATE}-01")
    panels = [("SWdown_f_tavg", "Downward shortwave radiation (forcing)", "W m⁻²"),
              ("soil_water_0_200cm", "Soil water 0–200 cm (model state)", "mm")]
    for ax, (var, label, unit) in zip(axes, panels):
        ax.set_facecolor(INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=RULE, lw=0.8)
        ax.axhline(0, color=MUTED, lw=0.8, alpha=0.6)
        ax.axvline(step, color=TEXT, lw=1.2, ls=(0, (4, 3)), alpha=0.8)
        for ver, col, ls in (("2.0", C20, "-"), ("2.1", C21, "-")):
            a = anom[(ver, var)]
            ax.plot(a.index, a.values, color=col, lw=0.8, alpha=0.35)
            sm12 = a.rolling(12, center=True, min_periods=9).mean()
            ax.plot(sm12.index, sm12.values, color=col, lw=2.6, ls=ls)
            last = sm12.dropna()
            if len(last):
                ax.annotate(f"GLDAS-{ver}", (last.index[-1], last.values[-1]), xytext=(10, 0),
                            textcoords="offset points", color=col, fontsize=15, fontweight="bold", va="center",
                            annotation_clip=False)
        ax.set_title(label, loc="left", color=TEXT, fontsize=17, pad=8)
        ax.set_ylabel(f"anomaly ({unit})", fontsize=14)
        ax.tick_params(labelsize=13)
    axes[0].text(step, 1.0, "documented GLDAS-2.1 radiation-forcing change, 2010-08  ", transform=axes[0].get_xaxis_transform(),
                 color=TEXT, fontsize=12.5, va="top", ha="right")
    axes[1].set_xlim(pd.Timestamp(f"{START}-01"), pd.Timestamp(f"{END}-01") + pd.offsets.MonthEnd(0))

    # headline numbers, right-aligned on each panel's title row
    def ci(x):
        return "" if not x else f" (95% CI {fmt(x[0])} to {fmt(x[1])})"

    def right(ax, txt):
        ax.text(1.0, 1.02, txt, transform=ax.transAxes, fontsize=13, color=MUTED, ha="right", va="bottom")
    if h["swdown_step_difference_W_m2"] is not None:
        right(axes[0], f"step at 2010-08:  2.1 {fmt(h['swdown_step_2.1_W_m2'])}  |  2.0 "
                       f"{fmt(h['swdown_step_2.0_W_m2'])}  |  2.1\u22122.0 {fmt(h['swdown_step_difference_W_m2'])} "
                       "W m\u207b\u00b2")
    else:
        right(axes[0], f"step at 2010-08:  2.1 {fmt(h['swdown_step_2.1_W_m2'])} W m\u207b\u00b2  |  2.0 awaiting data")
    soil = res["variables"]["soil_water_0_200cm"]["difference_2.1_minus_2.0"]["step_test"]
    scan = h.get("soil_difference_break_scan") or {}
    if soil.get("status") == "ok":
        rank = (f";  2010-08 ranks {scan['rank_of_step_date_by_abs_t']} of {scan['candidate_dates']} candidate dates"
                if scan.get("status") == "ok" and scan.get("rank_of_step_date_by_abs_t") else "")
        right(axes[1], f"step in 2.1\u22122.0 at 2010-08: {fmt(soil['step'])} mm{rank}")
    cw = h["common_window"] or ["?", "?"]
    trend_note = (f"   |   soil-water trend {cw[0][:4]}\u2013{cw[1][:4]}: 2.1 "
                  f"{fmt(h['soil_trend_2.1_common_mm_per_decade'])} ({fmt_p(h['soil_trend_2.1_common_p'])}), 2.0 "
                  f"{fmt(h['soil_trend_2.0_common_mm_per_decade'])} mm/decade ({fmt_p(h['soil_trend_2.0_common_p'])})")

    fig.text(0.085, 0.975, "Same land model, different forcing: GLDAS-2.1 vs GLDAS-2.0", fontsize=21,
             fontweight="bold", color=TEXT, va="top")
    sub = ("Land-area-weighted means on a common mask (glaciers and ice sheets excluded); deseasonalised; "
           "thin = monthly, bold = 12-month mean")
    fig.text(0.085, 0.918, sub, fontsize=12.5, color=MUTED, va="top")
    note = "" if cov["gldas20_complete"] else (
        f"   |   GLDAS-2.0 coverage {cov['gldas20_span'][0]}–{cov['gldas20_span'][1]} "
        f"({cov['gldas20_months']}/{cov['months_expected']} months)")
    fig.text(0.085, 0.025, f"Source: NASA GLDAS-2.1 / GLDAS-2.0 (Noah 0.25° monthly){note}{trend_note}", fontsize=11.5,
             color=MUTED, va="bottom")
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor=INK)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true", help="ignore the per-month cache")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, mstats = ingest(args.rebuild)
    res, raw, anom, common = analyse(df, mstats)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)
    figure(res, anom)
    print(json.dumps({"coverage": res["coverage"], "headline": res["headline"]}, indent=2))
    print(f"wrote {OUT_JSON}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
