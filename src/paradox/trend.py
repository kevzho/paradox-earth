"""S3: mean-trend inference  y_t = a + b t + sum_{m=2..12} g_m I_{m,t} + e_t.

Point estimate: the OLS slope (units per decade). Inference is an intersection-union test, calibrated by simulation
on AR(1) nulls (phi 0..0.95) and on surrogates built from real GLDAS residuals (soil water lag-1 ACF ~0.88;
precipitation with excess decadal variability):
  p_m  OLS slope with an AR(1)-parametric standard error, rho bias-corrected as rho + (2 + 4 rho)/n, t(n - k)
  p_a  OLS on calendar-year means with an AR(1) effective sample size (Santer et al. 2000)
  p    = max(p_m, p_a): significant only when both reject. Empirical size 3-6% under every null tried, versus
       29-34% for the previous Newey-West(5 lags) HAC test on soil water. The CI is the envelope of the two CIs, so it
       excludes 0 exactly when both tests reject. The monthly leg needs consecutive months; with gaps, p = p_a.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

METHOD = ("OLS slope; p = max(AR(1)-parametric SE with bias-corrected rho [t(n-k)], "
          "annual-mean OLS with AR(1) n_eff [Santer 2000]) (intersection-union)")


def _design(index: pd.DatetimeIndex) -> pd.DataFrame:
    # time in months since start so the slope is per month; month dummies with January as reference
    t = (index.year - index[0].year) * 12 + (index.month - index[0].month)
    X = pd.DataFrame({"t": t.astype(float)}, index=index)
    for m in range(2, 13):
        X[f"m{m}"] = (index.month == m).astype(float)
    return sm.add_constant(X, has_constant="add")


def _ar1_param_se(e: np.ndarray, x_t: np.ndarray, n: int, k: int):
    """SE of the OLS coefficient whose Frisch-Waugh-partialled regressor is x_t, under AR(1) errors."""
    rho = float(e[1:] @ e[:-1] / (e[:-1] @ e[:-1]))
    rho = float(np.clip(rho + (2 + 4 * rho) / n, -0.99, 0.99))
    s2u = float(((e[1:] - rho * e[:-1]) ** 2).sum() / (n - 1 - k))
    g = np.correlate(x_t, x_t, mode="full")[n - 1:]  # g_j = sum_t x_t x_{t+j}
    q = g[0] + 2.0 * float(g[1:] @ rho ** np.arange(1, n))
    return float(np.sqrt(s2u / (1 - rho ** 2) * q) / float(x_t @ x_t)), rho


def _annual_ess(y: pd.Series):
    """OLS on complete calendar-year means with an AR(1) effective sample size; slope per year."""
    a = y.groupby(y.index.year).agg(["mean", "size"])
    a = a.loc[a["size"] == 12, "mean"]
    x = a.index.values.astype(float)
    x = x - x.mean()
    b = float(x @ (a.values - a.values.mean()) / (x @ x))
    e = a.values - a.values.mean() - b * x
    r1 = float(np.clip(e[1:] @ e[:-1] / (e @ e), 0.0, 0.99)) if (e @ e) > 0 else 0.0
    neff = max(len(a) * (1 - r1) / (1 + r1), 3.0)
    df = max(neff - 2.0, 1.0)
    se = float(np.sqrt((e @ e) / df / (x @ x)))
    return b, se, df, r1, neff, len(a)


def seasonal_trend(y: pd.Series, units: str = "", alpha: float = 0.05) -> dict:
    """Trend per decade with the calibrated intersection-union test. Missing months are dropped, never imputed."""
    y = y.dropna()
    n = len(y)
    if n < 36:
        return {"status": "insufficient_data", "n": n}
    gaps = len(pd.date_range(y.index[0], y.index[-1], freq="MS")) - n
    X = _design(pd.DatetimeIndex(y.index))
    X = X.loc[:, (X.std() > 0) | (X.columns == "const")]  # drop dummies for months absent from sample
    k = X.shape[1]
    fit = sm.OLS(y.values, X.values).fit()
    i = list(X.columns).index("t")
    b = float(fit.params[i])
    ba, se_a, df_a, r1a, neff, ny = _annual_ess(y)
    if ny < 5:
        return {"status": "insufficient_data", "n": n, "complete_years": ny}
    p_a = float(2 * stats.t.sf(abs(ba / se_a), df_a)) if se_a > 0 else 1.0
    q_a = stats.t.ppf(1 - alpha / 2, df_a)
    lo, hi = (ba - q_a * se_a) / 12, (ba + q_a * se_a) / 12  # per month
    p_m = rho = se_m = None
    if gaps == 0:
        z = X.drop(columns="t").values
        x_t = X["t"].values - z @ np.linalg.lstsq(z, X["t"].values, rcond=None)[0]  # Frisch-Waugh
        se_m, rho = _ar1_param_se(np.asarray(fit.resid), x_t, n, k)
        p_m = float(2 * stats.t.sf(abs(b / se_m), n - k)) if se_m > 0 else 1.0
        q_m = stats.t.ppf(1 - alpha / 2, n - k)
        lo, hi = min(lo, b - q_m * se_m), max(hi, b + q_m * se_m)
    per_decade = 120.0
    return {
        "status": "ok",
        "slope": b * per_decade,
        "ci95": [float(lo * per_decade), float(hi * per_decade)],
        "se": se_m * per_decade if se_m is not None else None,
        "p_value": max(p_m, p_a) if p_m is not None else p_a,
        "p_monthly_ar1": p_m, "p_annual_ess": p_a, "rho_ar1_bc": rho, "annual_r1": r1a, "annual_neff": neff,
        "units": f"{units} per decade".strip(),
        "n": n, "gaps": int(gaps),
        "window": [str(y.index[0].date()), str(y.index[-1].date())],
        "method": METHOD if gaps == 0 else METHOD + " - monthly leg skipped (gaps), p = annual leg",
    }


def slope_difference(y1: pd.Series, y2: pd.Series, units: str = "") -> dict:
    """Direct test of b1 - b2: trend of the paired difference series on the common window. Testing the difference
    series absorbs cross-correlation between the two regions."""
    d = (y1 - y2).dropna()
    out = seasonal_trend(d, units)
    out["method"] = "paired_difference_" + out.get("method", "")
    return out
