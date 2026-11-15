"""S3: mean-trend inference  y_t = a + b t + sum_{m=2..12} g_m I_{m,t} + e_t  with HAC (Newey-West) SEs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _design(index: pd.DatetimeIndex) -> pd.DataFrame:
    # time in months since start so the slope is per month; month dummies with January as reference
    t = (index.year - index[0].year) * 12 + (index.month - index[0].month)
    X = pd.DataFrame({"t": t.astype(float)}, index=index)
    for m in range(2, 13):
        X[f"m{m}"] = (index.month == m).astype(float)
    return sm.add_constant(X, has_constant="add")


def newey_west_lags(n: int) -> int:
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def seasonal_trend(y: pd.Series, units: str = "", seasonal_dummies: bool = True, alpha: float = 0.05) -> dict:
    """Fit OLS with HAC SEs. Missing months are dropped (not imputed); slope reported per decade."""
    y = y.dropna()
    n = len(y)
    if n < 24:
        return {"status": "insufficient_data", "n": n}
    X = _design(pd.DatetimeIndex(y.index))
    if not seasonal_dummies:
        X = X[["const", "t"]]
    X = X.loc[:, (X.std() > 0) | (X.columns == "const")]  # drop dummies for months absent from sample
    lags = newey_west_lags(n)
    fit = sm.OLS(y.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    i = list(X.columns).index("t")
    lo, hi = fit.conf_int(alpha)[i]
    per_decade = 120.0
    return {
        "status": "ok",
        "slope": float(fit.params[i] * per_decade),
        "ci95": [float(lo * per_decade), float(hi * per_decade)],
        "se": float(fit.bse[i] * per_decade),
        "p_value": float(fit.pvalues[i]),
        "units": f"{units} per decade".strip(),
        "n": n,
        "window": [str(y.index[0].date()), str(y.index[-1].date())],
        "method": f"seasonal_regression_HAC(nw_lags={lags})",
    }


def slope_difference(y1: pd.Series, y2: pd.Series, units: str = "") -> dict:
    """Direct test of b1 - b2: trend of the paired difference series on the common window.
    HAC SEs on the difference absorb cross-correlation between the two regions."""
    d = (y1 - y2).dropna()
    out = seasonal_trend(d, units)
    out["method"] = "paired_difference_" + out.get("method", "")
    return out
