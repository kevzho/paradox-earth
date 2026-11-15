"""S4: calibrated variance-change test (prewhitened Brown-Forsythe), then ARIMA mean model + EGARCH(1,1)
conditional volatility with an
explicit eligibility gate (Manifest §6 S4). Input is a seasonal-anomaly series (see seasonal.anomalies).

EGARCH(1,1) as parameterised by `arch` (matches the manifest):
    log s2_t = omega + beta log s2_{t-1} + alpha (|z_{t-1}| - E|z|) + gamma z_{t-1}
gamma < 0  => negative anomalies (e.g. drier-than-normal soil) raise next-month volatility more than positive ones.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------------------------------------------- empirical baseline


def _ar_innovations(x: np.ndarray, months: np.ndarray, pmax: int = 4):
    """Residuals of x_t on [1, t, month dummies, x_{t-1..t-p}], p in 1..pmax by AIC on a common sample."""
    n = len(x)
    tt = np.arange(n, dtype=float)
    md = np.column_stack([(months == m) for m in range(2, 13)]).astype(float)

    def fit(q, start):
        lags = [x[start - j:n - j] for j in range(1, q + 1)]
        X = np.column_stack([np.ones(n - start), tt[start:], md[start:]] + lags)
        b, *_ = np.linalg.lstsq(X, x[start:], rcond=None)
        return x[start:] - X @ b, X.shape[1]

    aic = []
    for q in range(1, pmax + 1):
        e, k = fit(q, pmax)
        aic.append((n - pmax) * np.log(np.mean(e ** 2)) + 2 * k)
    p = int(np.argmin(aic)) + 1
    return fit(p, p)[0], p


def prewhitened_variance_change(anom: pd.Series, split: str | None = None, pmax: int = 4) -> dict:
    """Prewhitened, state-adjusted Brown-Forsythe test of an early-vs-late change in innovation variance.

    1. AR(p) innovations with trend + month dummies fitted on the whole record (a trend, a mean step or a
       reference-window climatology no longer masquerades as a variance change).
    2. z = |innovation - median of its half|; regress z on [1, late, month dummies, L, L^2] with L the standardised
       lagged level (bounded soil water has level-dependent variance).
    3. t of `late` deflated by sqrt((1-r)/(1+r)), r = lag-1 ACF of the z residuals; two-sided t(n-k) p-value.
    Simulated size 3.9-6.1% under AR(1) phi 0..0.95, trends, steps, seasonal heteroskedasticity and real-tile
    surrogates, versus 43-64% for Brown-Forsythe on raw anomalies at phi 0.87-0.95.
    """
    y = anom.asfreq("MS") if anom.index.freq is None else anom
    if y.isna().any() or len(y) < 96:
        return {"status": "insufficient_data", "n": int(y.notna().sum())}
    x = y.values.astype(float)
    if not np.isfinite(x).all() or np.std(x) <= 1e-9 * max(1.0, np.abs(x).max()):
        return {"status": "degenerate", "n": len(x)}
    months = y.index.month.values
    split_ts = pd.Timestamp(split) if split else y.index[len(y) // 2]
    e, p = _ar_innovations(x, months, pmax)
    late = (y.index[p:] >= split_ts).astype(float)
    ea, eb = e[late == 0], e[late == 1]
    z = np.where(late == 1, np.abs(e - np.median(eb)), np.abs(e - np.median(ea)))
    lev = x[p - 1:-1]
    lev = (lev - lev.mean()) / lev.std()
    mon = months[p:]
    X = np.column_stack([np.ones(len(z)), late] + [(mon == m).astype(float) for m in range(2, 13)] + [lev, lev ** 2])
    xtxi = np.linalg.inv(X.T @ X)
    b = xtxi @ (X.T @ z)
    u = z - X @ b
    n, k = X.shape
    t = b[1] / np.sqrt((u @ u) / (n - k) * xtxi[1, 1])
    r = max(float(u[1:] @ u[:-1] / (u @ u)), 0.0)
    t_adj = t * np.sqrt((1 - r) / (1 + r))
    return {"status": "ok", "split": str(split_ts.date()), "ar_order": p,
            "innovation_variance_ratio_late_over_early": float(np.var(eb, ddof=1) / np.var(ea, ddof=1)),
            "pw_bf_t": float(t_adj), "pw_bf_p": float(2 * stats.t.sf(abs(t_adj), n - k)), "z_lag1_acf": r}


def empirical_variability(anom: pd.Series, window: int = 36, split: str | None = None) -> dict:
    """Early-vs-late variance change of seasonal anomalies. The TEST is the prewhitened, state-adjusted
    Brown-Forsythe (pw_bf_p) on innovation variance; the raw anomaly variance ratio, MAD ratio and rolling variance are
    descriptive only (the raw ratio is biased up ~7% when the climatology reference excludes the late years).
    `split` = first date of the late period (default: midpoint)."""
    x = anom.dropna()
    if len(x) < 48:
        return {"status": "insufficient_data", "n": int(len(x))}
    split_ts = pd.Timestamp(split) if split else x.index[len(x) // 2]
    early, late = x[x.index < split_ts].values, x[x.index >= split_ts].values
    pw = prewhitened_variance_change(anom, split=str(split_ts.date()))
    mad = lambda a: stats.median_abs_deviation(a, scale="normal")  # noqa: E731
    return {
        "status": pw["status"],
        "n_early": int(len(early)), "n_late": int(len(late)), "split": str(split_ts.date()),
        "innovation_variance_ratio_late_over_early": pw.get("innovation_variance_ratio_late_over_early"),
        "pw_bf_p": pw.get("pw_bf_p"), "ar_order": pw.get("ar_order"),
        "variance_ratio_late_over_early": float(np.var(late, ddof=1) / np.var(early, ddof=1)),  # descriptive
        "mad_ratio_late_over_early": float(mad(late) / mad(early)),  # descriptive
        "rolling_variance": x.rolling(window, min_periods=int(window * 2 / 3)).var(),
        "method": ("prewhitened state-adjusted Brown-Forsythe on AR(p) innovations (trend + month dummies, level "
                   f"control, z-persistence deflation); raw ratio / MAD / rolling window={window} descriptive"),
    }


# ---------------------------------------------------------------- ARIMA mean model


def fit_arima(anom: pd.Series, max_p: int = 2, max_q: int = 2, trend: str = "ct"):
    """Select ARMA(p,q) (d=0 on anomalies, deterministic trend) by AIC among fits that converged."""
    y = anom.asfreq("MS") if anom.index.freq is None else anom
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = ARIMA(y, order=(p, 0, q), trend=trend).fit()
                except Exception:  # noqa: BLE001 - singular / non-invertible starts
                    continue
            converged = bool(res.mle_retvals.get("converged", True)) if res.mle_retvals else True
            if converged and np.isfinite(res.aic) and (best is None or res.aic < best.aic):
                best = res
    return best


# ---------------------------------------------------------------- EGARCH with eligibility gate


def arima_egarch(anom: pd.Series, min_n: int = 120, lb_lags: int = 12, arch_alpha: float = 0.05,
                 diag_alpha: float = 0.05, max_missing: float = 0.05, require_arch_effect: bool = True) -> dict:
    """ARIMA mean model -> ARCH-LM test -> EGARCH(1,1) on residuals -> diagnostics -> gate.

    egarch_status: 'fitted' (passed every gate), 'inconclusive' (fitted but a gate failed), or
    'unavailable' (could not be attempted). Callers should ship the empirical baseline whenever status != fitted.
    """
    from arch import arch_model  # local import keeps module import cheap

    out = {"egarch_status": "unavailable", "gates": {}}
    n_obs, n_missing = int(anom.notna().sum()), int(anom.isna().sum())
    out["gates"]["sufficient_data"] = n_obs >= min_n
    out["gates"]["missing_ok"] = n_missing / max(len(anom), 1) <= max_missing
    if not (out["gates"]["sufficient_data"] and out["gates"]["missing_ok"]):
        out["reason"] = f"n={n_obs}, missing={n_missing}"
        return out

    arima = fit_arima(anom)
    out["gates"]["mean_model_converged"] = arima is not None
    if arima is None:
        out["reason"] = "no ARIMA order converged"
        return out
    resid = arima.resid.dropna()
    p, _, q = arima.model.order
    lb_mean = acorr_ljungbox(resid, lags=[lb_lags], model_df=p + q)["lb_pvalue"].iloc[0]
    out["arima"] = {"order": [p, 0, q], "aic": float(arima.aic), "ljung_box_resid_p": float(lb_mean)}
    # Informational only: LB on raw residuals is over-sized under ARCH effects. Mean-model adequacy is gated
    # below on the EGARCH-standardised residuals (diagnostics_ok).
    out["arima"]["raw_resid_white"] = bool(lb_mean > diag_alpha)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        lm_stat, lm_p, _, _ = het_arch(resid.values, nlags=lb_lags)
    out["arch_lm"] = {"stat": float(lm_stat), "p_value": float(lm_p), "lags": lb_lags}
    out["gates"]["arch_evidence"] = bool(lm_p < arch_alpha)

    scale = float(resid.std(ddof=1))
    if not (np.isfinite(scale) and scale > 0):  # e.g. constant series over ice sheets or hyper-arid tiles
        out["reason"] = "degenerate (zero-variance) residuals"
        return out
    z_in = resid / scale  # arch optimiser is best-behaved near unit variance
    res = arch_model(z_in, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal", rescale=False).fit(
        disp="off", show_warning=False)
    out["egarch_status"] = "inconclusive"
    out["gates"]["converged"] = bool(res.convergence_flag == 0)

    pr, se = res.params, res.std_err
    beta = float(pr["beta[1]"])
    out["params"] = {k: {"est": float(pr[k]), "se": float(se[k])} for k in pr.index}
    out["persistence_beta"] = beta
    out["asymmetry_gamma"] = float(pr["gamma[1]"])
    out["gates"]["plausible_params"] = bool(abs(beta) < 1 and np.all(np.isfinite(se.values)))

    z = pd.Series(res.std_resid, index=resid.index).dropna()
    lb_z = acorr_ljungbox(z, lags=[lb_lags])["lb_pvalue"].iloc[0]
    lb_z2 = acorr_ljungbox(z ** 2, lags=[lb_lags], model_df=2)["lb_pvalue"].iloc[0]
    out["diagnostics"] = {"ljung_box_z_p": float(lb_z), "ljung_box_z2_p": float(lb_z2),
                          "loglik": float(res.loglikelihood), "aic": float(res.aic)}
    out["gates"]["diagnostics_ok"] = bool(lb_z > diag_alpha and lb_z2 > diag_alpha)

    out["conditional_volatility"] = pd.Series(res.conditional_volatility * scale, index=resid.index)
    out["residual_scale"] = scale
    required = ["sufficient_data", "missing_ok", "mean_model_converged",
                "converged", "plausible_params", "diagnostics_ok"] + (["arch_evidence"] if require_arch_effect else [])
    if all(out["gates"][g] for g in required):
        out["egarch_status"] = "fitted"
    else:
        out["reason"] = "failed gates: " + ", ".join(g for g in required if not out["gates"][g])
    out["note"] = ("Conditional volatility is model-estimated; it does not by itself prove a persistent change in "
                   "environmental variance. gamma<0 => negative anomalies raise volatility more.")
    return out
