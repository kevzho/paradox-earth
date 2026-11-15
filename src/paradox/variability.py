"""S4: empirical variability baseline, then ARIMA mean model + EGARCH(1,1) conditional volatility with an
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


def _block_bootstrap(x: np.ndarray, block: int, rng) -> np.ndarray:
    n = len(x)
    starts = rng.integers(0, max(n - block + 1, 1), size=int(np.ceil(n / block)))
    return np.concatenate([x[s:s + block] for s in starts])[:n]


def empirical_variability(anom: pd.Series, window: int = 36, block: int = 12, n_boot: int = 2000,
                          split: str | None = None, seed: int = 0) -> dict:
    """Rolling variance, early-vs-late variance ratio (Brown-Forsythe test + moving-block bootstrap CI on the
    log ratio), and robust MAD ratio. `split` = first date of the late period (default: midpoint)."""
    x = anom.dropna()
    if len(x) < 48:
        return {"status": "insufficient_data", "n": int(len(x))}
    split_ts = pd.Timestamp(split) if split else x.index[len(x) // 2]
    early, late = x[x.index < split_ts].values, x[x.index >= split_ts].values
    ratio = float(np.var(late, ddof=1) / np.var(early, ddof=1))
    bf = stats.levene(early, late, center="median")
    rng = np.random.default_rng(seed)
    boots = np.array([np.log(np.var(_block_bootstrap(late, block, rng), ddof=1) /
                             np.var(_block_bootstrap(early, block, rng), ddof=1)) for _ in range(n_boot)])
    lo, hi = np.exp(np.percentile(boots, [2.5, 97.5]))
    mad = lambda a: stats.median_abs_deviation(a, scale="normal")  # noqa: E731
    roll = x.rolling(window, min_periods=int(window * 2 / 3)).var()
    return {
        "status": "ok",
        "n_early": int(len(early)), "n_late": int(len(late)), "split": str(split_ts.date()),
        "variance_ratio_late_over_early": ratio,
        "variance_ratio_ci95": [float(lo), float(hi)],
        "brown_forsythe_p": float(bf.pvalue),
        "mad_ratio_late_over_early": float(mad(late) / mad(early)),
        "rolling_variance": roll,
        "method": f"BF test + moving-block bootstrap (block={block}, B={n_boot}); rolling window={window}",
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
