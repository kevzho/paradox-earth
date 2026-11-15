"""Smoke tests: every method runs on synthetic GLDAS-like data and returns sane, correctly-shaped output."""
import json

import numpy as np
import pandas as pd
import pytest

from paradox import contract, gldas, qc, seasonal, trend, variability
from paradox.pinn import (PINNConfig, Prepared, ablation, arrays_from_dataset, baselines, evaluate,
                          predict_storage, train)
from paradox.synthetic import make_synthetic


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("gldas")
    make_synthetic(d, n_years=14, nlat=8, nlon=10)
    return d


@pytest.fixture(scope="session")
def ds(raw_dir):
    return gldas.add_derived(gldas.to_monthly_water_units(gldas.open_gldas(raw_dir)))


def _series(n=240, slope=0.0, sigma=1.0, seed=0):
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    y = 10 * np.sin(2 * np.pi * idx.month / 12) + slope * np.arange(n) + sigma * rng.standard_normal(n)
    return pd.Series(y, idx)


# ---- data layer -------------------------------------------------------------------------------------------

def test_open_and_units(ds):
    assert ds.sizes["time"] == 14 * 12 and "net_flux" in ds and "land_water_store" in ds
    assert ds["Rainf_f_tavg"].attrs["units"] == "mm month-1"
    # unit round-trip: synthetic bucket closes the budget exactly in end-of-month form
    s = ds["land_water_store"].values
    resid = (s[1:] - s[:-1]) - ds["net_flux"].values[1:]
    assert np.nanmax(np.abs(resid)) < 0.05


def test_bbox_subset(raw_dir):
    sub = gldas.open_gldas(raw_dir, variables=["Tair_f_inst"], bbox=(-99.6, 30.5, -98.5, 31.5))
    assert 0 < sub.sizes["lon"] < 10 and 0 < sub.sizes["lat"] < 8


def test_regional_mean_area_weighted(ds):
    w = gldas.cell_weights(ds)
    s = gldas.regional_mean(ds["Tair_f_inst"], w)
    assert s.notna().all() and 250 < s.mean() < 320
    regs = gldas.tile_regions(ds, 1.0)
    df = gldas.region_series(ds, "soil_water_0_200cm", regs)
    assert df.shape[0] == ds.sizes["time"] and df.shape[1] >= 1


def test_qc(ds):
    rep = qc.qc_dataset(ds, [v for v in gldas.ALL_VARS if v in ds])
    assert rep["passed"] and rep["chronology"]["regular_monthly"]
    gap = qc.check_chronology(pd.to_datetime(["2000-01-01", "2000-02-01", "2000-04-01"]))
    assert gap["missing_months"] == 1 and not gap["regular_monthly"]


def test_contract(tmp_path):
    m = contract.SourceManifest("NASA", "X", "v", "1", "mm", "monthly", "0.25", "0.25")
    p = m.write(tmp_path / "m.json")
    assert json.loads(p.read_text())["dataset"] == "X"
    rec = contract.analysis_record("r1", "X", "v", ["2000", "2010"])
    assert rec["variability"]["egarch_status"] == "not_evaluated"


# ---- statistics -------------------------------------------------------------------------------------------

def test_seasonal_training_only():
    y = _series()
    a = seasonal.anomalies(y, ref_period=("2000-01-01", "2009-12-01"))
    assert set(a.columns) == {"raw", "climatology", "anomaly"}
    assert abs(a["anomaly"].loc[:"2009"].groupby(a.index[:120].month).mean()).max() < 1e-9
    with pytest.raises(ValueError):
        seasonal.monthly_climatology(y, ("2000-01-01", "2000-06-01"))


def test_trend_recovers_slope():
    y = _series(slope=0.01)  # 0.01 / month = 1.2 / decade
    r = trend.seasonal_trend(y, "mm")
    assert r["status"] == "ok" and r["ci95"][0] < 1.2 < r["ci95"][1] and r["p_value"] < 1e-6
    d = trend.slope_difference(y, _series(seed=1), "mm")
    assert d["ci95"][0] < 1.2 < d["ci95"][1]
    assert trend.seasonal_trend(y.iloc[:10])["status"] == "insufficient_data"


def test_empirical_variability_detects_change():
    y = _series(sigma=1.0)
    y.iloc[120:] = y.iloc[120:] * 1 + np.random.default_rng(5).normal(0, 2.0, 120)
    r = variability.empirical_variability(seasonal.anomalies(y)["anomaly"], n_boot=300)
    assert r["variance_ratio_late_over_early"] > 2 and r["brown_forsythe_p"] < 0.01
    assert r["variance_ratio_ci95"][0] > 1


def test_arima_egarch_on_egarch_process():
    rng = np.random.default_rng(1)
    n, h, x = 600, np.zeros(600), np.zeros(600)
    z = rng.standard_normal(n)
    for t in range(1, n):
        h[t] = -0.05 + 0.9 * h[t - 1] + 0.3 * (abs(z[t - 1]) - np.sqrt(2 / np.pi)) - 0.15 * z[t - 1]
        x[t] = 0.5 * x[t - 1] + np.exp(h[t] / 2) * z[t]
    s = pd.Series(x, pd.date_range("1950-01-01", periods=n, freq="MS"))
    r = variability.arima_egarch(s)
    assert r["egarch_status"] == "fitted", r.get("reason")
    assert 0.6 < r["persistence_beta"] < 1 and r["asymmetry_gamma"] < 0
    assert r["conditional_volatility"].notna().all()


def test_egarch_gate_rejects_white_noise_and_short():
    assert variability.arima_egarch(_series(n=60))["egarch_status"] == "unavailable"
    r = variability.arima_egarch(pd.Series(np.random.default_rng(2).standard_normal(240),
                                           pd.date_range("2000-01-01", periods=240, freq="MS")))
    assert r["egarch_status"] in {"inconclusive", "fitted"}
    assert r["gates"]["arch_evidence"] is False or r["egarch_status"] == "fitted"


# ---- physics-informed model ------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def prep(ds):
    arr, mask, months = arrays_from_dataset(ds)
    return Prepared(arr, mask, months, PINNConfig(epochs=3, alignment="forward", hidden=8, patch=6))


def test_pinn_prepare_no_leakage(prep):
    sp = prep.split
    assert sp["train"].max() < sp["val"].min() <= sp["val"].max() < sp["test"].min()
    assert prep.holdout_cells.any() and not (prep.holdout_cells & prep.train_cells).any()
    x, y, f_t, p_next = prep.batch(sp["train"][:2])
    assert x.shape[1] == prep.n_in and y.shape[1] == 4 and np.isfinite(x.numpy()).all()


def test_pinn_physics_residual_zero_on_truth(prep):
    from paradox.pinn import water_balance_residual
    import torch
    a, ts = prep.arr, prep.split["test"]
    ds_ = torch.from_numpy(a[ts + 1, 0] - a[ts, 0])
    f_t = torch.zeros_like(ds_)
    r = water_balance_residual(ds_, torch.from_numpy(a[ts + 1, 1]), torch.from_numpy(a[ts + 1, 2]),
                               torch.from_numpy(a[ts + 1, 3]), f_t, "forward")
    assert float(r[:, prep.mask].abs().max()) < 0.05


def test_pinn_train_eval_baselines(prep):
    model, hist = train(prep)
    assert len(hist) == 3 and all(np.isfinite(h["val_pred"]) for h in hist)
    s, res = predict_storage(model, prep, prep.split["test"])
    assert s.shape == prep.arr[prep.split["test"], 0].shape and np.isfinite(s).all()
    ev = evaluate(model, prep)
    assert {"pinn", "persistence", "seasonal_naive", "linear_ar"} <= set(ev["in_region"])
    assert "spatial_holdout" in ev
    b = baselines(prep, prep.split["test"])
    assert all(v.shape == s.shape for v in b.values())


def test_pinn_ablation_lambda_reduces_budget_residual(ds):
    cfg = PINNConfig(epochs=25, alignment="forward", hidden=16, seed=0)
    r = ablation(ds, cfg, lams=(0.0, 5.0))
    m0 = r["runs"]["lambda=0.0"]["test"]["in_region"]["pinn"]
    m1 = r["runs"]["lambda=5.0"]["test"]["in_region"]["pinn"]
    assert np.isfinite(m0["rmse_mm"]) and np.isfinite(m1["rmse_mm"])
    assert m1["budget_residual_mae_mm"] < m0["budget_residual_mae_mm"]


# ---- end to end ------------------------------------------------------------------------------------------

def test_pipeline_end_to_end(raw_dir, tmp_path):
    from paradox.pipeline import run_all
    out = run_all(raw_dir, tmp_path, pinn_cfg=PINNConfig(epochs=2, alignment="forward", hidden=8),
                  stats_kwargs={"tile_deg": 1.25, "variables": ("soil_water_0_200cm",)})
    recs = json.loads((tmp_path / "records.json").read_text())
    assert recs and recs[0]["trend"]["status"] == "ok"
    assert recs[0]["variability"]["egarch_status"] in {"fitted", "inconclusive", "unavailable"}
    assert (tmp_path / "pinn_ablation.json").exists() and (tmp_path / "manifests").is_dir()
    assert out["qc"]["passed"]
