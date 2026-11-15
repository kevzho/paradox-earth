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


def test_tile_series_matches_region_loop(ds):
    for tile in (0.5, 1.25):
        fast = gldas.tile_series(ds, "soil_water_0_200cm", tile)
        slow = gldas.region_series(ds, "soil_water_0_200cm", gldas.tile_regions(ds, tile))
        assert list(fast.columns) == list(slow.columns)
        pd.testing.assert_frame_equal(fast, slow, check_names=False, check_freq=False, rtol=1e-6)


def test_tile_ids_grid_aligned():
    g = gldas.TileGrid(np.array([-0.125, 0.125, 0.375]), np.array([-0.125, 0.125]), 0.25)
    assert g.ids == ["tile_-000.25_-0000.25", "tile_-000.25_+0000.00", "tile_+000.00_-0000.25",
                     "tile_+000.00_+0000.00", "tile_+000.25_-0000.25", "tile_+000.25_+0000.00"]


def test_block_mean_preserves_water_budget(ds):
    stack = np.stack([ds[v].values for v in ["land_water_store", "net_flux"]], axis=1)  # (T, 2, H, W)
    c = np.stack([gldas.block_mean(s, ds.lat.values, 2) for s in stack])
    assert c.shape[-2:] == (ds.sizes["lat"] // 2, ds.sizes["lon"] // 2)
    resid = (c[1:, 0] - c[:-1, 0]) - c[1:, 1]
    assert np.nanmax(np.abs(resid)) < 0.05 and np.isfinite(c).any()


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


def test_egarch_degenerate_series_does_not_crash():
    flat = pd.Series(np.full(240, 5.0), pd.date_range("2000-01-01", periods=240, freq="MS"))
    assert variability.arima_egarch(flat)["egarch_status"] == "unavailable"


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

def test_streaming_ingest_matches_in_memory(raw_dir, ds, tmp_path):
    from paradox.pipeline import ingest
    (tmp_path / "manifests").mkdir()
    ing = ingest(raw_dir, tmp_path, tile_deg=1.25, variables=("soil_water_0_200cm",), pinn_coarsen=1)
    assert ing.report == qc.qc_dataset(ds, [v for v in gldas.ALL_VARS if v in ds])
    pd.testing.assert_frame_equal(ing.series["soil_water_0_200cm"], gldas.tile_series(ds, "soil_water_0_200cm", 1.25),
                                  check_names=False, check_freq=False)
    arr, mask, months = ing.pinn.arrays()
    ref_arr, ref_mask, ref_months = arrays_from_dataset(ds)
    np.testing.assert_allclose(arr, ref_arr, atol=1e-4)
    assert (mask == ref_mask).all() and (months == ref_months).all()


def test_parallel_stats_match_serial(raw_dir, tmp_path):
    from paradox.pipeline import ingest, run_stats
    for sub in ["manifests", "volatility"]:
        (tmp_path / sub).mkdir()
    ing = ingest(raw_dir, tmp_path, tile_deg=1.25, variables=("soil_water_0_200cm",), pinn=False)
    _, serial = run_stats(ing.series, ing.units, tmp_path, workers=1)
    _, parallel = run_stats(ing.series, ing.units, tmp_path, workers=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_pipeline_end_to_end(raw_dir, tmp_path):
    from paradox.pipeline import run_all
    out = run_all(raw_dir, tmp_path, tile_deg=1.25, variables=("soil_water_0_200cm",), workers=1,
                  pinn_cfg=PINNConfig(epochs=2, alignment="forward", hidden=8), pinn_coarsen=2)
    recs = json.loads((tmp_path / "records.json").read_text())
    assert recs and recs[0]["trend"]["status"] == "ok"
    assert recs[0]["variability"]["egarch_status"] in {"fitted", "inconclusive", "unavailable"}
    assert (tmp_path / "pinn_ablation.json").exists() and (tmp_path / "manifests").is_dir()
    assert (tmp_path / "tiles.csv").exists() and out["pinn"]["coarsen"] == 2
    assert out["qc"]["passed"]


# ---- spatial statistics (§7) -------------------------------------------------------------------------------

def _lattice(n_lat=30, n_lon=40, west=-180.0, deg=1.0):
    rows = [(f"tile_{s:+07.2f}_{w:+08.2f}", w, s, w + deg, s + deg)
            for s in np.arange(0, n_lat) * deg for w in west + np.arange(n_lon) * deg]
    return pd.DataFrame(rows, columns=["region_id", "west", "south", "east", "north"])


def test_lattice_neighbors_queen_and_dateline_wrap():
    from paradox import spatial
    t = _lattice(n_lat=5, n_lon=360)  # spans 360 degrees -> wraps
    nb = spatial.lattice_neighbors(t, t.region_id)
    assert len(nb["tile_+002.00_+0010.00"]) == 8 and len(nb["tile_+000.00_+0010.00"]) == 5
    assert "tile_+002.00_+0179.00" in nb["tile_+002.00_-0180.00"]
    part = _lattice(n_lat=5, n_lon=20)  # regional -> no wrap
    assert "tile_+002.00_-0161.00" not in spatial.lattice_neighbors(part, part.region_id)["tile_+002.00_-0180.00"]


def test_lisa_recovers_planted_cluster_and_stays_quiet_on_noise():
    from paradox import spatial
    t = _lattice()
    rng = np.random.default_rng(0)
    v = pd.Series(rng.standard_normal(len(t)), index=t.region_id)
    planted = (t.south.between(10, 17) & t.west.between(-170, -163)).to_numpy()
    v[planted] += 2.0
    table, s = spatial.analyse_field(v, t, permutations=9999)
    lab = table["label"].reindex(v.index)
    assert lab[planted].eq("HH_cluster").mean() > 0.5
    assert lab[~planted].isin(["HH_cluster", "LL_cluster"]).mean() < 0.02
    assert s["global_moran"]["I"] > 0 and s["global_moran"]["p_sim_one_sided"] < 0.01
    null, s0 = spatial.analyse_field(pd.Series(rng.standard_normal(len(t)), index=t.region_id), t, permutations=9999)
    assert null["label"].ne("inconclusive").mean() < 0.01


def test_lisa_bh_calibrated_on_heavy_tailed_null():
    """Regression: BH on a normal approximation of the permutation z labelled ~18 tiles per t(2) null map."""
    from paradox import spatial
    t = _lattice()
    rng = np.random.default_rng(3)
    for draw in (rng.standard_t(2, len(t)), rng.lognormal(0, 1.5, len(t))):
        table, s = spatial.analyse_field(pd.Series(draw, index=t.region_id), t, permutations=9999)
        assert table["label"].ne("inconclusive").sum() <= 2, s["label_counts"]


def test_spatial_fills_records(raw_dir, tmp_path):
    from paradox.pipeline import run_all
    run_all(raw_dir, tmp_path, tile_deg=0.5, variables=("soil_water_0_200cm",), workers=1, pinn=False,
            spatial_perms=99)
    def no_nan(x):
        raise ValueError(f"non-strict JSON constant {x}")
    recs = json.loads((tmp_path / "records.json").read_text(), parse_constant=no_nan)
    json.loads((tmp_path / "spatial_summary.json").read_text(), parse_constant=no_nan)
    assert all(r["spatial"]["inference_status"] in {"evaluated", "not_evaluated"} for r in recs)
    assert any(r["spatial"]["inference_status"] == "evaluated" for r in recs)
    assert (tmp_path / "spatial_summary.json").exists()



# ---- regressions from review ---------------------------------------------------------------------------------

def test_failed_egarch_keeps_trend_and_baseline(monkeypatch):
    from paradox import pipeline

    def boom(*a, **k):
        raise np.linalg.LinAlgError("Singular matrix")
    monkeypatch.setattr(pipeline.variability, "arima_egarch", boom)
    tr, var_rec, row, cv = pipeline._analyse(("v", "r", _series(slope=0.01), "mm", None, True))
    assert tr["status"] == "ok" and row["slope_per_decade"] is not None
    assert var_rec["baseline_change"]["variance_ratio_late_over_early"] is not None
    assert var_rec["egarch_status"] == "unavailable" and "Singular" in var_rec["egarch"]["reason"] and cv is None


def test_qc_tolerates_float32_saturation():
    import xarray as xr
    d = xr.Dataset({"SoilMoi10_40cm_inst": (("time", "lat", "lon"), np.full((1, 1, 1), 300.0001, np.float32),
                                            {"units": "mm"})},
                   coords={"time": pd.to_datetime(["2000-01-01"]), "lat": [0.125], "lon": [0.125]})
    assert qc.qc_dataset(d)["passed"]
    d["SoilMoi10_40cm_inst"][:] = 301.0
    assert not qc.qc_dataset(d)["passed"]


def test_partial_edge_tiles_dropped(ds):
    # synthetic lat spans 30.0-32.0: with 1.25-degree tiles the 31.25-32.5 row is cut by the domain edge
    df = gldas.tile_series(ds, "soil_water_0_200cm", 1.25)
    assert all(c.startswith("tile_+030.00") for c in df.columns) and df.shape[1] >= 1
    with pytest.raises(ValueError):
        gldas.TileGrid(ds.lat.values, ds.lon.values, 0.3)


def test_spatial_islands_get_knn_label_and_gi_agreement_none():
    from paradox import spatial
    t = pd.concat([_lattice(n_lat=6, n_lon=6), _lattice(n_lat=1, n_lon=1, west=-170.0)], ignore_index=True)
    v = pd.Series(np.random.default_rng(1).standard_normal(len(t)), index=t.region_id)
    table, s = spatial.analyse_field(v, t, permutations=99)
    iso = "tile_+000.00_-0170.00"
    assert table.loc[iso, "label"] == "no_neighbors" and isinstance(table.loc[iso, "label_knn"], str)
    if not table["label"].isin(["HH_cluster", "LL_cluster"]).any():
        assert s["robustness"]["gi_star_sign_agrees_on_clusters"] is None


def test_gi_star_matches_esda_on_positive_data():
    from esda.getisord import G_Local
    from paradox import spatial
    t = _lattice(n_lat=10, n_lon=12)
    y = np.random.default_rng(4).lognormal(0, 0.5, len(t))
    w, _ = spatial.to_w(spatial.lattice_neighbors(t, t.region_id))
    ours = spatial.gi_star(y, w)["gi_z"].to_numpy()
    import copy
    wb = copy.deepcopy(w)
    wb.transform = "B"
    ref = G_Local(y, wb, transform="B", star=True, permutations=0).Zs
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-8)


def test_review2_edge_cases(ds, tmp_path):
    import xarray as xr
    from paradox import spatial
    # QC: zero floor is not loosened by the upper-bound tolerance
    d = xr.Dataset({"SWE_inst": (("time", "lat", "lon"), np.full((1, 1, 1), -9.0, np.float32), {"units": "mm"})},
                   coords={"time": pd.to_datetime(["2000-01-01"]), "lat": [0.125], "lon": [0.125]})
    assert not qc.qc_dataset(d)["passed"]
    # tiling: descending latitude gives the same tiles; global grids need tile_deg dividing 360
    asc = gldas.TileGrid(ds.lat.values, ds.lon.values, 0.5)
    desc = gldas.TileGrid(ds.lat.values[::-1], ds.lon.values, 0.5)
    assert sorted(asc.boxes) == sorted(desc.boxes) and asc.boxes
    glon = -179.875 + 0.25 * np.arange(1440)
    with pytest.raises(ValueError):
        gldas.TileGrid(np.array([0.125, 0.375]), glon, 1.75)
    assert len(gldas.TileGrid(np.array([0.125]), glon, 0.25).boxes) == 1440
    assert not gldas.TileGrid(np.array([0.125]), glon, 1.0).boxes  # one cell tall cannot fill a 1-degree tile
    # spatial: constant map and an all-islands map are reported, not labelled or fatal
    t = _lattice(n_lat=5, n_lon=5)
    with pytest.raises(ValueError):
        spatial.analyse_field(pd.Series(1.0, index=t.region_id), t, permutations=99)
    recs = [{"region_id": rid, "variable": "v", "trend": {"slope": 1.0},
             "variability": {"baseline_change": {"variance_ratio_late_over_early": 1.0 + i}}}
            for i, rid in enumerate(t.region_id)]
    summ = spatial.run_spatial(recs, t, tmp_path, permutations=99)
    assert summ["v__trend_slope"]["status"] == "error" and summ["v__log_variance_ratio"]["status"] == "ok"
