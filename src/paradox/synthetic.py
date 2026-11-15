"""Synthetic GLDAS-like monthly files for smoke tests. Native GLDAS variable names and units, with a
bucket model so storage changes obey  dS = P - ET - R  exactly (in the end-of-month sense)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LAYER_FRAC = [0.05, 0.15, 0.30, 0.50]  # share of 0-200 cm storage in each Noah layer


def make_synthetic(out_dir: str | Path, n_years: int = 16, nlat: int = 12, nlon: int = 16,
                   start: str = "2000-01-01", seed: int = 0, write: bool = True) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, periods=n_years * 12, freq="MS")
    lat = 30.0 + 0.25 * np.arange(nlat) + 0.125
    lon = -100.0 + 0.25 * np.arange(nlon) + 0.125
    nt = len(time)
    m = time.month.values
    season = np.sin(2 * np.pi * (m - 4) / 12)[:, None, None]
    tt = np.arange(nt)[:, None, None] / nt
    yy, xx = np.meshgrid(np.arange(nlat), np.arange(nlon), indexing="ij")

    T = 285 + 10 * season - 0.3 * yy[None] + rng.normal(0, 1, (nt, nlat, nlon))
    p_mean = 70 * (1 + 0.4 * season) * (1 + 0.02 * xx[None]) * (1 - 0.25 * tt * (xx[None] > nlon / 2))  # drying east
    # volatility clustering in the NE quadrant during the late period (exercise EGARCH / variance tests)
    vol = np.ones((nt, nlat, nlon))
    ne = (yy >= nlat // 2) & (xx >= nlon // 2)
    h = np.zeros(nt)
    e = rng.standard_normal(nt)
    for t in range(1, nt):
        h[t] = 0.85 * h[t - 1] + 0.35 * (abs(e[t - 1]) - 0.8) - 0.2 * e[t - 1]
    vol[:, ne] = np.exp(0.5 * h)[:, None] * (1 + tt[:, 0, :])
    P = np.clip(p_mean * (1 + 0.35 * vol * rng.standard_normal((nt, nlat, nlon))), 0, None)

    smax, S = 600.0, np.full((nlat, nlon), 400.0)
    S_out, ET_out, Qs_out, Qsb_out = (np.zeros((nt, nlat, nlon)) for _ in range(4))
    for t in range(nt):
        et = np.minimum(S, 2.5 * np.clip(T[t] - 273.0, 0, None) * S / smax)
        drain = 0.03 * S
        excess = np.clip(S + P[t] - et - drain - smax, 0, None)
        S = S + P[t] - et - drain - excess
        S_out[t], ET_out[t], Qs_out[t], Qsb_out[t] = S, et, excess, drain

    days = time.days_in_month.values[:, None, None].astype(float)
    data = {f"SoilMoi{n}_inst": S_out * f for n, f in
            zip(["0_10cm", "10_40cm", "40_100cm", "100_200cm"], LAYER_FRAC)}
    data.update({
        "SWE_inst": np.zeros_like(S_out), "CanopInt_inst": np.zeros_like(S_out),
        "Rainf_f_tavg": P / (86400 * days), "Evap_tavg": ET_out / (86400 * days),
        "Qs_acc": Qs_out / (8 * days), "Qsb_acc": Qsb_out / (8 * days), "Tair_f_inst": T,
    })
    units = {"Rainf_f_tavg": "kg m-2 s-1", "Evap_tavg": "kg m-2 s-1", "Qs_acc": "kg m-2", "Qsb_acc": "kg m-2",
             "Tair_f_inst": "K"}
    ocean = (yy == 0) & (xx < 3)  # a few masked cells to exercise NaN handling
    ds = xr.Dataset(
        {k: (("time", "lat", "lon"), np.where(ocean[None], np.nan, v).astype("float32"),
             {"units": units.get(k, "kg m-2")}) for k, v in data.items()},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"title": "SYNTHETIC GLDAS-like test data - not NASA data"},
    )
    if write:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(time):
            ds.isel(time=[i]).to_netcdf(out_dir / f"GLDAS_NOAH025_M.A{t:%Y%m}.021.nc4")
    return ds
