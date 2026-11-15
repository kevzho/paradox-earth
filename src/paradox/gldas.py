"""GLDAS-2.1 Noah monthly 0.25° (GLDAS_NOAH025_M) ingestion, unit harmonisation and regional aggregation.

Unit conventions after `to_monthly_water_units`:
  storages (soil moisture layers, SWE, canopy)  -> mm  (1 kg m-2 of water == 1 mm)
  fluxes   (precip, ET, runoff)                 -> mm per month (monthly TOTAL, not mean rate)
  air temperature                               -> K (unchanged)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATASET = "GLDAS_NOAH025_M"
VERSION = "2.1"

STORAGE_VARS = ["SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst", "SoilMoi40_100cm_inst",
                "SoilMoi100_200cm_inst", "SWE_inst", "CanopInt_inst"]
RATE_VARS = ["Rainf_f_tavg", "Evap_tavg"]            # kg m-2 s-1 (monthly mean of rate)
ACC_VARS = ["Qs_acc", "Qsb_acc"]                     # kg m-2 per 3 h (monthly mean of 3-hourly accumulations)
TEMP_VARS = ["Tair_f_inst"]
ALL_VARS = STORAGE_VARS + RATE_VARS + ACC_VARS + TEMP_VARS

PHYSICAL_RANGES = {  # loose plausibility bounds in harmonised units, used by QC
    "SoilMoi0_10cm_inst": (0, 100), "SoilMoi10_40cm_inst": (0, 300), "SoilMoi40_100cm_inst": (0, 600),
    "SoilMoi100_200cm_inst": (0, 1000), "SWE_inst": (0, 1e5), "CanopInt_inst": (0, 10),
    "Rainf_f_tavg": (0, 3000), "Evap_tavg": (-200, 1000), "Qs_acc": (0, 3000), "Qsb_acc": (0, 3000),
    "Tair_f_inst": (180, 340),
}


def open_gldas(paths, variables=None, bbox=None) -> xr.Dataset:
    """Open monthly GLDAS files (one per month) and concatenate along time.

    bbox = (west, south, east, north) in degrees; subset is applied per file to save memory.
    """
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        paths = sorted(p.glob("*.nc4")) + sorted(p.glob("*.nc")) if p.is_dir() else [p]
    paths = sorted(map(Path, paths))
    if not paths:
        raise FileNotFoundError("no GLDAS files found")
    variables = variables or ALL_VARS
    parts = []
    for f in paths:
        with xr.open_dataset(f) as ds:
            keep = [v for v in variables if v in ds]
            ds = ds[keep]
            if bbox is not None:
                w, s, e, n = bbox
                ds = ds.sel(lon=slice(w, e), lat=slice(s, n))
            parts.append(ds.load())
    ds = xr.concat(parts, dim="time").sortby("time")
    ds.attrs.update(source_files=[p.name for p in paths], dataset=DATASET, version=VERSION)
    return ds


def to_monthly_water_units(ds: xr.Dataset) -> xr.Dataset:
    """Convert GLDAS native units to mm / mm-per-month totals (see module docstring)."""
    out = ds.copy()
    t = pd.DatetimeIndex(ds["time"].values)
    days = xr.DataArray(t.days_in_month.values.astype(float), dims="time", coords={"time": ds["time"]})
    for v in RATE_VARS:
        if v in out:
            out[v] = ds[v] * 86400.0 * days
            out[v].attrs.update(units="mm month-1", note="monthly total from mean rate")
    for v in ACC_VARS:
        if v in out:
            out[v] = ds[v] * 8.0 * days  # 8 three-hour periods per day
            out[v].attrs.update(units="mm month-1", note="monthly total from mean 3-h accumulation")
    for v in STORAGE_VARS:
        if v in out:
            out[v].attrs.update(units="mm")
    out.attrs["harmonised_units"] = "storage mm; flux mm/month totals; T K"
    return out


def add_derived(ds: xr.Dataset) -> xr.Dataset:
    """Soil-water storage target (Manifest §10.2) and net flux. Not total terrestrial water storage."""
    ds = ds.copy()
    soil = [v for v in STORAGE_VARS[:4] if v in ds]
    if soil:
        ds["soil_water_0_200cm"] = sum(ds[v] for v in soil)
        ds["soil_water_0_200cm"].attrs.update(units="mm", note="sum of Noah soil layers 0-200 cm")
    stores = [v for v in STORAGE_VARS if v in ds]
    if stores:
        ds["land_water_store"] = sum(ds[v] for v in stores)
        ds["land_water_store"].attrs.update(
            units="mm", note="soil 0-200 cm + SWE + canopy; excludes groundwater/surface water (NOT TWS)")
    if all(v in ds for v in ["Rainf_f_tavg", "Evap_tavg", "Qs_acc", "Qsb_acc"]):
        ds["runoff"] = ds["Qs_acc"] + ds["Qsb_acc"]
        ds["net_flux"] = ds["Rainf_f_tavg"] - ds["Evap_tavg"] - ds["runoff"]
        ds["net_flux"].attrs.update(units="mm month-1", note="P - ET - (Qs + Qsb)")
    return ds


def cell_weights(ds: xr.Dataset, area_var: str | None = None) -> xr.DataArray:
    """Area weights: true cell area if supplied, else cos(latitude) (regular lat/lon grid approximation)."""
    if area_var and area_var in ds:
        return ds[area_var]
    w = np.cos(np.deg2rad(ds["lat"]))
    return w.broadcast_like(ds["lon"]).transpose("lat", "lon")


def regional_mean(da: xr.DataArray, weights: xr.DataArray, min_coverage: float = 0.5) -> pd.Series:
    """Area-weighted mean over lat/lon, NaN-aware. Months with < min_coverage of weight become NaN."""
    valid = da.notnull()
    wsum = weights.where(valid).sum(("lat", "lon"))
    total = weights.where(valid.any("time")).sum()
    mean = (da * weights).sum(("lat", "lon"), skipna=True) / wsum
    mean = mean.where(wsum / total >= min_coverage)
    return mean.to_series()


def tile_regions(ds: xr.Dataset, tile_deg: float) -> dict[str, tuple[float, float, float, float]]:
    """Split the domain into square tiles; returns {region_id: (w, s, e, n)}."""
    lon0, lon1 = float(ds.lon.min()), float(ds.lon.max())
    lat0, lat1 = float(ds.lat.min()), float(ds.lat.max())
    regions = {}
    for s in np.arange(lat0, lat1 + 1e-9, tile_deg):
        for w in np.arange(lon0, lon1 + 1e-9, tile_deg):
            rid = f"tile_{s:+07.2f}_{w:+08.2f}"
            regions[rid] = (w, s, w + tile_deg, s + tile_deg)
    return regions


def region_series(ds: xr.Dataset, var: str, regions: dict, min_coverage: float = 0.5) -> pd.DataFrame:
    """Area-weighted regional series for each region box; columns = region ids."""
    weights = cell_weights(ds)
    cols = {}
    for rid, (w, s, e, n) in regions.items():
        sub = ds[var].where((ds.lon >= w) & (ds.lon < e) & (ds.lat >= s) & (ds.lat < n))
        if int(sub.notnull().any("time").sum()) == 0:
            continue
        cols[rid] = regional_mean(sub, weights, min_coverage)
    return pd.DataFrame(cols)
