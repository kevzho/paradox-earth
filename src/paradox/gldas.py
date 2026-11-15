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


def list_files(paths) -> list[Path]:
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        paths = sorted(p.glob("*.nc4")) + sorted(p.glob("*.nc")) if p.is_dir() else [p]
    paths = sorted(map(Path, paths))
    if not paths:
        raise FileNotFoundError("no GLDAS files found")
    return paths


def load_file(path, variables=None, bbox=None) -> xr.Dataset:
    """Load one GLDAS file into memory, keeping only `variables` inside bbox = (west, south, east, north)."""
    variables = variables or ALL_VARS
    with xr.open_dataset(path) as ds:
        ds = ds[[v for v in variables if v in ds]]
        if bbox is not None:
            w, s, e, n = bbox
            ds = ds.sel(lon=slice(w, e), lat=slice(s, n))
        return ds.load()


def open_gldas(paths, variables=None, bbox=None) -> xr.Dataset:
    """Open monthly GLDAS files (one per month) and concatenate along time. Holds everything in memory, so use
    it for small domains only; `pipeline.ingest` streams file by file for continental/global runs."""
    paths = list_files(paths)
    ds = xr.concat([load_file(f, variables, bbox) for f in paths], dim="time").sortby("time")
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


class TileGrid:
    """Square tiles aligned to multiples of `tile_deg` (so ids are stable across domains), with each grid cell
    assigned to the tile containing its centre. Aggregation is one np.bincount per month, not one pass per tile."""

    def __init__(self, lat, lon, tile_deg: float):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        s = np.round(np.floor(lat / tile_deg + 1e-9) * tile_deg, 6)
        w = np.round(np.floor(lon / tile_deg + 1e-9) * tile_deg, 6)
        s_u, s_i = np.unique(s, return_inverse=True)
        w_u, w_i = np.unique(w, return_inverse=True)
        self.tile_deg, self.shape = tile_deg, (len(lat), len(lon))
        self.lat, self.lon = lat, lon
        self.cell_tile = (s_i[:, None] * len(w_u) + w_i[None, :]).ravel()
        corners = [(a + 0.0, b + 0.0) for a in s_u for b in w_u]  # + 0.0 turns -0.0 into 0.0 for the ids
        self.ids = [f"tile_{a:+07.2f}_{b:+08.2f}" for a, b in corners]
        self.boxes = {rid: (b, a, b + tile_deg, a + tile_deg) for rid, (a, b) in zip(self.ids, corners)}
        self.cell_weight = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], self.shape).ravel()

    def sums(self, field: np.ndarray):
        """Per-tile (sum of w*x over valid cells, sum of w over valid cells) for one 2-D field."""
        v = np.asarray(field, float).ravel()
        ok = np.isfinite(v)
        n = len(self.ids)
        sw = np.bincount(self.cell_tile[ok], weights=self.cell_weight[ok], minlength=n)
        swx = np.bincount(self.cell_tile[ok], weights=self.cell_weight[ok] * v[ok], minlength=n)
        return swx, sw

    def matches(self, lat, lon) -> bool:
        return np.array_equal(np.asarray(lat, float), self.lat) and np.array_equal(np.asarray(lon, float), self.lon)


class TileAccumulator:
    """Streams months in and builds area-weighted tile series with the same semantics as `regional_mean`:
    a month is kept only if its valid weight is >= min_coverage of the weight of cells ever valid in the tile."""

    def __init__(self, grid: TileGrid, variables):
        self.grid, self.variables = grid, list(variables)
        self.times, self.swx, self.sw = [], {v: [] for v in self.variables}, {v: [] for v in self.variables}
        self.ever = {v: np.zeros(grid.shape[0] * grid.shape[1], bool) for v in self.variables}

    def update(self, ds: xr.Dataset):
        for i, t in enumerate(ds["time"].values):
            self.times.append(t)
            for v in self.variables:
                field = ds[v].isel(time=i).values
                swx, sw = self.grid.sums(field)
                self.swx[v].append(swx)
                self.sw[v].append(sw)
                self.ever[v] |= np.isfinite(field).ravel()

    def finalize(self, min_coverage: float = 0.5) -> dict[str, pd.DataFrame]:
        g, idx = self.grid, pd.DatetimeIndex(self.times, name="time")
        out = {}
        for v in self.variables:
            total = np.bincount(g.cell_tile[self.ever[v]], weights=g.cell_weight[self.ever[v]], minlength=len(g.ids))
            swx, sw = np.array(self.swx[v]), np.array(self.sw[v])
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(sw > 0, swx / sw, np.nan)
                mean = np.where(sw / total >= min_coverage, mean, np.nan)
            keep = total > 0
            df = pd.DataFrame(mean[:, keep], index=idx, columns=np.array(g.ids)[keep]).sort_index()
            out[v] = df
        return out


def tile_regions(ds: xr.Dataset, tile_deg: float) -> dict[str, tuple[float, float, float, float]]:
    """Grid-aligned square tiles covering the domain; returns {region_id: (w, s, e, n)}."""
    return TileGrid(ds.lat.values, ds.lon.values, tile_deg).boxes


def tile_series(ds: xr.Dataset, var: str, tile_deg: float, min_coverage: float = 0.5) -> pd.DataFrame:
    """Vectorised equivalent of region_series(ds, var, tile_regions(ds, tile_deg))."""
    acc = TileAccumulator(TileGrid(ds.lat.values, ds.lon.values, tile_deg), [var])
    acc.update(ds)
    return acc.finalize(min_coverage)[var]


def block_mean(stack: np.ndarray, lat, factor: int):
    """Area-weighted (cos-lat) block mean of a (C, H, W) stack over factor x factor cells. A cell counts only if all
    C channels are finite there, so linear budgets such as dS = P - ET - R are preserved exactly after coarsening.
    Trailing rows/columns that do not fill a block are dropped."""
    if factor == 1:
        return stack
    c, h, w = stack.shape
    h2, w2 = h // factor, w // factor
    x = stack[:, :h2 * factor, :w2 * factor]
    joint = np.isfinite(x).all(0)
    wt = np.where(joint, np.cos(np.deg2rad(np.asarray(lat, float)[:h2 * factor]))[:, None], 0.0)
    num = (np.where(joint, x, 0.0) * wt).reshape(c, h2, factor, w2, factor).sum((2, 4))
    den = wt.reshape(h2, factor, w2, factor).sum((1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan).astype(np.float32)


def block_coords(coord, factor: int) -> np.ndarray:
    c = np.asarray(coord, float)
    n = len(c) // factor
    return c[:n * factor].reshape(n, factor).mean(1)


def region_series(ds: xr.Dataset, var: str, regions: dict, min_coverage: float = 0.5) -> pd.DataFrame:
    """Area-weighted regional series for arbitrary region boxes; columns = region ids. Loops over regions, so
    use `tile_series` / `TileAccumulator` for regular tilings."""
    weights = cell_weights(ds)
    cols = {}
    for rid, (w, s, e, n) in regions.items():
        sub = ds[var].where((ds.lon >= w) & (ds.lon < e) & (ds.lat >= s) & (ds.lat < n))
        if int(sub.notnull().any("time").sum()) == 0:
            continue
        cols[rid] = regional_mean(sub, weights, min_coverage)
    return pd.DataFrame(cols)
