"""Convert FLDAS_NOAH01_C_GL_M monthly files (0.1 deg) into GLDAS_NOAH025_M-lookalike NetCDFs at 0.5 deg.

The output can be read UNCHANGED by paradox.gldas.list_files / load_file / to_monthly_water_units / add_derived, so the
existing pipeline can run on FLDAS as an independent-forcing replicate (FLDAS: CHIRPS + MERRA-2 forcing, Noah 3.6).

Per input month:
  * X -> lon, Y -> lat (ascending lat, lon in -180..180). FLDAS covers lat -59.95..89.95, lon -179.95..179.95 at
    0.1 deg = 1500 x 3600 cells, i.e. exactly 300 x 720 whole 5x5 blocks -> 0.5 deg centres -59.75..89.75,
    -179.75..179.75 (any trailing partial block would be trimmed).
  * soil layers m3 m-3 x layer thickness (mm; from the SOIL_LAYER_THICKNESSES global attr, given in cm, else
    100/300/600/1000) -> SoilMoi0_10cm_inst, SoilMoi10_40cm_inst, SoilMoi40_100cm_inst, SoilMoi100_200cm_inst
    (kg m-2). NOTE: FLDAS soil moisture is a time-step monthly MEAN (_tavg); GLDAS monthly files store the monthly
    average of 3-hourly instantaneous values (_inst). The two are comparable monthly means; names kept for
    compatibility.
  * SWE_inst passed through (kg m-2).
  * CanopInt_inst = 0 (FLDAS does not distribute canopy interception storage; GLDAS canopy water is < ~1 mm, so
    land_water_store is lower by that amount). Set to 0 where the water variables are valid, NaN elsewhere.
  * Rainf_f_tavg (FLDAS: TOTAL precipitation, verified Rainf_f >= Snowf_tavg everywhere), Evap_tavg passed
    through (kg m-2 s-1).
  * Qs_tavg, Qsb_tavg (kg m-2 s-1) x 10800 s -> Qs_acc, Qsb_acc (kg m-2 per 3 h, the GLDAS accumulation
    convention that to_monthly_water_units multiplies by 8 x days).
  * Tair_f_tavg -> Tair_f_inst (K; it is a monthly mean, not an instantaneous value).
  * SWdown_f_tavg, LWdown_f_tavg kept (W m-2) when present.

Masking and coarsening (both at the native 0.1 deg BEFORE block averaging):
  * glacier / land-ice cells (0-200 cm soil water > paradox.gldas.ICE_SOIL_MM = 1500 mm; Noah stores ice at full
    saturation = 2000 mm) are set to NaN for all WATER-BUDGET variables (soil layers, SWE, canopy, precipitation,
    ET, runoff). SWE is masked too, because ice-sheet SWE (> 1e5 mm) would otherwise dominate any half-ice block
    while its soil came only from the non-ice cells, giving a meaningless land_water_store. Consequence: unlike
    GLDAS, the shim has no water variables on ice sheets at all; they are simply absent (NaN).
  * water-budget variables share ONE joint validity mask (a cell counts only if every one of them is finite), as
    in paradox.gldas.block_mean, so linear budgets such as net_flux = P - ET - R stay consistent after coarsening.
    Meteorological variables (Tair, SWdown, LWdown) use their own joint mask.
  * 5x5 block mean with cos(lat) weights; a 0.5 deg cell is kept only if >= 13 of its 25 cells are valid
    (else NaN).

Re-runnable: existing outputs are skipped (use --overwrite to rebuild); outputs are written to a temp name and
renamed, so an interrupted run never leaves a truncated file. Only completed FLDAS_NOAH01_C_GL_M.A{YYYYMM}.001.nc
files are considered (in-progress downloads use other names); unreadable inputs are reported and skipped.

Usage:
  .venv/bin/python scripts/shim_fldas.py                       # all inputs, 4 processes
  .venv/bin/python scripts/shim_fldas.py --workers 2 --check   # then sanity-check outputs with paradox.gldas
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paradox import gldas  # noqa: E402

IN_DIR = ROOT / "data/raw/fldas"
OUT_DIR = ROOT / "data/raw/fldas_shim"
IN_RE = re.compile(r"^FLDAS_NOAH01_C_GL_M\.A(\d{6})\.001\.nc$")
OUT_NAME = "GLDAS_NOAH025_M.A{ym}.fldas_shim05.nc4"  # GLDAS-style name: list_files globs *.nc4, A{YYYYMM} preserved

FACTOR = 5          # 0.1 deg -> 0.5 deg
MIN_VALID = 13      # of FACTOR**2 = 25 native cells
SEC_PER_3H = 10800.0
DEFAULT_THICK_MM = [100.0, 300.0, 600.0, 1000.0]

SOIL_IN = ["SoilMoi00_10cm_tavg", "SoilMoi10_40cm_tavg", "SoilMoi40_100cm_tavg", "SoilMoi100_200cm_tavg"]
SOIL_OUT = ["SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst", "SoilMoi40_100cm_inst", "SoilMoi100_200cm_inst"]
# water-budget group (joint mask, ice-masked): out name -> (in name, scale, units, long_name)
WATER = {
    "SWE_inst": ("SWE_inst", 1.0, "kg m-2", "Snow depth water equivalent"),
    "Rainf_f_tavg": ("Rainf_f_tavg", 1.0, "kg m-2 s-1", "Total precipitation rate"),
    "Evap_tavg": ("Evap_tavg", 1.0, "kg m-2 s-1", "Evapotranspiration"),
    "Qs_acc": ("Qs_tavg", SEC_PER_3H, "kg m-2", "Storm surface runoff (per 3 h, from FLDAS mean rate x 10800 s)"),
    "Qsb_acc": ("Qsb_tavg", SEC_PER_3H, "kg m-2", "Baseflow-groundwater runoff (per 3 h, from mean rate x 10800 s)"),
}
# meteorological group (own joint mask, not ice-masked)
MET = {
    "Tair_f_inst": ("Tair_f_tavg", 1.0, "K", "Air temperature (FLDAS monthly MEAN, stored under the GLDAS name)"),
    "SWdown_f_tavg": ("SWdown_f_tavg", 1.0, "W m-2", "Downward shortwave radiation flux"),
    "LWdown_f_tavg": ("LWdown_f_tavg", 1.0, "W m-2", "Downward longwave radiation flux"),
}


def soil_thickness_mm(attrs) -> list[float]:
    t = attrs.get("SOIL_LAYER_THICKNESSES")
    if t is None or len(np.atleast_1d(t)) != 4:
        return DEFAULT_THICK_MM
    t = np.asarray(t, float)
    # LIS writes thicknesses in cm (10, 30, 60, 100); accept metres or mm too by checking the 0-200 cm total
    total = t.sum()
    scale = 10.0 if abs(total - 200) < 1 else 1000.0 if abs(total - 2) < 0.01 else 1.0 if abs(total - 2000) < 10 else None
    return DEFAULT_THICK_MM if scale is None else list(np.round(t * scale, 3))


def block_weighted(fields: list[np.ndarray], valid: np.ndarray, wlat: np.ndarray) -> list[np.ndarray]:
    """cos-lat weighted FACTOR x FACTOR block mean of each (H, W) field over the shared `valid` mask; blocks with
    fewer than MIN_VALID valid cells -> NaN. H, W must be multiples of FACTOR."""
    h, w = valid.shape
    hb, wb = h // FACTOR, w // FACTOR
    wt = np.where(valid, wlat[:, None], 0.0)
    den = wt.reshape(hb, FACTOR, wb, FACTOR).sum((1, 3))
    cnt = valid.reshape(hb, FACTOR, wb, FACTOR).sum((1, 3))
    keep = (cnt >= MIN_VALID) & (den > 0)
    out = []
    for f in fields:
        num = (np.where(valid, f.astype(np.float64), 0.0) * wt).reshape(hb, FACTOR, wb, FACTOR).sum((1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(keep, num / den, np.nan).astype(np.float32))
    return out


def convert(src: Path, dst: Path) -> str:
    with xr.open_dataset(src) as ds:
        ds = ds.rename({"X": "lon", "Y": "lat"}) if "X" in ds.dims else ds
        ds = ds.assign_coords(lon=((ds["lon"] + 180.0) % 360.0) - 180.0).sortby(["lat", "lon"])
        thick = soil_thickness_mm(ds.attrs)
        t0 = pd.Timestamp(ds["time"].values[0])
        tstamp = pd.Timestamp(year=t0.year, month=t0.month, day=1)
        need = SOIL_IN + [v[0] for v in WATER.values()] + [v[0] for v in MET.values()]
        missing = [v for v in SOIL_IN + [v[0] for v in WATER.values()] + ["Tair_f_tavg"] if v not in ds]
        if missing:
            raise KeyError(f"{src.name}: missing {missing}")
        lat_full, lon_full = ds["lat"].values.astype(float), ds["lon"].values.astype(float)
        hb, wb = len(lat_full) // FACTOR, len(lon_full) // FACTOR
        h, w = hb * FACTOR, wb * FACTOR  # trim to whole blocks
        arr = {v: ds[v].isel(time=0).values[:h, :w].astype(np.float32) for v in need if v in ds}
    lat, lon = lat_full[:h], lon_full[:w]
    wlat = np.cos(np.deg2rad(lat))

    soil = [arr[v] * t for v, t in zip(SOIL_IN, thick)]
    soil_total = soil[0] + soil[1] + soil[2] + soil[3]
    ice = np.isfinite(soil_total) & (soil_total > gldas.ICE_SOIL_MM)
    water_in = soil + [arr[spec[0]] * spec[1] for spec in WATER.values()]
    water_valid = np.logical_and.reduce([np.isfinite(f) for f in water_in]) & ~ice
    water_out = block_weighted(water_in, water_valid, wlat)

    met_names = [k for k, spec in MET.items() if spec[0] in arr]
    met_in = [arr[MET[k][0]] * MET[k][1] for k in met_names]
    met_valid = np.logical_and.reduce([np.isfinite(f) for f in met_in])
    met_out = block_weighted(met_in, met_valid, wlat)

    lat_c = np.round(lat.reshape(hb, FACTOR).mean(1), 4)
    lon_c = np.round(lon.reshape(wb, FACTOR).mean(1), 4)
    coords = {"time": [tstamp], "lat": lat_c, "lon": lon_c}
    dims = ("time", "lat", "lon")
    out = xr.Dataset(coords=coords)
    for name, fld, t in zip(SOIL_OUT, water_out[:4], thick):
        out[name] = (dims, fld[None])
        out[name].attrs = {"units": "kg m-2", "long_name": f"Soil moisture {name[7:-5]} (FLDAS monthly mean "
                                                         f"m3 m-3 x {t:g} mm layer thickness)"}
    for (name, spec), fld in zip(WATER.items(), water_out[4:]):
        out[name] = (dims, fld[None])
        out[name].attrs = {"units": spec[2], "long_name": spec[3]}
    canop = np.where(np.isfinite(water_out[0]), 0.0, np.nan).astype(np.float32)
    out["CanopInt_inst"] = (dims, canop[None])
    out["CanopInt_inst"].attrs = {"units": "kg m-2", "long_name": "Plant canopy surface water",
                                  "note": "set to 0: canopy interception storage is not distributed in FLDAS"}
    for name, fld in zip(met_names, met_out):
        out[name] = (dims, fld[None])
        out[name].attrs = {"units": MET[name][2], "long_name": MET[name][3]}
    out["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    out["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    out.attrs = {
        "title": "FLDAS Noah 0.1 deg monthly, reshaped to look like GLDAS_NOAH025_M (paradox shim)",
        "source_file": src.name,
        "source_dataset": "FLDAS_NOAH01_C_GL_M v001 (Noah 3.6, CHIRPS + MERRA-2 forcing)",
        "grid": f"0.5 deg, cos-lat weighted {FACTOR}x{FACTOR} block mean of 0.1 deg, >= {MIN_VALID}/25 valid cells",
        "ice_mask": f"0.1 deg cells with 0-200 cm soil water > {gldas.ICE_SOIL_MM:g} mm set NaN for soil, SWE, "
                    "canopy, precipitation, ET and runoff before coarsening",
        "joint_mask": "water-budget variables share one validity mask; Tair/SWdown/LWdown share another",
        "soil_layer_thickness_mm": np.asarray(thick, float),
        "conventions_note": "Qs_acc/Qsb_acc are kg m-2 per 3 h (rate x 10800 s); soil, SWE and Tair are FLDAS "
                            "monthly means stored under GLDAS _inst names; CanopInt_inst = 0",
    }
    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(-9999.0)}
           for v in out.data_vars}
    enc["time"] = {"units": "days since 2000-01-01", "calendar": "standard", "dtype": "float64"}
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    out.to_netcdf(tmp, encoding=enc, format="NETCDF4")
    os.replace(tmp, dst)
    return f"{src.name} -> {dst.name} (ice cells masked: {int(ice.sum())})"


def _job(src: str, dst: str):
    try:
        return True, convert(Path(src), Path(dst))
    except Exception as e:  # report and continue; a later re-run retries it
        return False, f"{Path(src).name}: {type(e).__name__}: {e}"


def check(out_dir: Path) -> None:
    """Open the shim outputs exactly as the pipeline does and print plausibility diagnostics."""
    files = gldas.list_files(out_dir)
    print(f"[check] {len(files)} shim files in {out_dir}")
    grid = None
    for f in files:
        ds = gldas.add_derived(gldas.to_monthly_water_units(gldas.load_file(f, variables=gldas.ALL_VARS +
                                                                             ["SWdown_f_tavg", "LWdown_f_tavg"])))
        if grid is None:
            grid = gldas.TileGrid(ds.lat.values, ds.lon.values, 1.0)
            print(f"[check] grid lat {ds.lat.values[0]}..{ds.lat.values[-1]} ({ds.sizes['lat']}), lon "
                  f"{ds.lon.values[0]}..{ds.lon.values[-1]} ({ds.sizes['lon']}); TileGrid(1.0): "
                  f"{len(grid.ids)} tiles, {len(grid.boxes)} complete, {grid.n_cells} cells/tile")
            missing = [v for v in gldas.ALL_VARS if v not in ds]
            print(f"[check] GLDAS variables missing: {missing or 'none'}")
        w = np.cos(np.deg2rad(ds.lat.values))[:, None]

        def stat(v):
            a = ds[v].isel(time=0).values
            ok = np.isfinite(a)
            wm = float((a[ok] * np.broadcast_to(w, a.shape)[ok]).sum() / np.broadcast_to(w, a.shape)[ok].sum())
            return f"{v}: min {np.nanmin(a):.1f} p50 {np.nanmedian(a):.1f} max {np.nanmax(a):.1f} wmean {wm:.1f}"
        sw = ds["soil_water_0_200cm"].values
        print(f"[check] {pd.Timestamp(ds.time.values[0]):%Y-%m} n_land={int(np.isfinite(sw).sum())} | "
              f"{stat('soil_water_0_200cm')} | {stat('Rainf_f_tavg')} | {stat('Evap_tavg')} | {stat('runoff')} | "
              f"{stat('Tair_f_inst')}")
        assert np.nanmax(sw) <= gldas.ICE_SOIL_MM and np.nanmin(sw) >= 0, "soil water out of range"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-dir", type=Path, default=IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="convert at most N pending files (0 = all)")
    ap.add_argument("--check", action="store_true", help="after converting, open outputs with paradox.gldas")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.tmp*"):
        stale.unlink()

    srcs = sorted(p for p in args.in_dir.iterdir() if IN_RE.match(p.name))
    todo = []
    for p in srcs:
        dst = args.out_dir / OUT_NAME.format(ym=IN_RE.match(p.name).group(1))
        if args.overwrite or not dst.exists():
            todo.append((str(p), str(dst)))
    n_pending = len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(srcs)} FLDAS inputs, {len(srcs) - n_pending} already shimmed, {n_pending} pending, {len(todo)} to convert "
          f"with {min(args.workers, max(len(todo), 1))} workers", flush=True)
    t0, failed = time.time(), []
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, 6))) as ex:
        futs = [ex.submit(_job, s, d) for s, d in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            ok, msg = fut.result()
            if not ok:
                failed.append(msg)
            print(f"[{i}/{len(todo)} {time.time() - t0:.0f}s] {'ok ' if ok else 'ERR'} {msg}", flush=True)
    print(f"done: {len(todo) - len(failed)} written, {len(failed)} failed", flush=True)
    for m in failed:
        print("  failed:", m)
    if args.check:
        check(args.out_dir)


if __name__ == "__main__":
    main()
