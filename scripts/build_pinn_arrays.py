"""Stream the GLDAS files into block-averaged arrays for the physics-informed forecaster.

  python scripts/build_pinn_arrays.py --raw data/raw/gldas --coarsen 4 --out data/pinn/gldas_1deg.npz

Channels (pinn.CHANNELS order, then forcing inputs): land_water_store (mm, soil 0-200 cm + SWE + canopy, ice cells
masked), Rainf_f_tavg, Evap_tavg, runoff (mm/month totals), Tair_f_inst (K), SWdown_f_tavg, LWdown_f_tavg (W m-2).
Cells are restricted to Noah land (no inland water) and averaged over blocks with cos(lat) weights, counting a cell
only where every channel is finite, so dS = P - ET - R budgets survive coarsening. Missing stays NaN.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paradox import gldas  # noqa: E402

CHANNELS = ["land_water_store", "Rainf_f_tavg", "Evap_tavg", "runoff", "Tair_f_inst", "SWdown_f_tavg", "LWdown_f_tavg"]
FORCING = ["SWdown_f_tavg", "LWdown_f_tavg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/gldas")
    ap.add_argument("--coarsen", type=int, default=4)
    ap.add_argument("--out", default="data/pinn/gldas_1deg.npz")
    args = ap.parse_args()
    files = gldas.list_files(args.raw)
    land = ice = lat = lon = None
    frames, times = [], []
    for k, f in enumerate(files):
        raw = gldas.add_derived(gldas.to_monthly_water_units(gldas.load_file(f, variables=gldas.ALL_VARS + FORCING)))
        if land is None:
            land, ice = gldas.land_ice_masks(raw)
            lat, lon = raw.lat.values, raw.lon.values
        m = gldas.apply_masks(raw, land, ice, CHANNELS)
        for i, t in enumerate(m["time"].values):
            stack = np.stack([m[v].isel(time=i).values for v in CHANNELS]).astype(np.float32)
            frames.append(gldas.block_mean(stack, lat, args.coarsen))
            times.append(pd.Timestamp(t))
        if (k + 1) % 24 == 0:
            print(f"pinn arrays: {k + 1}/{len(files)} files", flush=True)
    arr = np.stack(frames)  # (T, C, H, W), NaN where invalid
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, arr=arr, channels=np.array(CHANNELS), times=np.array([t.strftime("%Y-%m") for t in times]),
             lat=gldas.block_coords(lat, args.coarsen), lon=gldas.block_coords(lon, args.coarsen),
             coarsen=args.coarsen, ice_cells=int(ice.sum()))
    print(f"wrote {out}: {arr.shape}, finite cells (all channels, all months): "
          f"{int(np.isfinite(arr).all(axis=(0, 1)).sum())}")


if __name__ == "__main__":
    main()
