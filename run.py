"""Paradox Earth runner.

  python run.py --synthetic                                     # smoke run on generated GLDAS-like data
  python run.py --raw data/raw/gldas --bbox -125 24 -66 50 --tile 1.0   # CONUS
  python run.py --raw data/raw/gldas --tile 1.0 --pinn-coarsen 4        # global; PINN on a 1° grid
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from paradox.pinn import PINNConfig  # noqa: E402
from paradox.pipeline import run_all  # noqa: E402
from paradox.synthetic import make_synthetic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/gldas")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="omit for global")
    ap.add_argument("--tile", type=float, default=1.0, help="region tile size in degrees")
    ap.add_argument("--ref", nargs=2, metavar=("START", "END"), help="climatology reference period")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processes for per-region statistics")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lam", type=float, default=0.1, help="physics weight for the lambda>0 arm")
    ap.add_argument("--alignment", default="centered", choices=["centered", "forward"])
    ap.add_argument("--no-pinn", action="store_true", help="skip the forecasting extension")
    ap.add_argument("--pinn-coarsen", type=int, default=1, help="block-average the PINN grid (4 => 1 degree)")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        args.raw = "data/raw/synthetic"
        make_synthetic(args.raw)
        args.alignment = "forward"  # synthetic storages are end-of-month states

    cfg = PINNConfig(epochs=args.epochs, alignment=args.alignment)
    out = run_all(args.raw, args.out, bbox=args.bbox, tile_deg=args.tile,
                  ref_period=tuple(args.ref) if args.ref else None, workers=args.workers,
                  pinn_cfg=cfg, lams=(0.0, args.lam), pinn=not args.no_pinn, pinn_coarsen=args.pinn_coarsen)
    print("QC passed:", out["qc"]["passed"])
    print(out["summary"].groupby(["variable", "egarch_status"]).size().to_string())
    if out["pinn"]:
        for k, r in out["pinn"]["runs"].items():
            m = r["test"]["in_region"]
            print(k, {n: round(v["rmse_mm"], 2) for n, v in m.items()})


if __name__ == "__main__":
    main()
