"""Physics-informed emulator of the GLDAS-2.1 land water store (Manifest §10, Gate G).

  python scripts/run_pinn_experiment.py --arrays data/pinn/gldas_1deg_2012_2025.npz --out outputs_pinn --device cuda

What it is: a one-month, forcing-driven emulator of GLDAS-2.1 Noah. Given the last 6 months of the state and fluxes
and next month's weather (P, T; plus SWdown, LWdown in the radiation-aware arm), a small CNN predicts next month's
change in the land water store (soil 0-200 cm + SWE + canopy; NOT total water storage) and the fluxes ET, R, with
loss L = L_pred + lambda * L_water_balance. Skill is skill against GLDAS model output, not against observations.

Design
  window    2012-01..2025-05 at 1 degree; cells whose store ever exceeds MAX_STORE_MM (perennial snow) excluded
  splits    by TARGET month: train 2012-07..2019-12, val 2020-01..2021-12, test 2022-01..2024-06 (one radiation-
            forcing regime); out-of-regime (OOR) 2024-07..2025-05 = degraded WWMCA radiation, never trained on
  space     eastern 25% of columns (90E-180E) held out of the loss; 5-column guard band used by neither side
  arms      base (P, T known) | radiation-aware (+ SWdown, LWdown known) x lambda {0, 0.1, 1, 10} x seeds
  metrics   cos(lat)-weighted, on the storage CHANGE dS (level skill is trivial): RMSE, MSESS = 1 - MSE/MSE_ref vs
            persistence (dS = 0), climatological increment (train mean dS per calendar month and cell) and a
            linear AR with the same inputs; ET and R MAE; budget residual |dS - (F_t + F_{t+1})/2| with the model's
            own fluxes and with GLDAS fluxes; GLDAS's own residual (the truth floor) for reference.
"""
import argparse
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paradox.contract import write_json  # noqa: E402
from paradox.pinn import ET, P, R, S, PINNConfig, Prepared, baselines, physical, train  # noqa: E402

MAX_STORE_MM = 3000.0
WINDOW = ("2012-01", "2025-05")
SPLITS = {"train": ("2012-01", "2019-12"), "val": ("2020-01", "2021-12"), "test": ("2022-01", "2024-06"),
          "oor": ("2024-07", "2025-05")}
SAME_SEASON = ("2023-07", "2024-05")  # in-regime months matching the OOR calendar months, for a fair comparison


def load(path):
    z = np.load(path)
    times = [str(t) for t in z["times"]]
    i0, i1 = times.index(WINDOW[0]), times.index(WINDOW[1])
    arr = z["arr"][i0:i1 + 1]
    times = times[i0:i1 + 1]
    finite = np.isfinite(arr).all(axis=(0, 1))
    store_ok = np.nan_to_num(arr[:, 0], nan=np.inf).max(axis=0) <= MAX_STORE_MM
    mask = finite & store_ok
    info = {"window": list(WINDOW), "cells_finite": int(finite.sum()), "cells_used": int(mask.sum()),
            "cells_dropped_perennial_snow": int((finite & ~store_ok).sum()), "max_store_mm": MAX_STORE_MM,
            "channels": [str(c) for c in z["channels"]], "coarsen": int(z["coarsen"])}
    months = np.array([int(t[5:7]) for t in times])
    return np.nan_to_num(arr).astype(np.float32), mask, months, times, info, np.asarray(z["lat"]), np.asarray(z["lon"])


def target_idx(times, window, a, b):
    return np.array([t for t in range(window - 1, len(times) - 1) if a <= times[t + 1] <= b])


def predict_physical(model, prep, ts, chunk=8):
    """Physical next-month dS, P, ET, R (mm) from the network, in chunks on the model's device."""
    dev = next(model.parameters()).device
    outs = {k: [] for k in ("ds", "p", "et", "r")}
    with torch.no_grad():
        for i in range(0, len(ts), chunk):
            x, _, _, p_next = (t.to(dev) for t in prep.batch(ts[i:i + chunk]))
            ds, p, et, r = physical(model(x), prep, p_next if prep.cfg.forcing_known else None)
            for k, v in zip(outs, (ds, p, et, r)):
                outs[k].append(v.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


class Scorer:
    """Area-weighted metrics on a fixed set of cells and target months."""

    def __init__(self, arr, lat, cells, ts, clim_ds):
        self.a, self.ts, self.cells = arr, ts, cells
        w = np.cos(np.deg2rad(lat))[:, None] * cells
        self.w = w / w.sum()
        self.ds_true = arr[ts + 1, S] - arr[ts, S]
        self.f_t = arr[ts, P] - arr[ts, ET] - arr[ts, R]
        self.f_next = arr[ts + 1, P] - arr[ts + 1, ET] - arr[ts + 1, R]
        self.refs = {"persistence": np.zeros_like(self.ds_true), "clim_increment": clim_ds}

    def wmean(self, x):  # x: (n_t, H, W) -> scalar weighted mean over cells and months
        return float((x * self.w[None]).sum(axis=(1, 2)).mean())

    def mse(self, ds_pred):
        return self.wmean((ds_pred - self.ds_true) ** 2)

    def score(self, ds_pred, et=None, r=None, p_next=None, ar_ds=None):
        m = {"ds_rmse_mm": float(np.sqrt(self.mse(ds_pred)))}
        refs = dict(self.refs)
        if ar_ds is not None:
            refs["linear_ar"] = ar_ds
        for k, ref in refs.items():
            m[f"msess_vs_{k}"] = 1.0 - self.mse(ds_pred) / self.mse(ref)
        if et is not None:
            m["et_mae_mm"] = self.wmean(np.abs(et - self.a[self.ts + 1, ET]))
            m["r_mae_mm"] = self.wmean(np.abs(r - self.a[self.ts + 1, R]))
            own = p_next - et - r
            m["budget_residual_own_fluxes_mm"] = self.wmean(np.abs(ds_pred - 0.5 * (self.f_t + own)))
        m["budget_residual_gldas_fluxes_mm"] = self.wmean(np.abs(ds_pred - 0.5 * (self.f_t + self.f_next)))
        return m

    def truth_floor(self):
        return self.wmean(np.abs(self.ds_true - 0.5 * (self.f_t + self.f_next)))

    def cell_mse(self, ds_pred):
        return ((ds_pred - self.ds_true) ** 2).mean(0)


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data/pinn/gldas_1deg_2012_2025.npz")
    ap.add_argument("--out", default="outputs_pinn")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.1, 1.0, 10.0])
    ap.add_argument("--arms", nargs="+", default=["base", "radiation"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--guard", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)

    arr_all, mask, months, times, info, lat, lon = load(args.arrays)
    print(f"data {arr_all.shape} cells used {info['cells_used']} (dropped {info['cells_dropped_perennial_snow']} "
          f"perennial-snow cells) device {args.device}", flush=True)
    design = {**info, "splits_by_target_month": SPLITS, "same_season_in_regime": SAME_SEASON, "arms": args.arms,
              "lambdas": args.lams, "seeds": args.seeds, "epochs": args.epochs, "device": args.device,
              "guard_columns": args.guard, "git_sha": git_sha(), "training": "full grid (no random crops)",
              "selection": "best validation dS loss (same rule for every lambda)",
              "target": "next-month change in GLDAS land water store (soil 0-200 cm + SWE + canopy; NOT TWS)",
              "note": "emulation skill against GLDAS-2.1 model output, not real-world forecast skill"}
    results = {"design": design, "arms": {}}
    maps = {}
    for arm in args.arms:
        n_forcing = 2 if arm == "radiation" else 0
        arr = arr_all[:, :5 + n_forcing]
        base = PINNConfig(epochs=args.epochs, hidden=args.hidden, alignment="centered", device=args.device,
                          n_forcing=n_forcing, val_every=2, batch=16, patch=10 ** 6, guard=args.guard)
        sp = {k: target_idx(times, base.window, *v) for k, v in SPLITS.items()}
        prep0 = Prepared(arr, mask, months, base, splits=sp)
        # climatological increment: train-period mean dS for each calendar month and cell
        tr = sp["train"]
        ds_tr = arr[tr + 1, S] - arr[tr, S]
        clim = np.stack([ds_tr[months[tr + 1] == mth].mean(0) for mth in range(1, 13)])
        seasons = {"test": sp["test"], "oor": sp["oor"],
                   "same_season_in_regime": target_idx(times, base.window, *SAME_SEASON)}
        scorers = {}
        for split, ts in seasons.items():
            for region, cells in (("in_region", prep0.train_cells), ("spatial_holdout", prep0.holdout_cells)):
                scorers[(split, region)] = Scorer(arr, lat, cells, ts, clim[months[ts + 1] - 1])
        ar_ds = {split: baselines(prep0, ts)["linear_ar"] - arr[ts, S] for split, ts in seasons.items()}
        arm_res = {"n_inputs": prep0.n_in, "split_sizes": {k: int(len(v)) for k, v in {**sp, **seasons}.items()},
                   "cells": {"train": int(prep0.train_cells.sum()), "holdout": int(prep0.holdout_cells.sum())},
                   "references": {}, "truth_floor_mm": {}, "runs": {}}
        for (split, region), sc in scorers.items():
            refs = {"linear_ar": sc.score(ar_ds[split])}
            refs["linear_ar"].pop("msess_vs_linear_ar", None)
            refs["clim_increment"] = sc.score(sc.refs["clim_increment"], ar_ds=ar_ds[split])
            arm_res["references"].setdefault(split, {})[region] = refs
            arm_res["truth_floor_mm"].setdefault(split, {})[region] = sc.truth_floor()
        for lam in args.lams:
            for seed in args.seeds:
                cfg = PINNConfig(**{**asdict(base), "lam": lam, "seed": seed})
                prep = Prepared(arr, mask, months, cfg, splits=sp)
                t0 = time.time()
                model, hist = train(prep, cfg)
                key = f"lam={lam}|seed={seed}"
                run = {"train_seconds": round(time.time() - t0, 1), "best_val_ds": min(h["val_ds"] for h in hist),
                       "history": hist}
                for split, ts in seasons.items():
                    phys = predict_physical(model, prep, ts)
                    p_next = arr[ts + 1, P]  # known forcing
                    for region in ("in_region", "spatial_holdout"):
                        run.setdefault(split, {})[region] = scorers[(split, region)].score(
                            phys["ds"], phys["et"], phys["r"], p_next, ar_ds[split])
                    if split == "test":
                        sc = scorers[("test", "in_region")]
                        maps.setdefault(f"{arm}|lam={lam}|cell_mse_model", []).append(sc.cell_mse(phys["ds"]))
                torch.save(model.state_dict(), out / "models" / f"pinn_{arm}_lam{lam}_seed{seed}.pt")
                arm_res["runs"][key] = run
                t = run["test"]
                print(f"{arm} {key}: {run['train_seconds']}s | test dS RMSE {t['in_region']['ds_rmse_mm']:.2f} "
                      f"MSESS vs AR {t['in_region']['msess_vs_linear_ar']:+.3f} (holdout "
                      f"{t['spatial_holdout']['msess_vs_linear_ar']:+.3f}) budget(own) "
                      f"{t['in_region']['budget_residual_own_fluxes_mm']:.2f} | OOR MSESS vs AR "
                      f"{run['oor']['in_region']['msess_vs_linear_ar']:+.3f}", flush=True)
                results["arms"][arm] = arm_res
                write_json(out / "results.json", results)
        # seed summaries and paired lambda effects (same seed)
        summ = {}
        mets = ("ds_rmse_mm", "msess_vs_linear_ar", "msess_vs_clim_increment", "msess_vs_persistence", "et_mae_mm",
                "r_mae_mm", "budget_residual_own_fluxes_mm", "budget_residual_gldas_fluxes_mm")
        for lam in args.lams:
            for split in seasons:
                for region in ("in_region", "spatial_holdout"):
                    for met in mets:
                        v = np.array([arm_res["runs"][f"lam={lam}|seed={s}"][split][region][met] for s in args.seeds])
                        d = summ.setdefault(f"lam={lam}", {}).setdefault(split, {}).setdefault(region, {})
                        d[met] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
                        if lam != args.lams[0]:
                            v0 = np.array([arm_res["runs"][f"lam={args.lams[0]}|seed={s}"][split][region][met]
                                           for s in args.seeds])
                            diff = v - v0
                            d[met]["paired_diff_vs_lam0"] = {"mean": float(diff.mean()),
                                                             "sd": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0}
        arm_res["summary"] = summ
        results["arms"][arm] = arm_res
        write_json(out / "results.json", results)
        # per-cell skill map vs linear AR (seed mean), test period, in-region scorer cells
        sc = scorers[("test", "in_region")]
        ar_cell = sc.cell_mse(ar_ds["test"])
        for lam in args.lams:
            mm = np.mean(maps.pop(f"{arm}|lam={lam}|cell_mse_model"), axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                maps[f"{arm}|lam={lam}|msess_vs_ar"] = np.where(mask, 1 - mm / ar_cell, np.nan).astype(np.float32)
    np.savez_compressed(out / "skill_maps.npz", lat=lat, lon=lon, mask=mask, **maps)
    print("done ->", out / "results.json", flush=True)


if __name__ == "__main__":
    main()
