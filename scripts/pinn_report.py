"""Summarise the forecaster experiment for the explorer, README and slides.

  python scripts/pinn_report.py --results outputs_pinn --site site --demo demo

Writes site/data/pinn.json (table rows: seed mean +- sd per arm and lambda, reference baselines, truth floor, paired
lambda effects), demo/pinn_tradeoff.png (budget residual vs dS skill across lambda, one point per seed) and
demo/pinn_skill_map.png (per-cell dS skill vs the linear AR, test period, best configuration).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paradox.contract import write_json  # noqa: E402

BG, FG, MUTED, GRID = "#0a1322", "#e6edf6", "#8ea0b8", "#1d2b44"
ARM_COLOR = {"base": "#4fa3f7", "radiation": "#f2873a"}
ARM_LABEL = {"base": "rain + temperature known", "radiation": "+ radiation known"}
METS = ("ds_rmse_mm", "msess_vs_linear_ar", "msess_vs_clim_increment", "msess_vs_persistence", "et_mae_mm",
        "r_mae_mm", "budget_residual_own_fluxes_mm", "budget_residual_gldas_fluxes_mm")


def seed_stats(runs, lam, split, region, met):
    v = np.array([r[split][region][met] for k, r in runs.items() if k.startswith(f"lam={lam}|")])
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs_pinn")
    ap.add_argument("--site", default="site")
    ap.add_argument("--demo", default="demo")
    ap.add_argument("--display-arm", default="base")
    ap.add_argument("--display-lambda", type=float, default=1.0,
                    help="configuration shown on the skill map: the budget-closing setting the text recommends")
    args = ap.parse_args()
    res = json.loads((Path(args.results) / "results.json").read_text())
    d = res["design"]
    lams = d["lambdas"]
    rows, refs, floor, paired = [], {}, {}, {}
    for arm, a in res["arms"].items():
        for split in ("test", "oor", "same_season_in_regime"):
            for region in ("in_region", "spatial_holdout"):
                floor.setdefault(split, {})[region] = a["truth_floor_mm"][split][region]
                for name, m in a["references"][split][region].items():
                    refs.setdefault(arm, {}).setdefault(split, {}).setdefault(region, {})[name] = m
        for lam in lams:
            row = {"arm": arm, "arm_label": ARM_LABEL.get(arm, arm), "lambda": lam}
            for split in ("test", "oor", "same_season_in_regime"):
                for region in ("in_region", "spatial_holdout"):
                    row.setdefault(split, {})[region] = {m: seed_stats(a["runs"], lam, split, region, m) for m in METS}
            rows.append(row)
            if lam != lams[0]:
                s = a["summary"][f"lam={lam}"]["test"]["in_region"]
                paired.setdefault(arm, {})[str(lam)] = {m: s[m].get("paired_diff_vs_lam0") for m in METS}
    # displayed configuration is fixed up front (not chosen on test skill): the setting the text recommends
    best = next(r for r in rows if r["arm"] == args.display_arm and float(r["lambda"]) == args.display_lambda)
    out = {"design": d, "rows": rows, "references": refs, "truth_floor_mm": floor, "paired_lambda_effects": paired,
           "best_for_display": {"arm": best["arm"], "lambda": best["lambda"]},
           "figures": {"tradeoff": f"../{args.demo}/pinn_tradeoff.png", "skill_map": f"../{args.demo}/pinn_skill_map.png"}}
    write_json(Path(args.site) / "data" / "pinn.json", out)

    # trade-off: budget residual (own fluxes) vs dS skill vs linear AR, per seed, test in-region
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160, facecolor=BG)
    ax.set_facecolor(BG)
    for arm, a in res["arms"].items():
        for lam in lams:
            pts = [(r["test"]["in_region"]["budget_residual_own_fluxes_mm"], r["test"]["in_region"]["msess_vs_linear_ar"])
                   for k, r in a["runs"].items() if k.startswith(f"lam={lam}|")]
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=28, color=ARM_COLOR.get(arm, FG), alpha=0.55, edgecolor="none")
            ax.scatter(np.mean(xs), np.mean(ys), s=110, color=ARM_COLOR.get(arm, FG), edgecolor=FG, lw=1.2, zorder=3)
            ax.annotate(f"λ={lam:g}", (np.mean(xs), np.mean(ys)), xytext=(8, 6), textcoords="offset points",
                        color=FG, fontsize=10)
    fl = floor["test"]["in_region"]
    ax.axvline(fl, color=MUTED, ls="--", lw=1)
    ax.text(fl, ax.get_ylim()[1], " GLDAS's own budget\n residual (truth floor)", color=MUTED, fontsize=9, va="top")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.text(ax.get_xlim()[1], 0, "linear AR baseline  ", color=MUTED, fontsize=9, ha="right", va="bottom")
    for arm in res["arms"]:
        ax.scatter([], [], color=ARM_COLOR.get(arm, FG), label=ARM_LABEL.get(arm, arm), s=60)
    ax.legend(frameon=False, labelcolor=FG, loc="lower right")
    ax.set_xlabel("water-budget residual |ΔS − (P − ET − R)|, mm/month  (lower = closer to the physics)", color=FG)
    ax.set_ylabel("skill vs linear AR on next-month ΔS (MSESS)", color=FG)
    ax.set_title("Physics weight λ trades accuracy for budget closure", color=FG, loc="left", fontsize=14,
                 fontweight="bold")
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.6)
    fig.text(0.01, 0.01, "Test months 2022-01..2024-06, training region; small dots = seeds, large = seed mean. "
             "Emulation of NASA GLDAS-2.1 land water store at 1°.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(Path(args.demo) / "pinn_tradeoff.png", facecolor=BG)
    plt.close(fig)

    # skill map for the displayed configuration
    z = np.load(Path(args.results) / "skill_maps.npz")
    key = f"{best['arm']}|lam={best['lambda']}|msess_vs_ar"
    m = np.where(z["mask"], np.clip(z[key], -1, 1), np.nan)
    lat, lon = z["lat"], z["lon"]
    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=160, facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.pcolormesh(lon, lat, m, cmap="RdBu", vmin=-0.6, vmax=0.6, shading="auto")
    ax.axvline(90, color=FG, lw=0.8, ls="--")
    ax.text(93, -52, "held out of training\n(90°E–180°E)", color=FG, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("skill vs linear AR (blue = network better)", color=FG)
    cb.ax.tick_params(colors=MUTED)
    ax.set_title("Where the network beats a linear model (next-month storage change, 2022–2024)",
                 color=FG, loc="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.01, f"{ARM_LABEL.get(best['arm'], best['arm'])}, λ={best['lambda']:g}, mean of seeds. In deserts "
             "storage barely changes, so per-cell skill there is unstable; read the area-weighted totals instead. "
             "Emulation of NASA GLDAS-2.1 at 1°.", color=MUTED, fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(Path(args.demo) / "pinn_skill_map.png", facecolor=BG)
    plt.close(fig)
    print("wrote", Path(args.site) / "data" / "pinn.json", "and figures; best for display:", out["best_for_display"])


if __name__ == "__main__":
    main()
