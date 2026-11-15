"""Physics-informed land-water forecaster (Manifest §10, Tier-3 research extension).

Target (§10.2): next-month GLDAS land-water store S = soil water 0-200 cm + SWE + canopy (mm). This is NOT total
terrestrial water storage (no groundwater / surface water).

Weather-pattern dynamics: a small spatiotemporal CNN reads the last `window` months of the state and weather
drivers [S, P, ET, R, T] (stacked along channels) and predicts next-month [dS, P, ET, R]. With
`forcing_known=True` the next-month weather (P, T) is supplied as input (forcing-driven emulation, e.g. from a
reanalysis or NWP); otherwise the model must forecast the weather too.

Loss (§10.4):  L = L_prediction + lambda * L_water_balance
    water-balance residual  r = dS_pred - F,  F = P - ET - R (mm/month)
    alignment 'forward' : F = F_{t+1}                 (end-of-month storages, e.g. synthetic bucket)
    alignment 'centered': F = (F_t + F_{t+1}) / 2     (monthly-MEAN storages, as in GLDAS monthly files)
Ablate lambda = 0 vs lambda > 0 with `ablation()`. Baselines: persistence, seasonal naive, pooled linear AR.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

CHANNELS = ["S", "P", "ET", "R", "T"]
S, P, ET, R, T = range(5)


@dataclass
class PINNConfig:
    window: int = 6
    hidden: int = 32
    lam: float = 0.1
    alignment: str = "centered"      # 'centered' for GLDAS monthly means, 'forward' for end-of-month states
    forcing_known: bool = True
    train_frac: float = 0.7
    val_frac: float = 0.15
    spatial_holdout_frac: float = 0.25  # eastern-most lon columns withheld from training loss
    epochs: int = 60
    batch: int = 16
    lr: float = 1e-3
    patch: int = 64                   # random crop size during training for large grids
    seed: int = 0
    device: str = "cpu"


# ---------------------------------------------------------------- data


def arrays_from_dataset(ds: xr.Dataset, store: str = "land_water_store"):
    """(T, 5, H, W) float array in mm / mm-month / K, cell mask (H, W), month numbers (T,)."""
    a = np.stack([ds[store].values, ds["Rainf_f_tavg"].values, ds["Evap_tavg"].values,
                  ds["runoff"].values, ds["Tair_f_inst"].values], axis=1).astype(np.float32)
    mask = np.isfinite(a).all(axis=(0, 1))
    return np.nan_to_num(a), mask, ds["time"].dt.month.values


class Prepared:
    """Normalised samples, chronological splits and train-only statistics."""

    def __init__(self, arr, mask, months, cfg: PINNConfig):
        self.cfg, self.arr, self.mask, self.months = cfg, arr, mask, months
        nt, _, h, w = arr.shape
        L = cfg.window
        self.t_idx = np.arange(L - 1, nt - 1)  # input ends at t, target is t+1
        n = len(self.t_idx)
        n_tr, n_va = int(n * cfg.train_frac), int(n * cfg.val_frac)
        self.split = {"train": self.t_idx[:n_tr], "val": self.t_idx[n_tr:n_tr + n_va], "test": self.t_idx[n_tr + n_va:]}
        if min(len(v) for v in self.split.values()) == 0:
            raise ValueError("series too short for train/val/test split")
        # spatial holdout: eastern columns
        self.train_cells = mask.copy()
        k = int(round(w * cfg.spatial_holdout_frac))
        if k:
            self.train_cells[:, w - k:] = False
        self.holdout_cells = mask & ~self.train_cells
        # train-only normalisation (times up to the last training target, train cells only)
        t_last = self.split["train"][-1] + 1
        sub = arr[:t_last + 1][:, :, self.train_cells]          # (t, C, ncells)
        self.mu = sub.mean(axis=(0, 2))
        self.sd = sub.std(axis=(0, 2)) + 1e-6
        ds_tr = np.diff(arr[:t_last + 1, S], axis=0)[:, self.train_cells]
        self.mu_ds, self.sd_ds = float(ds_tr.mean()), float(ds_tr.std() + 1e-6)

    def norm(self, x, c):
        return (x - self.mu[c]) / self.sd[c]

    def inputs(self, t: int) -> np.ndarray:
        cfg, a = self.cfg, self.arr
        win = (a[t - cfg.window + 1:t + 1] - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        h, w = a.shape[-2:]
        mth = self.months[t + 1]
        extra = [np.full((h, w), np.sin(2 * np.pi * mth / 12)), np.full((h, w), np.cos(2 * np.pi * mth / 12)),
                 self.mask.astype(np.float32)]
        if cfg.forcing_known:
            extra += [self.norm(a[t + 1, P], P), self.norm(a[t + 1, T], T)]
        x = np.concatenate([win.reshape(-1, h, w), np.stack(extra)], axis=0)
        return (x * self.mask).astype(np.float32)

    def targets(self, t: int) -> np.ndarray:
        a = self.arr
        y = np.stack([(a[t + 1, S] - a[t, S] - self.mu_ds) / self.sd_ds,
                      self.norm(a[t + 1, P], P), self.norm(a[t + 1, ET], ET), self.norm(a[t + 1, R], R)])
        return y.astype(np.float32)

    def batch(self, ts):
        x = torch.from_numpy(np.stack([self.inputs(t) for t in ts]))
        y = torch.from_numpy(np.stack([self.targets(t) for t in ts]))
        f_t = torch.from_numpy(np.stack([self.arr[t, P] - self.arr[t, ET] - self.arr[t, R] for t in ts]))
        p_next = torch.from_numpy(np.stack([self.arr[t + 1, P] for t in ts]))
        return x, y, f_t, p_next

    @property
    def n_in(self) -> int:
        return self.cfg.window * len(CHANNELS) + 3 + (2 if self.cfg.forcing_known else 0)


# ---------------------------------------------------------------- model


class WaterBalanceNet(nn.Module):
    def __init__(self, c_in: int, hidden: int = 32, c_out: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(hidden, c_out, 1),
        )

    def forward(self, x):
        return self.net(x)


def physical(out: torch.Tensor, prep: Prepared, p_next: torch.Tensor | None = None):
    """De-normalise network output to mm: returns dS, P, ET, R (each (B, H, W))."""
    mu, sd = prep.mu, prep.sd
    ds = out[:, 0] * prep.sd_ds + prep.mu_ds
    p = out[:, 1] * float(sd[P]) + float(mu[P]) if p_next is None else p_next
    et = out[:, 2] * float(sd[ET]) + float(mu[ET])
    r = out[:, 3] * float(sd[R]) + float(mu[R])
    return ds, p, et, r


def water_balance_residual(ds, p, et, r, f_t, alignment: str):
    f_next = p - et - r
    f = 0.5 * (f_t + f_next) if alignment == "centered" else f_next
    return ds - f


def _losses(model, prep: Prepared, x, y, f_t, p_next, cell_mask):
    cfg = prep.cfg
    out = model(x)
    m = cell_mask.float()
    w = torch.ones(4)
    if cfg.forcing_known:
        w[1] = 0.0  # P_{t+1} is an input, do not "predict" it
    se = ((out - y) ** 2) * w[None, :, None, None]
    l_pred = (se.sum(1) * m).sum() / (m.sum() * w.sum() * out.shape[0])
    ds, p, et, r = physical(out, prep, p_next if cfg.forcing_known else None)
    res = water_balance_residual(ds, p, et, r, f_t, cfg.alignment) / prep.sd_ds
    l_wb = ((res ** 2) * m).sum() / (m.sum() * out.shape[0])
    return l_pred, l_wb


def _crop(tensors, size, rng):
    h, w = tensors[0].shape[-2:]
    if h <= size and w <= size:
        return tensors
    i = int(rng.integers(0, max(h - size, 0) + 1))
    j = int(rng.integers(0, max(w - size, 0) + 1))
    return [t[..., i:i + size, j:j + size] for t in tensors]


def train(prep: Prepared, cfg: PINNConfig | None = None, verbose: bool = False):
    cfg = cfg or prep.cfg
    prep.cfg = cfg
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = WaterBalanceNet(prep.n_in, cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_mask = torch.from_numpy(prep.train_cells)
    best, best_state, history = np.inf, None, []
    xv, yv, fv, pv = prep.batch(prep.split["val"])
    for ep in range(cfg.epochs):
        model.train()
        order = rng.permutation(prep.split["train"])
        for i in range(0, len(order), cfg.batch):
            x, y, f_t, p_next = prep.batch(order[i:i + cfg.batch])
            m = train_mask.expand(x.shape[0], *train_mask.shape)
            x, y, f_t, p_next, m = _crop([x, y, f_t, p_next, m], cfg.patch, rng)
            l_pred, l_wb = _losses(model, prep, x, y, f_t, p_next, m[0])
            loss = l_pred + cfg.lam * l_wb
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():  # model selection on validation PREDICTION loss only (fair across lambda)
            vp, vw = _losses(model, prep, xv, yv, fv, pv, train_mask)
        history.append({"epoch": ep, "val_pred": float(vp), "val_wb": float(vw)})
        if float(vp) < best:
            best, best_state = float(vp), {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and ep % 10 == 0:
            print(f"ep {ep:3d} val_pred={float(vp):.4f} val_wb={float(vw):.4f}")
    model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------- evaluation & baselines


def _metrics(pred, truth, cells, budget_res=None):
    e = (pred - truth)[:, cells]
    out = {"mae_mm": float(np.abs(e).mean()), "rmse_mm": float(np.sqrt((e ** 2).mean()))}
    if budget_res is not None:
        out["budget_residual_mae_mm"] = float(np.abs(budget_res[:, cells]).mean())
    return out


def predict_storage(model, prep: Prepared, ts):
    x, _, f_t, p_next = prep.batch(ts)
    with torch.no_grad():
        out = model(x)
        ds, p, et, r = physical(out, prep, p_next if prep.cfg.forcing_known else None)
        res = water_balance_residual(ds, p, et, r, f_t, prep.cfg.alignment)
    s_next = prep.arr[ts, S] + ds.numpy()
    return s_next, res.numpy()


def _ar_features(prep: Prepared, t):
    a, mth = prep.arr, prep.months[t + 1]
    feats = [a[t, S], a[t - 1, S], a[t, P], a[t, ET], a[t, R], a[t, T]]
    if prep.cfg.forcing_known:
        feats += [a[t + 1, P], a[t + 1, T]]
    h, w = a.shape[-2:]
    feats += [np.full((h, w), np.sin(2 * np.pi * mth / 12)), np.full((h, w), np.cos(2 * np.pi * mth / 12)),
              np.ones((h, w))]
    return np.stack(feats, -1)


def baselines(prep: Prepared, ts) -> dict:
    a = prep.arr
    persistence = a[ts, S]
    seasonal = np.stack([a[t + 1 - 12, S] if t + 1 >= 12 else a[t, S] for t in ts])
    tr = [t for t in prep.split["train"] if t >= 1]
    X = np.concatenate([_ar_features(prep, t)[prep.train_cells] for t in tr])
    yv = np.concatenate([(a[t + 1, S] - a[t, S])[prep.train_cells] for t in tr])
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    ar = np.stack([a[t, S] + _ar_features(prep, t) @ coef for t in ts])
    return {"persistence": persistence, "seasonal_naive": seasonal, "linear_ar": ar}


def evaluate(model, prep: Prepared, split: str = "test") -> dict:
    ts = prep.split[split]
    truth = prep.arr[ts + 1, S]
    s_pred, res = predict_storage(model, prep, ts)
    out = {}
    for name, cells in [("in_region", prep.train_cells), ("spatial_holdout", prep.holdout_cells)]:
        if not cells.any():
            continue
        out[name] = {"pinn": _metrics(s_pred, truth, cells, res)}
        for b, pred in baselines(prep, ts).items():
            out[name][b] = _metrics(pred, truth, cells)
    return out


def ablation(ds: xr.Dataset, cfg: PINNConfig, lams=(0.0, 0.1), store: str = "land_water_store") -> dict:
    """Train identical models differing only in lambda; report test metrics for each (Manifest §10.4-10.5)."""
    arr, mask, months = arrays_from_dataset(ds, store)
    results = {"config": asdict(cfg), "runs": {}}
    for lam in lams:
        c = PINNConfig(**{**asdict(cfg), "lam": lam})
        prep = Prepared(arr, mask, months, c)
        model, hist = train(prep, c)
        results["runs"][f"lambda={lam}"] = {"test": evaluate(model, prep, "test"),
                                           "best_val_pred": min(h["val_pred"] for h in hist)}
        results["runs"][f"lambda={lam}"]["_model"] = model
    results["note"] = ("Deterministic point forecasts: no CRPS/interval coverage. GLDAS is model output, so skill is "
                       "emulation skill vs GLDAS, not vs observations. GRACE is a complementary total-storage check, "
                       "not ground truth for this target.")
    return results
