"""S1: quality checks. Never modifies data; returns a report that travels with the provenance manifest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .gldas import PHYSICAL_RANGES


def check_chronology(time_values) -> dict:
    t = pd.DatetimeIndex(time_values)
    months = t.year * 12 + t.month
    gaps = np.diff(months)
    return {
        "n_steps": int(len(t)),
        "start": str(t.min().date()) if len(t) else None,
        "end": str(t.max().date()) if len(t) else None,
        "monotonic": bool(t.is_monotonic_increasing),
        "duplicates": int(t.duplicated().sum()),
        "missing_months": int((gaps[gaps > 1] - 1).sum()) if len(gaps) else 0,
        "regular_monthly": bool(len(gaps) == 0 or np.all(gaps == 1)),
    }


class QCAccumulator:
    """Streams months in, keeping only per-cell counters (O(grid) memory, independent of record length).
    Land mask = cells ever valid in the first variable. The median is exact on grids up to `p50_cells` cells and
    taken from a fixed strided sample of cells otherwise."""

    def __init__(self, variables, p50_cells: int = 10_000):
        self.variables, self.p50_cells = list(variables), p50_cells
        self.times, self.units, self.state = [], {}, None

    def _init(self, shape):
        n = shape[0] * shape[1]
        self.sample = np.arange(0, n, int(np.ceil(n / self.p50_cells)))
        self.state = {v: {"finite": np.zeros(n, np.int32), "oor": np.zeros(n, np.int32),
                          "min": np.full(n, np.inf), "max": np.full(n, -np.inf), "sample": []}
                      for v in self.variables}

    def update(self, ds: xr.Dataset):
        for v in self.variables:
            self.units.setdefault(v, ds[v].attrs.get("units", "MISSING"))
        if self.state is None:
            self._init(ds[self.variables[0]].shape[-2:])
        for i, t in enumerate(ds["time"].values):
            self.times.append(t)
            for v in self.variables:
                x = ds[v].isel(time=i).values.ravel().astype(float)
                fin = np.isfinite(x)
                lo, hi = PHYSICAL_RANGES.get(v, (-np.inf, np.inf))
                s = self.state[v]
                s["finite"] += fin
                s["oor"] += fin & ((x < lo) | (x > hi))
                s["min"] = np.fmin(s["min"], x)
                s["max"] = np.fmax(s["max"], x)
                s["sample"].append(x[self.sample])

    def report(self) -> dict:
        report = {"chronology": check_chronology(self.times), "variables": {}}
        nt = len(self.times)
        land = self.state[self.variables[0]]["finite"] > 0  # cells that are ever valid = land mask
        land_sample = land[self.sample]
        for v in self.variables:
            s = self.state[v]
            any_valid = bool(s["finite"][land].sum())
            samp = np.array(s["sample"])[:, land_sample]
            report["variables"][v] = {
                "units": self.units[v],
                "missing_fraction_land": float(1 - s["finite"][land].sum() / (land.sum() * nt)) if land.any() else 1.0,
                "out_of_range": int(s["oor"][land].sum()),
                "min": float(s["min"][land].min()) if any_valid else None,
                "p50": float(np.nanmedian(samp)) if any_valid and np.isfinite(samp).any() else None,
                "max": float(s["max"][land].max()) if any_valid else None,
                "p50_method": "exact" if len(self.sample) == len(land) else f"strided sample of {len(self.sample)} cells",
            }
        c = report["chronology"]
        report["passed"] = bool(c["monotonic"] and c["duplicates"] == 0 and
                                all(r["out_of_range"] == 0 and r["units"] != "MISSING"
                                    for r in report["variables"].values()))
        return report


def qc_dataset(ds: xr.Dataset, variables=None) -> dict:
    """Chronology, units presence, missingness, physical-range violations, basic distribution summary."""
    acc = QCAccumulator(variables or [v for v in ds.data_vars])
    acc.update(ds)
    return acc.report()
