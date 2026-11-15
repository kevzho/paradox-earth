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


def qc_dataset(ds: xr.Dataset, variables=None) -> dict:
    """Chronology, units presence, missingness, physical-range violations, basic distribution summary."""
    variables = variables or [v for v in ds.data_vars]
    report = {"chronology": check_chronology(ds["time"].values), "variables": {}}
    land = None
    for v in variables:
        da = ds[v]
        if land is None:
            land = da.notnull().any("time")  # cells that are ever valid = land mask
        vals = da.where(land).values
        finite = np.isfinite(vals)
        lo, hi = PHYSICAL_RANGES.get(v, (-np.inf, np.inf))
        out_of_range = int(((vals < lo) | (vals > hi))[finite].sum()) if finite.any() else 0
        report["variables"][v] = {
            "units": da.attrs.get("units", "MISSING"),
            "missing_fraction_land": float(1 - finite.mean()) if vals.size else 1.0,
            "out_of_range": out_of_range,
            "min": float(np.nanmin(vals)) if finite.any() else None,
            "p50": float(np.nanmedian(vals)) if finite.any() else None,
            "max": float(np.nanmax(vals)) if finite.any() else None,
        }
    c = report["chronology"]
    report["passed"] = bool(c["monotonic"] and c["duplicates"] == 0 and
                            all(r["out_of_range"] == 0 and r["units"] != "MISSING"
                                for r in report["variables"].values()))
    return report
