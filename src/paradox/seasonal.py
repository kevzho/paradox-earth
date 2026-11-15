"""S2: seasonal adjustment via monthly climatology. Raw series is preserved; climatology can be
estimated on a reference (training-only) window so held-out periods never inform preprocessing."""
from __future__ import annotations

import pandas as pd


def monthly_climatology(y: pd.Series, ref_period: tuple | None = None) -> pd.Series:
    """Mean for each calendar month (index 1..12), estimated only inside ref_period if given."""
    ref = y.loc[ref_period[0]:ref_period[1]] if ref_period else y
    clim = ref.groupby(ref.index.month).mean()
    if clim.reindex(range(1, 13)).isna().any():
        raise ValueError("reference period does not cover all 12 calendar months")
    return clim


def anomalies(y: pd.Series, ref_period: tuple | None = None) -> pd.DataFrame:
    """Return DataFrame with raw, climatology and anomaly columns (missing values stay missing)."""
    clim = monthly_climatology(y, ref_period)
    c = pd.Series(clim.reindex(y.index.month).values, index=y.index)
    return pd.DataFrame({"raw": y, "climatology": c, "anomaly": y - c})
