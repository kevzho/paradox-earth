"""§7 spatial statistics on per-tile maps (trend slope, log variance ratio): global Moran's I, local Moran's I (LISA)
with conditional permutation inference and Benjamini-Hochberg FDR, plus robustness to an alternative neighbourhood
(k-nearest neighbours on the sphere) and Getis-Ord Gi*.

Primary weights (§7.3): queen contiguity on the tile lattice, row-standardised; on a global domain the lattice wraps at
±180° so dateline tiles are neighbours. Tiles with no valued neighbour are reported as `no_neighbors`, never tested.

Inference: two-sided conditional-permutation pseudo p-values (p_sim), then BH. esda's two-sided p_sim has a floor of
about 3/(permutations+1) (0.003 at 999), and BH can only reject at that floor once >= n * floor / alpha tiles reach it
(~6% of tiles at 999, ~0.6% at 9,999), so the default is the manifest's 9,999 (§7.4); 999 has almost no power here.
A normal approximation of the permutation z-score is NOT used: it is anti-conservative in the tails for skewed or
heavy-tailed maps (false clusters on iid t(2) or lognormal maps).
Labels (§7.2) are assigned only where the BH-adjusted q < alpha: `HH_cluster`, `LL_cluster` (clusters) and
`HL_outlier`, `LH_outlier` (spatial outliers); every other tested tile is `inconclusive`. Caveats (§7.5): BH assumes
independence or positive dependence among tests; neighbouring LISA tests share data, so q-values are a documented
convention, not a guarantee. Permutation nulls shuffle values across the map, which ignores large-scale spatial
structure (e.g. smooth GLDAS forcing); dependence-preserving nulls are Experiment E.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .contract import write_json

FIELDS = {  # field name -> (record path, transform)
    "trend_slope": (("trend", "slope"), None),
    "log_variance_ratio": (("variability", "baseline_change", "variance_ratio_late_over_early"), np.log),
}
QUADRANTS = {1: "HH_cluster", 2: "LH_outlier", 3: "LL_cluster", 4: "HL_outlier"}  # esda quadrant codes
QUEEN = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------- weights


def lattice_neighbors(tiles: pd.DataFrame, ids) -> dict:
    """Queen neighbours among `ids` on the regular tile lattice; wraps in longitude when the domain spans 360°."""
    t = tiles.set_index("region_id").loc[list(ids)]
    deg = float((t["east"] - t["west"]).iloc[0])
    lon0, lat0 = float(tiles["west"].min()), float(tiles["south"].min())
    n_global = int(round(360.0 / deg))
    wraps = float(tiles["east"].max() - tiles["west"].min()) >= 360.0 - 1e-6
    ij = {rid: (int(round((s - lat0) / deg)), int(round((w - lon0) / deg)))
          for rid, s, w in zip(t.index, t["south"], t["west"])}
    at = {v: k for k, v in ij.items()}
    nbrs = {}
    for rid, (i, j) in ij.items():
        out = []
        for di, dj in QUEEN:
            jj = (j + dj) % n_global if wraps else j + dj
            other = at.get((i + di, jj))
            if other is not None and other != rid:
                out.append(other)
        nbrs[rid] = sorted(set(out))
    return nbrs


def knn_neighbors(tiles: pd.DataFrame, ids, k: int = 8) -> dict:
    """k nearest tile centres by chord distance on the unit sphere (wraps the dateline and shrinks toward poles)."""
    from sklearn.neighbors import NearestNeighbors

    t = tiles.set_index("region_id").loc[list(ids)]
    lat = np.deg2rad((t["south"] + t["north"]) / 2)
    lon = np.deg2rad((t["west"] + t["east"]) / 2)
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    k = min(k, len(t) - 1)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(xyz).kneighbors(xyz)
    names = np.array(t.index)
    return {names[i]: sorted(names[row[1:]]) for i, row in enumerate(idx)}


def to_w(nbrs: dict):
    """libpysal W (row-standardised) without islands, plus the list of island ids."""
    from libpysal.weights import W

    islands = [r for r, n in nbrs.items() if not n]
    keep = {r: n for r, n in nbrs.items() if n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = W(keep, id_order=list(keep))
    w.transform = "r"
    return w, islands


# ---------------------------------------------------------------- statistics


def moran_global(y: np.ndarray, w, permutations: int, seed: int) -> dict:
    from esda.moran import Moran

    np.random.seed(seed)  # esda's global Moran draws from the global numpy RNG
    m = Moran(y, w, transformation="r", permutations=permutations)
    # esda's global p_sim is folded one-sided (tail on the side of the observed I); p_norm is two-sided
    return {"I": float(m.I), "expected_I": float(m.EI), "z_sim": float(m.z_sim),
            "p_sim_one_sided": float(m.p_sim), "p_norm_two_sided": float(m.p_norm), "n": int(m.n),
            "permutations": permutations}


def lisa(y: np.ndarray, w, permutations: int, alpha: float, seed: int, n_jobs: int = 1) -> pd.DataFrame:
    """Local Moran's I with conditional permutation pseudo p-values, BH-FDR q-values and cluster/outlier labels."""
    from esda.moran import Moran_Local

    lm = Moran_Local(y, w, transformation="r", permutations=permutations, seed=seed, n_jobs=n_jobs,
                     alternative="two-sided", keep_simulations=False)
    if not (np.isfinite(lm.p_sim).all() and np.isfinite(lm.Is).all()):
        raise ValueError("non-finite local Moran statistics (degenerate map?)")
    q = multipletests(lm.p_sim, alpha=alpha, method="fdr_bh")[1]
    labels = np.where(q < alpha, [QUADRANTS[int(c)] for c in lm.q], "inconclusive")
    return pd.DataFrame({"Ii": lm.Is, "quadrant": [QUADRANTS[int(c)] for c in lm.q], "p_sim": lm.p_sim,
                         "q_bh": q, "label": labels}, index=w.id_order)


def gi_star(y: np.ndarray, w) -> pd.DataFrame:
    """Getis-Ord Gi* z-scores (Ord & Getis 1995, binary weights including self) with normal p-values. Closed form, so
    valid for signed values such as trend slopes, unlike the ratio form G = sum(w y) / sum(y)."""
    n = len(y)
    a = w.sparse.copy().astype(float)
    a.data[:] = 1.0  # binary contiguity
    a = a + _identity(n)
    xbar, s = y.mean(), y.std()
    wsum = np.asarray(a.sum(1)).ravel()
    w2sum = np.asarray(a.multiply(a).sum(1)).ravel()
    num = a @ y - xbar * wsum
    den = s * np.sqrt((n * w2sum - wsum ** 2) / (n - 1))
    z = num / den
    return pd.DataFrame({"gi_z": z, "gi_p": 2 * stats.norm.sf(np.abs(z))}, index=w.id_order)


def _identity(n):
    from scipy import sparse

    return sparse.identity(n, format="csr")


# ---------------------------------------------------------------- orchestration


def analyse_field(values: pd.Series, tiles: pd.DataFrame, permutations: int = 9999, alpha: float = 0.05,
                  seed: int = 0, knn_k: int = 8, n_jobs: int = 1):
    """All §7 statistics for one map. Returns (per-tile table, summary dict)."""
    values = values[np.isfinite(values)]
    if not values.std() > 0:
        raise ValueError("zero-variance map: local Moran is undefined")
    w, islands = to_w(lattice_neighbors(tiles, values.index))
    if w.n < 3:
        raise ValueError(f"only {w.n} tiles have a queen neighbour")
    y = values.loc[w.id_order].to_numpy(float)
    table = lisa(y, w, permutations, alpha, seed, n_jobs).join(gi_star(y, w))
    table.insert(0, "value", y)

    for rid in islands:
        table.loc[rid, ["value", "label"]] = [values[rid], "no_neighbors"]
    w_knn, _ = to_w(knn_neighbors(tiles, values.index, knn_k))
    alt = lisa(values.loc[w_knn.id_order].to_numpy(float), w_knn, permutations, alpha, seed, n_jobs)
    table["label_knn"] = alt["label"].reindex(table.index)  # kNN gives every tile neighbours, islands included

    sig = table["label"].isin(QUADRANTS.values())
    summary = {
        "n_tiles": int(len(values)), "n_islands": len(islands),
        "global_moran": moran_global(y, w, permutations, seed),
        "global_moran_knn": moran_global(values.loc[w_knn.id_order].to_numpy(float), w_knn, permutations, seed),
        "label_counts": table["label"].value_counts().to_dict(),
        "n_raw_p_below_alpha": int((table["p_sim"] < alpha).sum()),
        "robustness": {
            "knn_k": knn_k,
            "agreement_on_significant": float((table.loc[sig, "label"] == table.loc[sig, "label_knn"]).mean())
            if sig.any() else None,
            "gi_star_sign_agrees_on_clusters": _gi_agreement(table),
        },
        "method": {"weights": "queen lattice (dateline wrap when global), row-standardised",
                   "alt_weights": f"kNN k={knn_k} on unit sphere",
                   "inference": "two-sided conditional permutation pseudo p-value (p_sim) -> BH",
                   "permutations": permutations, "multiple_testing": f"Benjamini-Hochberg, q < {alpha}", "seed": seed},
    }
    return table, summary


def _gi_agreement(table: pd.DataFrame):
    """Share of LISA HH/LL clusters whose Gi* z has the matching sign; None when there are no clusters."""
    hh, ll = table["label"].eq("HH_cluster"), table["label"].eq("LL_cluster")
    n = int(hh.sum() + ll.sum())
    if not n:
        return None
    return float(((table.loc[hh, "gi_z"] > 0).sum() + (table.loc[ll, "gi_z"] < 0).sum()) / n)


def _get(rec: dict, path):
    for k in path:
        rec = rec.get(k) if isinstance(rec, dict) else None
    return rec


def run_spatial(records: list, tiles: pd.DataFrame, out_dir, permutations: int = 9999, alpha: float = 0.05,
                seed: int = 0, n_jobs: int = 1, fields=None) -> dict:
    """Fill record['spatial'] for every record; write spatial/<variable>__<field>.csv and spatial_summary.json."""
    out_dir = Path(out_dir)
    (out_dir / "spatial").mkdir(parents=True, exist_ok=True)
    fields = fields or FIELDS
    by_var = {}
    for r in records:
        by_var.setdefault(r["variable"], []).append(r)
    summary, tables = {}, {}
    for var, recs in by_var.items():
        for field, (path, fn) in fields.items():
            raw = pd.Series({r["region_id"]: _get(r, path) for r in recs}, dtype=float)
            vals = fn(raw) if fn is not None else raw
            if vals.notna().sum() < 10:
                summary[f"{var}__{field}"] = {"status": "insufficient_tiles", "n_tiles": int(vals.notna().sum())}
                continue
            try:  # one degenerate map must not abort the run (or the other maps)
                table, s = analyse_field(vals, tiles, permutations, alpha, seed, n_jobs=n_jobs)
            except Exception as e:  # noqa: BLE001
                summary[f"{var}__{field}"] = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
                continue
            table.index.name = "region_id"
            table.to_csv(out_dir / "spatial" / f"{var}__{field}.csv")
            summary[f"{var}__{field}"], tables[(var, field)] = {"status": "ok", **s}, table
    for r in records:
        per_field = {}
        for field in fields:
            t = tables.get((r["variable"], field))
            if t is None or r["region_id"] not in t.index:
                per_field[field] = {"label": "missing"}
                continue
            row = t.loc[r["region_id"]]
            knn = row.get("label_knn")
            per_field[field] = {"label": row["label"], "label_knn": knn if isinstance(knn, str) else None,
                                "Ii": _f(row["Ii"]), "p_sim": _f(row["p_sim"]), "q_bh": _f(row["q_bh"]),
                                "gi_z": _f(row["gi_z"])}
        primary = per_field.get("trend_slope", {}).get("label")
        r["spatial"] = {"lisa_label": primary,
                        "inference_status": "evaluated" if primary not in (None, "missing", "no_neighbors")
                        else "not_evaluated",
                        "fields": per_field, "permutations": permutations, "alpha_fdr": alpha}
    write_json(out_dir / "spatial_summary.json", summary)
    return summary


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def run_from_outputs(out_dir="outputs", **kw) -> dict:
    """Re-run §7 on an existing stats run (records.json + tiles.csv) without recomputing trends/EGARCH."""
    out_dir = Path(out_dir)
    records = json.loads((out_dir / "records.json").read_text())
    summary = run_spatial(records, pd.read_csv(out_dir / "tiles.csv"), out_dir, **kw)
    write_json(out_dir / "records.json", records)
    return summary
