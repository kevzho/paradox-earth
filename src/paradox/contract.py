"""Common data contract (Manifest §5.1) and analysis record (Manifest §16)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SourceManifest:
    provider: str
    dataset: str
    variable: str
    version: str
    units: str
    temporal_resolution: str
    spatial_resolution: str
    effective_spatial_support: str
    coverage: dict = field(default_factory=dict)  # {"time": [start, end], "bbox": [w, s, e, n]}
    quality_flags: list = field(default_factory=list)
    missing_fraction: float | None = None
    provenance: dict = field(default_factory=dict)  # files, checksums, retrieval method, processing steps
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


def file_checksum(path: str | Path, n_bytes: int = 1 << 20) -> str:
    """sha256 of the first `n_bytes` (fast fingerprint for large NetCDF files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def analysis_record(region_id: str, dataset: str, variable: str, window, trend: dict | None = None,
                    variability: dict | None = None, manifest_path: str | None = None,
                    lens: str = "climate", quality: dict | None = None) -> dict:
    """Build a §16 analysis record. Spatial/ML/validation fields are left for downstream stages."""
    return {
        "region_id": region_id,
        "lens": lens,
        "dataset": dataset,
        "variable": variable,
        "window": [str(w) for w in window],
        "trend": trend or {"slope": None, "ci95": None, "units": None, "method": "seasonal_regression_HAC"},
        "variability": variability or {"baseline_change": None, "egarch_status": "not_evaluated"},
        "spatial": {"lisa_label": None, "inference_status": "not_evaluated"},
        "ml": {"ranking_score": None, "model_version": None},
        "validation": {"status": "pending"},
        "quality": quality or {},
        "provenance": {"manifest_path": manifest_path},
    }
