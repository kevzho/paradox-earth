"""Fetch the datasets listed in DATA_TO_USE.md via NASA Earthdata (requires a free login).

  pip install earthaccess
  python scripts/download_data.py gldas --start 2000-01 --end 2025-12 --bbox -125 24 -66 50
  python scripts/download_data.py grace --dry-run

Credentials: ~/.netrc or EARTHDATA_USERNAME / EARTHDATA_PASSWORD, else an interactive prompt.
"""
import argparse
from pathlib import Path

SOURCES = {  # key: (short_name, version) -- versions marked "verify" in DATA_TO_USE.md
    "gldas": ("GLDAS_NOAH025_M", "2.1"),
    "gldas20": ("GLDAS_NOAH025_M", "2.0"),  # Princeton-forced control run, 1948-2014 (same Noah model)
    "fldas": ("FLDAS_NOAH01_C_GL_M", "001"),  # Noah 3.6.1 with homogeneous MERRA-2 + CHIRPS forcing
    "viirs_ntl": ("VNP46A3", None),
    "modis_lc": ("MCD12C1", "061"),
    "grace": ("TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4", None),
    "modis_vi": ("MOD13C2", "061"),
}

ap = argparse.ArgumentParser()
ap.add_argument("source", choices=sorted(SOURCES))
ap.add_argument("--start", default="2000-01")
ap.add_argument("--end", default="2025-12")
ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
ap.add_argument("--out", default=None)
ap.add_argument("--dry-run", action="store_true", help="list granules only")
args = ap.parse_args()

import earthaccess  # noqa: E402

short, version = SOURCES[args.source]
earthaccess.login()
kw = {"short_name": short, "temporal": (f"{args.start}-01", f"{args.end}-28")}
if version:
    kw["version"] = version
if args.bbox:
    kw["bounding_box"] = tuple(args.bbox)
granules = earthaccess.search_data(**kw)
print(f"{short} v{version or '?'}: {len(granules)} granules")
if not args.dry_run and granules:
    out = Path(args.out or f"data/raw/{args.source}")
    out.mkdir(parents=True, exist_ok=True)
    if args.source == "fldas":  # never mix the GDAS-forced CP stream into the C stream
        granules = [g for g in granules if "_CP_" not in g["umm"]["GranuleUR"]]
    earthaccess.download(granules, str(out), threads=8)
    print("saved to", out, "- GLDAS files are global; the pipeline applies --bbox at load time.")
