"""Integrity check: does the live ledger still reproduce the archived baseline research?

    python -m scripts.verify_ledger [archive_dir]

Compares every baseline (date, symbol) row's signal flags, forward return and all features against the
current ledger. Dividend restatements legitimately rescale old prices, so tiny float differences are
tolerated; flag flips or missing rows are not.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.research import ledger
from app.research.features import FEATURES

ARCHIVE = Path("data/research/archive/baseline-2026-10-19")
TOL = 1e-6


def verify(archive: Path = ARCHIVE) -> dict:
    current, _ = ledger.load()
    base = pd.read_csv(archive / "dataset.csv.gz", parse_dates=["date"])
    merged = base.merge(current, on=["date", "symbol"], suffixes=("_old", "_new"))
    flips = {c: int((merged[f"{c}_old"] != merged[f"{c}_new"]).sum()) for c in ("buy", "strong")}
    numeric = ["fwd", *FEATURES]
    diffs = {c: float(np.nanmax(np.abs(merged[f"{c}_old"] - merged[f"{c}_new"]))) for c in numeric
             if merged[f"{c}_old"].notna().any()}
    nan_mismatch = sum(int((merged[f"{c}_old"].isna() != merged[f"{c}_new"].isna()).sum()) for c in numeric)
    return {"baseline_rows": len(base), "ledger_rows": len(current), "matched": len(merged),
            "missing_from_ledger": len(base) - len(merged), "flag_flips": flips,
            "max_abs_diff": max(diffs.values()), "worst_column": max(diffs, key=diffs.get),
            "nan_mismatches": nan_mismatch,
            "intact": len(merged) == len(base) and not any(flips.values()) and nan_mismatch == 0
            and max(diffs.values()) <= TOL}


if __name__ == "__main__":
    result = verify(Path(sys.argv[1]) if len(sys.argv) > 1 else ARCHIVE)
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("INTACT" if result["intact"] else "MISMATCH")
    sys.exit(0 if result["intact"] else 1)
