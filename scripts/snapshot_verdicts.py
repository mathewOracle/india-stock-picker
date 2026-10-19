"""Dump every live verdict/score/plan to JSON, to prove refactors don't change behaviour.

Usage: python scripts/snapshot_verdicts.py OUT.json   then diff two snapshots.
"""

import json
import sys

from app import analysis


def snapshot() -> dict:
    return {r.symbol: {"verdict": r.verdict, "scores": {k: round(float(v), 6) for k, v in r.scores.items()},
                       "plan": [round(float(x), 6) for x in (r.plan.entry, r.plan.stop, r.plan.target1, r.plan.target2)],
                       "flags": sorted(r.flags), "gates": [[g.name, bool(g.passed)] for g in r.gates]}
            for r in analysis.build().reports}


if __name__ == "__main__":
    json.dump(snapshot(), open(sys.argv[1], "w"), indent=1, sort_keys=True)
    print(f"wrote {sys.argv[1]}")
