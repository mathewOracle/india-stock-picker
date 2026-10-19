"""Append-only backtest ledger: the 10-year research is computed ONCE and then only extended.

Each row = one (date, symbol) evaluation: the signal flags, features, forward return and the simulated
trade. Rows are immutable once `final` (enough future bars existed to know the outcome); only recent,
still-open rows are recomputed, and only NEW dates are evaluated on later runs.

Integrity: per stock we keep a fingerprint (closes at 3 past dates). If Yahoo restated that stock's
prices non-uniformly (dividend/split), the fingerprint ratios drift and just that stock is replayed in full.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app import data
from app.research.backtest import EVAL_EVERY
from app.research.features import FEATURES

LEDGER_PATH = data.DB_PATH.parent / "research" / "ledger.csv.gz"
META_PATH = LEDGER_PATH.with_name("ledger_meta.json")
RESTATE_TOL = data.RESTATE_TOL
FLAT = ("date", "symbol", "buy", "strong", "fwd", "final")
OUTCOME = ("entry_date", "exit_date", "entry", "exit", "reason", "bars", "net_return", "r_multiple")
COLUMNS = FLAT + tuple(f"o_{c}" for c in OUTCOME) + tuple(FEATURES)


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    flat = []
    for r in rows:
        out = r["outcome"] or {}
        flat.append({**{c: r[c] for c in FLAT}, **{f"o_{c}": out.get(c) for c in OUTCOME}, **r["features"]})
    frame = pd.DataFrame(flat, columns=list(COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"])   # one canonical date type: merge keys must compare equal
    return frame


def frame_to_rows(df: pd.DataFrame) -> list[dict]:
    """Inverse of rows_to_frame (what the evidence code consumes)."""
    rows = []
    for rec in df.to_dict("records"):
        outcome = ({c: rec[f"o_{c}"] for c in OUTCOME} if isinstance(rec["o_entry_date"], str) else None)
        rows.append({**{c: rec[c] for c in FLAT}, "date": pd.Timestamp(rec["date"]).strftime("%Y-%m-%d"),
                     "outcome": outcome, "features": {f: rec[f] for f in FEATURES}})
    return rows


def load() -> tuple[pd.DataFrame | None, dict]:
    if not LEDGER_PATH.exists():
        return None, {"symbols": {}}
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {"symbols": {}}
    return pd.read_csv(LEDGER_PATH, parse_dates=["date"]), meta


def save(df: pd.DataFrame, meta: dict) -> None:
    """Both files go to temp names first and replace the old ones last: never a half-written ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv, tmp_meta = LEDGER_PATH.with_suffix(".tmp.gz"), META_PATH.with_suffix(".tmp")
    df.sort_values(["date", "symbol"]).to_csv(tmp_csv, index=False)
    tmp_meta.write_text(json.dumps(meta))
    os.replace(tmp_csv, LEDGER_PATH)
    os.replace(tmp_meta, META_PATH)


def fingerprint(history: pd.DataFrame, dates: list[pd.Timestamp]) -> dict[str, float]:
    return {d.strftime("%Y-%m-%d"): float(history["Close"].loc[d]) for d in dates if d in history.index}


def sample_dates(final_dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    return [final_dates[0], final_dates[len(final_dates) // 2], final_dates[-1]] if final_dates else []


def history_consistent(stored: dict[str, float], history: pd.DataFrame) -> bool:
    """True if today's prices are the stored ones up to a single uniform scale factor."""
    if not stored:
        return False
    ratios = []
    for day, close in stored.items():
        ts = pd.Timestamp(day)
        if ts not in history.index:
            return False
        ratios.append(float(history["Close"].loc[ts]) / close)
    return max(ratios) / min(ratios) - 1 <= RESTATE_TOL


@dataclass
class Plan:
    new_dates: list[pd.Timestamp]
    tasks: dict[str, list[pd.Timestamp]]            # symbol -> dates to (re)compute
    restated: list[str] = field(default_factory=list)
    reused: int = 0


def plan_update(ledger: pd.DataFrame | None, meta: dict, histories: dict[str, pd.DataFrame],
                bench_index: pd.DatetimeIndex, first_grid: list[pd.Timestamp]) -> Plan:
    """What must be computed so the ledger covers every grid date up to the latest data."""
    if ledger is None or ledger.empty:
        return Plan(first_grid, {s: first_grid for s in histories})
    grid = sorted(set(ledger["date"]) | set(_extend_grid(max(ledger["date"]), bench_index)))
    new_dates = [d for d in grid if d > max(ledger["date"])]
    tasks, restated = {}, []
    for symbol, history in histories.items():
        own = ledger[ledger["symbol"] == symbol]
        if own.empty or not history_consistent(meta["symbols"].get(symbol, {}).get("fp", {}), history):
            if not own.empty:
                restated.append(symbol)
            tasks[symbol] = grid
            continue
        pending = list(own.loc[~own["final"].astype(bool), "date"])
        if pending or new_dates:
            tasks[symbol] = sorted(set(pending) | set(new_dates))
    return Plan(new_dates, tasks, restated, reused=len(ledger))


def _extend_grid(last: pd.Timestamp, bench_index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Continue the weekly grid from the last stored date (never re-anchor: that would shift every date)."""
    pos = int(bench_index.searchsorted(last))
    return list(bench_index[pos + EVAL_EVERY::EVAL_EVERY])


def merge_rows(ledger: pd.DataFrame | None, fresh: pd.DataFrame, tasks: dict[str, list[pd.Timestamp]]) -> pd.DataFrame:
    """Replace exactly the recomputed (symbol, date) pairs; everything else stays untouched."""
    if ledger is None or ledger.empty:
        return fresh
    redo = pd.MultiIndex.from_tuples([(s, d) for s, ds in tasks.items() for d in ds])
    keep = ~pd.MultiIndex.from_frame(ledger[["symbol", "date"]]).isin(redo)
    return pd.concat([ledger[keep], fresh], ignore_index=True)


def new_meta(ledger: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> dict:
    symbols = {}
    for symbol, history in histories.items():
        final = sorted(ledger.loc[(ledger["symbol"] == symbol) & ledger["final"].astype(bool), "date"])
        if final:
            symbols[symbol] = {"fp": fingerprint(history, sample_dates(final))}
    return {"symbols": symbols, "updated_through": ledger["date"].max().strftime("%Y-%m-%d")}
