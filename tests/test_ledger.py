"""The incremental ledger must equal a from-scratch run, and must only compute what is new."""

import numpy as np
import pandas as pd
import pytest

from app.research import ledger
from app.research.backtest import FINAL_BARS, eval_dates, replay_stock
from tests.test_backtest import random_walk

SYMBOLS = ["A.NS", "B.NS", "C.NS"]


def build(histories: dict, bench: pd.DataFrame, existing=None, meta=None):
    """Mini version of pipeline.run_backtest (in-process, no pool)."""
    plan = ledger.plan_update(existing, meta or {"symbols": {}}, histories, bench.index, eval_dates(bench.index))
    rows = [r for s, dates in plan.tasks.items() for r in replay_stock(s, histories[s], bench["Close"], dates)]
    fresh = ledger.rows_to_frame(rows)
    frame = ledger.merge_rows(existing, fresh, plan.tasks) if plan.tasks else existing
    return frame, ledger.new_meta(frame, histories), plan


def world(n=900):
    histories = {s: random_walk(n=n, seed=i + 1) for i, s in enumerate(SYMBOLS)}
    return histories, random_walk(n=n, seed=99)


def canonical(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def test_incremental_equals_from_scratch():
    histories, bench = world()
    scratch, _, _ = build(histories, bench)
    cut = 700
    part, meta, _ = build({s: h.iloc[:cut] for s, h in histories.items()}, bench.iloc[:cut])
    extended, _, plan = build(histories, bench, part, meta)
    pd.testing.assert_frame_equal(canonical(extended), canonical(scratch), check_dtype=False)
    assert len(plan.new_dates) > 0 and not plan.restated


def test_second_run_with_no_new_data_computes_nothing_new():
    histories, bench = world()
    ledger_a, meta, _ = build(histories, bench)
    _, _, plan = build(histories, bench, ledger_a, meta)
    assert plan.new_dates == []
    pending = ledger_a.loc[~ledger_a["final"].astype(bool)]
    assert sum(len(d) for d in plan.tasks.values()) == len(pending)   # only still-open rows are revisited
    assert len(pending) < len(ledger_a) * 0.1


def test_rows_become_final_only_when_enough_future_exists():
    histories, bench = world(n=700)
    frame, _, _ = build(histories, bench)
    last_final = frame.loc[frame["final"].astype(bool), "date"].max()
    pos = histories["A.NS"].index.get_loc(last_final)
    assert pos + FINAL_BARS < len(histories["A.NS"])
    assert not frame.loc[frame["date"] > last_final, "final"].astype(bool).any()


def test_pending_rows_are_settled_once_data_arrives():
    histories, bench = world()
    cut = 700
    part, meta, _ = build({s: h.iloc[:cut] for s, h in histories.items()}, bench.iloc[:cut])
    open_before = part.loc[~part["final"].astype(bool)]
    extended, _, _ = build(histories, bench, part, meta)
    settled = extended.merge(open_before[["date", "symbol"]], on=["date", "symbol"])
    assert len(settled) == len(open_before) and settled["final"].astype(bool).all()


def test_restated_stock_is_replayed_in_full_and_others_are_not():
    histories, bench = world()
    part, meta, _ = build({s: h.iloc[:700] for s, h in histories.items()}, bench.iloc[:700])
    changed = {s: h.copy() for s, h in histories.items()}
    old_cut = changed["B.NS"].index[300]
    changed["B.NS"].loc[changed["B.NS"].index < old_cut, ["Open", "High", "Low", "Close"]] *= 0.9  # non-uniform restatement
    _, _, plan = build(changed, bench, part, meta)
    assert plan.restated == ["B.NS"]
    full_grid = set(part["date"]) | set(plan.new_dates)
    assert set(plan.tasks["B.NS"]) == full_grid                    # restated stock: replayed from the start
    assert set(plan.tasks["A.NS"]) < set(plan.tasks["B.NS"])         # untouched stocks: only new + open rows
    assert set(plan.tasks["A.NS"]) >= set(plan.new_dates)


def test_uniform_rescale_is_not_a_restatement():
    histories, bench = world()
    part, meta, _ = build({s: h.iloc[:700] for s, h in histories.items()}, bench.iloc[:700])
    scaled = {s: h.copy() for s, h in histories.items()}
    scaled["A.NS"][["Open", "High", "Low", "Close"]] *= 0.97   # whole series re-based: signals are scale-free
    assert build(scaled, bench, part, meta)[2].restated == []


def test_roundtrip_through_disk_format(tmp_path, monkeypatch):
    histories, bench = world(n=700)
    frame, meta, _ = build(histories, bench)
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "l.csv.gz")
    monkeypatch.setattr(ledger, "META_PATH", tmp_path / "m.json")
    ledger.save(frame, meta)
    loaded, meta2 = ledger.load()
    pd.testing.assert_frame_equal(canonical(loaded), canonical(frame), check_dtype=False)
    assert meta2 == meta
    rows = ledger.frame_to_rows(loaded)
    assert len(rows) == len(frame) and {"outcome", "features"} <= set(rows[0])
