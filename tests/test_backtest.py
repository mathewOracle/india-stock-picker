import math

import numpy as np
import pandas as pd
import pytest

from app.research import backtest as bt
from app.research.backtest import COST_PER_SIDE, MAX_HOLD, non_overlapping, replay_stock, simulate
from app.research.evidence import SHRINK_K, summarize
from app.research.features import FEATURES


def bars(opens, highs, lows, closes) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=len(opens))
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1e6}, index=index)


def flat(n, price=100.0):
    return [price] * n


# ---------- trade simulation ----------

def test_target_hit_books_target_minus_costs():
    df = bars([100, 100, 101, 104], [100, 101, 102, 111], [100, 99, 100, 103], [100, 100, 101, 110])
    out = simulate(df, 0, stop=95, target=110)
    assert out.reason == "target" and out.exit == 110
    assert out.net_return == pytest.approx(110 / 100 - 1 - 2 * COST_PER_SIDE)


def test_same_day_stop_and_target_assumes_stop():
    df = bars([100, 100], [100, 120], [100, 80], [100, 100])
    assert simulate(df, 0, stop=95, target=110).reason == "stop"


def test_gap_through_stop_fills_at_the_worse_open():
    df = bars([100, 100, 90], [100, 101, 91], [100, 99, 89], [100, 100, 90])
    out = simulate(df, 0, stop=95, target=110)
    assert out.reason == "stop (gap)" and out.exit == 90


def test_no_entry_when_open_gaps_beyond_a_level():
    df = bars([100, 94], [100, 95], [100, 93], [100, 94])
    assert simulate(df, 0, stop=95, target=110) is None


def test_time_exit_and_unknown_outcome():
    n = MAX_HOLD + 2
    df = bars(flat(n), flat(n, 101), flat(n, 99), flat(n))
    assert simulate(df, 0, stop=90, target=120).reason == "time"
    short = df.iloc[: MAX_HOLD - 5]  # data ends while the trade is still open
    assert simulate(short, 0, stop=90, target=120) is None


def test_non_overlapping_skips_signals_while_in_a_trade():
    def row(date, entry, exit_):
        return {"date": date, "symbol": "X", "buy": True,
                "outcome": {"entry_date": entry, "exit_date": exit_, "net_return": 0.01, "r_multiple": 0.5,
                            "bars": 5, "reason": "time", "entry": 1, "exit": 1}}
    rows = [row("2024-01-01", "2024-01-02", "2024-01-20"), row("2024-01-08", "2024-01-09", "2024-01-25"),
            row("2024-01-22", "2024-01-23", "2024-02-10")]
    assert [t["signal_date"] for t in non_overlapping(rows, "buy")] == ["2024-01-01", "2024-01-22"]


# ---------- the no-lookahead proof ----------

def random_walk(n=700, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    opens = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(opens, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(opens, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"Open": opens, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(5e5, 2e6, n).astype(float)}, index=pd.bdate_range("2020-01-01", periods=n))


def test_replay_is_identical_with_future_data_removed():
    history = random_walk()
    bench = random_walk(seed=11)["Close"]
    # 265 and 290 sit between WARMUP (260) and WINDOW (300): the slice-start edge case that once
    # wrapped around to the END of the series (= future bars) -- keep them in this test forever
    dates = [history.index[i] for i in (265, 290, 300, 420, 555)]
    full = replay_stock("X", history, bench, dates)
    assert len(full) == len(dates)
    for date, row in zip(dates, full):
        cut = replay_stock("X", history.loc[:date], bench.loc[:date], [date])[0]
        assert (row["buy"], row["strong"]) == (cut["buy"], cut["strong"])
        for name in FEATURES:
            a, b = row["features"][name], cut["features"][name]
            assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-12), name
        assert cut["outcome"] is None and math.isnan(cut["fwd"])  # the future really was unavailable


def test_every_row_has_exactly_the_pinned_features():
    rows = replay_stock("X", random_walk(), random_walk(seed=3)["Close"], list(random_walk().index[::25]))
    assert rows and all(set(r["features"]) == set(FEATURES) for r in rows)


# ---------- evidence ----------

def _rows(symbol, signal_returns, control_returns):
    rows = []
    day = pd.Timestamp("2020-01-01")
    for k, ret in enumerate(signal_returns + control_returns):
        entry = day + pd.Timedelta(days=10 * k + 1)
        rows.append({"date": (entry - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "symbol": symbol,
                     "buy": k < len(signal_returns), "strong": False,
                     "outcome": {"entry_date": entry.strftime("%Y-%m-%d"),
                                 "exit_date": (entry + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                                 "net_return": ret, "r_multiple": ret * 20, "bars": 5, "reason": "time",
                                 "entry": 1, "exit": 1}})
    return rows


def test_universe_passes_only_with_a_significant_edge():
    rng = np.random.default_rng(0)
    good = {f"S{i}": _rows(f"S{i}", list(rng.normal(0.03, 0.02, 10)), list(rng.normal(0.0, 0.02, 30))) for i in range(5)}
    assert summarize(good)["passed"] is True
    noise = {f"S{i}": _rows(f"S{i}", list(rng.normal(0.0, 0.02, 10)), list(rng.normal(0.0, 0.02, 30))) for i in range(5)}
    assert summarize(noise)["universe"]["buy"]["t_stat"] < 1.65 or summarize(noise)["passed"] is False


def test_single_stock_is_shrunk_toward_the_universe():
    rng = np.random.default_rng(1)
    rows = {f"S{i}": _rows(f"S{i}", list(rng.normal(0.03, 0.01, 10)), list(rng.normal(0, 0.01, 30))) for i in range(4)}
    rows["LUCKY"] = _rows("LUCKY", [0.20, 0.25], list(rng.normal(0, 0.01, 30)))  # 2 huge wins
    lucky = summarize(rows)["stocks"]["LUCKY"]
    assert lucky["evidence"].startswith("mostly universe")
    assert lucky["shrunk_return"] < lucky["avg_return"]  # 2 trades can't outvote the prior
    expected = (2 * lucky["avg_return"] + SHRINK_K * summarize(rows)["universe"]["buy"]["avg_return"]) / (2 + SHRINK_K)
    assert lucky["shrunk_return"] == pytest.approx(expected)
