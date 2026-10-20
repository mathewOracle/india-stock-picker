from datetime import time as clock

import numpy as np
import pandas as pd
import pytest

from app.intraday.indicators import with_intraday_indicators
from app.intraday.setups import detect_setups
from app.intraday.snapshot import assemble, build_snapshot
from app.research import intraday_backtest as ib


def session_index(day: str) -> pd.DatetimeIndex:
    return pd.date_range(f"{day} 09:15", f"{day} 15:25", freq="5min")


def make_raw(days=8, seed=0, start="2026-09-01") -> pd.DataFrame:
    """Random-walk 5m bars with a different drift each session so several setups can fire."""
    rng = np.random.default_rng(seed)
    frames, price = [], 1000.0
    for day in pd.bdate_range(start, periods=days):
        idx = session_index(day.strftime("%Y-%m-%d"))
        drift = rng.normal(0, 0.0006)
        close = price * np.cumprod(1 + drift + rng.normal(0, 0.0015, len(idx)))
        open_ = np.concatenate([[price * (1 + rng.normal(0, 0.004))], close[:-1]])
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0007, len(idx))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0007, len(idx))))
        frames.append(pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                                    "Volume": rng.lognormal(11, 0.4, len(idx))}, index=idx))
        price = close[-1]
    return pd.concat(frames)


def flat_day(price=100.0, bars=75) -> pd.DataFrame:
    idx = session_index("2026-09-02")[:bars]
    return pd.DataFrame({"Open": price, "High": price + 0.01, "Low": price - 0.01, "Close": price, "Volume": 1e5}, index=idx)


def set_bar(day, pos, **cols):
    for k, v in cols.items():
        day.iloc[pos, day.columns.get_loc(k)] = v


# ---------- trade simulation ----------

def test_long_hits_target():
    day = flat_day()
    set_bar(day, 22, High=103.0)
    t = ib.simulate(day, 20, "Long", stop=98.0, target=103.0)
    assert t["reason"] == "target" and t["exit"] == 103.0 and t["entry"] == 100.0
    assert t["net_return"] == pytest.approx(0.03 - 2 * ib.COST_PER_SIDE)
    assert t["r_multiple"] == pytest.approx(1.5 - 2 * ib.COST_PER_SIDE * 100 / 2)


def test_stop_wins_when_one_bar_touches_both_levels():
    day = flat_day()
    set_bar(day, 22, High=103.5, Low=97.5)
    t = ib.simulate(day, 20, "Long", stop=98.0, target=103.0)
    assert t["reason"] == "stop" and t["exit"] == 98.0


def test_gap_through_the_stop_fills_at_the_open_not_the_stop():
    day = flat_day()
    set_bar(day, 23, Open=96.0, Low=95.5, High=96.5, Close=96.0)
    t = ib.simulate(day, 20, "Long", stop=98.0, target=105.0)
    assert t["reason"] == "stop (gap)" and t["exit"] == 96.0


def test_short_side_mirrors_the_long_side():
    day = flat_day()
    set_bar(day, 22, Low=96.8)
    t = ib.simulate(day, 20, "Short", stop=102.0, target=97.0)
    assert t["reason"] == "target" and t["net_return"] == pytest.approx(0.03 - 2 * ib.COST_PER_SIDE)
    day2 = flat_day()
    set_bar(day2, 22, High=102.5)
    assert ib.simulate(day2, 20, "Short", stop=102.0, target=97.0)["exit"] == 102.0


def test_unfilled_trade_is_squared_off_at_1515_close():
    day = flat_day()
    set_bar(day, 71, Close=100.4)                                 # bar starting 15:10 closes the day's trade
    t = ib.simulate(day, 20, "Long", stop=98.0, target=105.0)
    assert t["reason"] == "square-off" and t["exit"] == pytest.approx(100.4)


def test_entry_bar_uses_the_open_as_entry_and_only_its_range_for_exits():
    day = flat_day()
    set_bar(day, 21, Open=100.5, Low=97.0)                        # next bar: opens 100.5, then trades through the stop
    t = ib.simulate(day, 20, "Long", stop=98.0, target=105.0)
    assert t["entry"] == 100.5 and t["reason"] == "stop"


def test_no_trade_when_next_open_already_beyond_a_level():
    day = flat_day()
    set_bar(day, 21, Open=104.0, High=104.2, Low=103.9, Close=104.0)
    assert ib.simulate(day, 20, "Long", stop=98.0, target=103.0) is None
    assert ib.simulate(flat_day(), 20, "Long", stop=101.0, target=103.0) is None   # stop above entry = nonsense


def test_incomplete_session_or_no_next_bar_gives_no_trade():
    assert ib.simulate(flat_day(bars=60), 20, "Long", stop=98.0, target=103.0) is None   # day ends 14:10
    assert ib.simulate(flat_day(), 71, "Long", stop=98.0, target=103.0) is None            # nothing after the signal


def test_decisions_are_every_15_minutes_from_0930_to_1445():
    times = [flat_day().index[p].time() for p in ib.decision_positions(flat_day())]
    assert times[0] == clock(9, 25) and times[-1] == clock(14, 40)                          # bar OPENS; they close 09:30 / 14:45
    assert len(times) == 22 and all((t.hour * 60 + t.minute - 565) % 15 == 0 for t in times)


# ---------- parity with live + no lookahead ----------

def test_prefix_of_precomputed_indicators_equals_recomputing_on_truncated_data():
    raw = make_raw()
    window = raw[raw.index.normalize() >= raw.index.normalize().unique()[-5]]
    bars = with_intraday_indicators(window)
    today = window.index.normalize().unique()[-1]
    first_today = int(np.argmax(window.index.normalize() == today))
    for pos in (first_today + 3, first_today + 25, first_today + 50):
        fast = assemble(window.iloc[: pos + 1], bars.iloc[: pos + 1])
        slow = build_snapshot(window.iloc[: pos + 1])
        for field in ("price", "atr", "rvol", "prev_close", "day_high", "day_low", "gap", "above_vwap_share"):
            assert getattr(fast, field) == pytest.approx(getattr(slow, field), rel=1e-9), field
        assert fast.opening_range == slow.opening_range
        for col in ("vwap", "rsi", "ema20", "supertrend"):
            if col in fast.bars:
                assert fast.last[col] == pytest.approx(slow.last[col], rel=1e-9, nan_ok=True), col
        key = lambda snap: [(s.playbook.key, s.side, round(s.plan.stop, 6), round(s.plan.target1, 6)) for s in detect_setups(snap)]
        assert key(fast) == key(slow)


def trades_for(raw, **kw):
    return ib.replay_symbol("TEST.NS", raw, **kw)


def test_changing_the_future_never_changes_earlier_trades():
    raw = make_raw(days=9, seed=3)
    last_day = raw.index.normalize().unique()[-1]
    altered = raw.copy()
    mask = altered.index.normalize() == last_day
    altered.loc[mask, ["Open", "High", "Low", "Close"]] *= 1.2          # a violently different final session
    before = [t for t in trades_for(raw)["trades"] if t["date"] < last_day.strftime("%Y-%m-%d")]
    after = [t for t in trades_for(altered)["trades"] if t["date"] < last_day.strftime("%Y-%m-%d")]
    assert before == after and before                                   # identical, and the test is not vacuous


def test_incremental_replay_equals_full_replay():
    raw = make_raw(days=10, seed=4)
    full = trades_for(raw)
    days = sorted(set(raw.index.date))
    half = trades_for(raw[raw.index.normalize() <= pd.Timestamp(days[5])])
    rest = trades_for(raw, only_after=half["through"])
    assert half["trades"] + rest["trades"] == full["trades"]
    assert rest["through"] == full["through"] == days[-1].isoformat()


def test_an_unfinished_session_is_not_replayed():
    raw = make_raw(days=6)
    cut = raw.iloc[:-30]                                                # today's session ends ~12:55
    assert trades_for(cut)["through"] == sorted(set(raw.index.date))[-2].isoformat()


def test_one_trade_per_stock_session_setup_side():
    raw = make_raw(days=10, seed=5)
    keys = [(t["date"], t["setup"], t["side"]) for t in trades_for(raw)["trades"]]
    assert len(keys) == len(set(keys))


# ---------- evidence ----------

def fake_trades(n_days, per_day, mean, sd=0.01, seed=0, setup="orb", side="Long") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-05-01", periods=n_days)]
    rows = [{"symbol": f"S{i}", "date": d, "decision": "10:00", "setup": setup, "side": side, "entry": 100, "exit": 100,
             "reason": "target", "bars": 5, "net_return": rng.normal(mean, sd), "r_multiple": rng.normal(mean * 50, 1)}
            for d in dates for i in range(per_day)]
    return pd.DataFrame(rows)


def test_planted_edge_is_supported_and_noise_is_not():
    strong = ib.stats_block(fake_trades(60, 5, mean=0.004, seed=1))
    noise = ib.stats_block(fake_trades(60, 5, mean=0.0, seed=2))
    assert strong["status"] == "supported" and strong["t_stat"] > ib.bonferroni_t()
    assert noise["status"] == "not supported"


def test_too_little_data_is_insufficient_never_a_pass():
    assert ib.stats_block(fake_trades(10, 5, mean=0.05))["status"] == "insufficient data"      # 50 trades, 10 days
    assert ib.stats_block(fake_trades(60, 1, mean=0.05))["status"] == "insufficient data"      # 60 trades < 100
    assert ib.stats_block(pd.DataFrame(columns=ib.COLUMNS))["status"] == "insufficient data"


def test_a_edge_in_only_one_half_is_not_supported():
    a = fake_trades(30, 4, mean=0.02, seed=3)
    b = fake_trades(30, 4, mean=-0.004, seed=4).assign(date=lambda d: d["date"].map(lambda x: (pd.Timestamp(x) + pd.Timedelta(days=100)).strftime("%Y-%m-%d")))
    assert ib.stats_block(pd.concat([a, b]))["status"] != "supported"


def test_trades_on_the_same_day_are_clustered_so_t_is_not_inflated():
    rng = np.random.default_rng(7)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-05-01", periods=40)]
    market = {d: rng.normal(0.0, 0.01) for d in dates}                         # a shared market move each day
    rows = [{"date": d, "net_return": market[d] + rng.normal(0, 0.002), "r_multiple": 0.0, "reason": "x"} for d in dates for _ in range(10)]
    df = pd.DataFrame(rows)
    clustered, days = ib.day_clustered_t(df)
    naive = df["net_return"].mean() / (df["net_return"].std() / np.sqrt(len(df)))
    assert days == 40 and abs(clustered) < abs(naive) / 2                       # naive pretends 400 independent trades


def test_summary_covers_every_setup_on_both_sides():
    out = ib.summarize(fake_trades(40, 3, mean=0.0))
    assert len(out["setups"]) == len(ib.PLAYBOOKS) * 2 and out["bonferroni_t"] == pytest.approx(2.69, abs=0.01)
    assert out["setups"]["orb:Long"]["n"] == 120 and out["setups"]["orb:Short"]["status"] == "insufficient data"
