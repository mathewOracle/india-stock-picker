from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.intraday.analysis import IntradayReport, MarketStatus, decide, market_phase
from app.intraday.checks import DailyContext, tradeability_issues
from app.intraday.indicators import floor_pivots, opening_range, relative_volume, vwap
from app.intraday.setups import PLAYBOOKS, Setup, detect_setups
from app.intraday.snapshot import build_snapshot
from app.tradeplan import TradePlan


def make_session_bars(days: list[list[float]], volume: float = 1000.0, start="2026-10-15") -> pd.DataFrame:
    """Each inner list = closes for one session of 5-minute bars starting 09:15."""
    frames = []
    for offset, closes in enumerate(days):
        day = pd.Timestamp(start) + pd.Timedelta(days=offset)
        index = pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=len(closes), freq="5min")
        closes = np.asarray(closes, dtype=float)
        opens = np.concatenate([[closes[0]], closes[:-1]])
        frames.append(pd.DataFrame({
            "Open": opens, "High": np.maximum(opens, closes) + 0.2, "Low": np.minimum(opens, closes) - 0.2,
            "Close": closes, "Volume": np.full(len(closes), volume)}, index=index))
    return pd.concat(frames)


CONTEXT = DailyContext(traded_value_cr=800, atr_pct=0.02, beta=1.0, daily_technical=65, fundamental=70,
                       earnings_in_days=None, sector_change=0.004, nifty_change=0.002)


def test_floor_pivots_textbook_values():
    pv = floor_pivots(high=110, low=90, close=100)
    assert pv.p == pytest.approx(100) and pv.r1 == pytest.approx(110) and pv.s1 == pytest.approx(90)
    assert pv.r2 == pytest.approx(120) and pv.s2 == pytest.approx(80)
    assert pv.bc <= pv.p <= pv.tc


def test_vwap_equals_typical_price_when_flat():
    bars = make_session_bars([[100.0] * 10])
    assert vwap(bars)["vwap"].iloc[-1] == pytest.approx(100.0)


def test_opening_range_forming_then_known():
    assert opening_range(make_session_bars([[100, 101]])) is None  # only 09:15-09:20 so far
    high, low = opening_range(make_session_bars([[100, 102, 99, 101, 101]]))
    assert high == pytest.approx(102.2) and low == pytest.approx(98.8)


def test_relative_volume_compares_same_time_of_day():
    quiet, busy = [[100] * 20] * 2, [100] * 10
    bars = make_session_bars(quiet + [busy])
    bars.loc[bars.index.date == bars.index[-1].date(), "Volume"] = 3000.0
    assert relative_volume(bars, bars.index[-1].date()) == pytest.approx(3.0)


def _breakout_bars(run_to: float) -> pd.DataFrame:
    prior = list(np.linspace(100, 101, 75))
    today = [101, 101.5, 101.2, 101.4] + list(np.linspace(101.6, run_to, 6))  # OR high = 101.7
    bars = make_session_bars([prior, prior, today])
    bars.loc[bars.index.date == bars.index[-1].date(), "Volume"] = 2500.0
    return bars


def test_orb_long_detected_on_fresh_breakout_with_volume():
    keys = {(s.playbook.key, s.side) for s in detect_setups(build_snapshot(_breakout_bars(101.95)))}
    assert ("orb", "Long") in keys


def test_orb_refuses_to_chase_an_extended_breakout():
    keys = {s.playbook.key for s in detect_setups(build_snapshot(_breakout_bars(104.0)))}
    assert "orb" not in keys


def test_every_setup_respects_stop_and_target_sides():
    for setup in detect_setups(build_snapshot(_breakout_bars(101.95))):
        d = 1 if setup.side == "Long" else -1
        assert (setup.plan.entry - setup.plan.stop) * d > 0
        assert (setup.plan.target1 - setup.plan.entry) * d > 0
        assert setup.plan.reward_risk <= 3.0 + 1e-9  # no fantasy R:R


def test_playbook_registry_is_complete():
    assert len(PLAYBOOKS) == 7 and all(p.rules for p in PLAYBOOKS.values())


@pytest.mark.parametrize("hhmm,phase,accepting", [
    ("08:30", "Pre-market", False), ("09:20", "Opening range forming", False),
    ("11:00", "Open: trading window", True), ("15:00", "Open: no new entries (square-off time)", False),
    ("16:00", "Closed", False)])
def test_market_phase(hhmm, phase, accepting):
    h, m = map(int, hhmm.split(":"))
    label, _, entries = market_phase(datetime(2026, 10, 19, h, m))  # a Monday
    assert label == phase and entries is accepting
    assert market_phase(datetime(2026, 10, 18, 11, 0))[0] == "Closed (weekend)"


def test_hard_filters():
    assert tradeability_issues(CONTEXT) == []
    illiquid = DailyContext(**{**CONTEXT.__dict__, "traded_value_cr": 20, "earnings_in_days": 0})
    issues = tradeability_issues(illiquid)
    assert any("Illiquid" in i for i in issues) and "Results today" in issues


def _report(setups, bias=70, issues=()):
    snap = build_snapshot(make_session_bars([[100] * 75, [100] * 75, [100] * 30]))
    return IntradayReport("X.NS", "X", "Tech", snap, CONTEXT, [], [], list(setups), list(issues),
                          scores={"technical": bias, "context": bias, "bias": bias})


OPEN = MarketStatus("Open: trading window", True, True, None, None, "")


def test_decide_explains_every_no_trade():
    assert decide(_report([], issues=["Results today"]), OPEN)[:2] == ("No trade", "Results today")
    assert decide(_report([]), OPEN)[1] == "No playbook setup active"
    late = MarketStatus("Open: no new entries (square-off time)", True, False, None, None, "")
    good = Setup(PLAYBOOKS["orb"], "Long", "test trigger", TradePlan(100, 98, 104, 106, "OR mid"))  # 2R
    poor = Setup(PLAYBOOKS["orb"], "Long", "test trigger", TradePlan(100, 98, 101, 106, "OR mid"))  # 0.5R
    assert decide(_report([good], bias=80), OPEN)[0] == "Long setup"
    assert "bias" in decide(_report([good], bias=20), OPEN)[1]          # long into bearish tape
    assert "reward/risk" in decide(_report([poor], bias=80), OPEN)[1]
    assert "14:45" in decide(_report([good], bias=80), late)[1]
    fade = Setup(PLAYBOOKS["vwap_reversion"], "Long", "t", TradePlan(100, 98, 104, 106, "x"))
    assert decide(_report([fade], bias=20), OPEN)[0] == "Long setup"    # counter-trend setups are exempt
