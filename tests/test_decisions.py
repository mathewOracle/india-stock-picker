import pytest

from app.scoring import composite, decide, flags
from app.tradeplan import make_plan

STRONG = {"composite": 75, "fundamental": 75, "technical": 75, "chart": 70, "reward_risk": 2.5}


def test_composite_weights():
    assert composite({"fundamental": 100, "technical": 0, "chart": 0}) == pytest.approx(40)


def test_all_gates_pass_gives_strong_buy():
    verdict, gates = decide(STRONG, [])
    assert verdict == "Strong Buy" and all(g.passed for g in gates)


def test_hard_flag_vetoes_buy():
    verdict, gates = decide(STRONG, ["Falling knife"])
    assert verdict not in ("Strong Buy", "Buy")
    assert any(not g.passed and "Falling knife" in g.shown for g in gates)


def test_great_business_with_bad_chart_goes_to_watchlist():
    verdict, gates = decide({**STRONG, "technical": 30, "chart": 30, "composite": 52}, [])
    assert verdict == "Watchlist"
    assert any(not g.passed for g in gates)


def test_poor_reward_risk_blocks_buy():
    verdict, _ = decide({**STRONG, "reward_risk": 0.8, "fundamental": 60}, [])
    assert verdict not in ("Strong Buy", "Buy")


def test_weak_everything_is_avoid():
    assert decide({"composite": 30, "fundamental": 30, "technical": 30, "chart": 30, "reward_risk": 2}, [])[0] == "Avoid"


def test_flags_detect_falling_knife():
    m = {"pct_vs_200dma": -0.1, "off_high": -0.4, "st_dir": -1, "pe": 20, "de": 0.5,
         "piotroski": 5, "piotroski_max": 9, "rsi": 40, "reward_risk": 2, "extension": 0.5}
    assert "Falling knife" in flags(m)


def test_extended_stock_is_capped_at_buy():
    """Regression: a stock that just spiked 3+ ATR above its 20-EMA must not be a Strong Buy."""
    verdict, gates = decide({**STRONG, "extension": 3.4}, ["Extended (chase risk)"])
    assert verdict == "Buy"
    assert decide({**STRONG, "extension": 1.0}, [])[0] == "Strong Buy"


def test_red_flag_gate_reads_naturally():
    _, gates = decide(STRONG, [])
    assert gates[-1].shown == "none"


def test_plan_uses_support_and_resistance():
    plan = make_plan(price=100, atr=2, support=97, resistances=[110, 120])
    assert plan.stop == pytest.approx(96)        # 0.5 ATR under support
    assert plan.target1 == 110 and plan.target2 == 120
    assert plan.reward_risk == pytest.approx(2.5)


def test_plan_blue_sky_falls_back_to_r_multiples():
    plan = make_plan(price=100, atr=2, support=None, resistances=[])
    assert plan.stop == pytest.approx(96) and plan.target1 == pytest.approx(108)
    assert plan.reward_risk == pytest.approx(2.0)


def test_plan_ignores_resistance_that_is_miles_away():
    """Regression: a 52w high 40% overhead produced a fantasy 9.5 reward/risk."""
    plan = make_plan(price=100, atr=2, support=None, resistances=[140])
    assert plan.target1 == pytest.approx(108) and plan.reward_risk == pytest.approx(2.0)
    assert plan.target2 == pytest.approx(112)
