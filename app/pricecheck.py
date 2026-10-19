"""Price-only assessment (technical + chart + trade plan) shared by the live report AND the backtest.

One code path on purpose: what the backtest replays is exactly what the dashboard shows,
so the two can never drift apart.
"""

from dataclasses import dataclass

import pandas as pd

from app.patterns import ChartReport, analyze_chart
from app.scoring import GATES, HARD_FLAGS, PILLAR_WEIGHTS, price_flags
from app.signals import Signal, score
from app.technical import EXTENDED_ATR, analyze_technical, extension_atr, price_metrics
from app.tradeplan import TradePlan, make_plan

_PRICE_TOTAL = PILLAR_WEIGHTS["technical"] + PILLAR_WEIGHTS["chart"]
PRICE_WEIGHTS = {"technical": PILLAR_WEIGHTS["technical"] / _PRICE_TOTAL, "chart": PILLAR_WEIGHTS["chart"] / _PRICE_TOTAL}


@dataclass(frozen=True)
class PriceAssessment:
    technical: list[Signal]
    chart: ChartReport
    plan: TradePlan
    scores: dict[str, float]   # technical, chart, price (re-weighted technical+chart composite)
    metrics: dict
    flags: list[str]


def assess_price(ind: pd.DataFrame, bench: pd.Series | None) -> PriceAssessment:
    """`ind` must end at the evaluation date; `bench` must not extend beyond it (no lookahead)."""
    last = ind.iloc[-1]
    chart = analyze_chart(ind)
    plan = make_plan(float(last.Close), float(last.atr), chart.support, chart.resistances)
    technical = analyze_technical(ind, bench)
    scores = {"technical": score(technical), "chart": score(chart.signals)}
    scores["price"] = sum(scores[k] * w for k, w in PRICE_WEIGHTS.items())
    metrics = {**price_metrics(ind["Close"]), "rsi": float(last.rsi), "st_dir": float(last.st_dir),
               "atr": float(last.atr), "extension": extension_atr(ind), "reward_risk": plan.reward_risk}
    return PriceAssessment(technical, chart, plan, scores, metrics, price_flags(metrics))


def price_signal(pa: PriceAssessment, level: str = "Buy") -> bool:
    """The live gates for `level` minus the fundamental gate -- the rule the backtest tests."""
    gate = GATES[level]
    passed = (pa.scores["technical"] >= gate["technical"] and pa.scores["chart"] >= gate["chart"]
              and pa.scores["price"] >= gate["composite"] and pa.plan.reward_risk >= gate["reward_risk"]
              and not any(f in HARD_FLAGS for f in pa.flags))
    if level == "Strong Buy":
        passed = passed and not pa.metrics["extension"] > EXTENDED_ATR
    return passed
