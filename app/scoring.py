"""Decision engine: combine the three pillar scores and apply hard gates.

A stock is only *suggested* when fundamentals, technicals AND the chart agree
and the trade has an acceptable reward/risk. Great businesses with broken
charts go to the Watchlist instead of being force-fed as "Buy".
"""

import math
from dataclasses import dataclass

from app.fundamentals import LOSS_PE
from app.technical import EXTENDED_ATR

PILLAR_WEIGHTS = {"fundamental": 0.40, "technical": 0.35, "chart": 0.25}
assert math.isclose(sum(PILLAR_WEIGHTS.values()), 1.0)

HARD_FLAGS = {"Falling knife", "Loss-making", "High leverage", "Weak Piotroski"}

# verdict -> minimum composite, fundamental, technical, chart, reward/risk
GATES = {
    "Strong Buy": {"composite": 66, "fundamental": 60, "technical": 62, "chart": 55, "reward_risk": 2.0},
    "Buy": {"composite": 57, "fundamental": 52, "technical": 52, "chart": 45, "reward_risk": 1.5},
}
WATCHLIST = {"fundamental": 72, "composite": 50}  # excellent business, chart not ready yet
HOLD_COMPOSITE = 45
VERDICTS = ["Strong Buy", "Buy", "Watchlist", "Hold", "Avoid"]
BUY_LEVELS = ("Strong Buy", "Buy")


@dataclass(frozen=True)
class Gate:
    name: str
    shown: str  # human-readable actual value, e.g. "74.7" or "none"
    passed: bool


def composite(scores: dict[str, float]) -> float:
    return sum(scores[pillar] * weight for pillar, weight in PILLAR_WEIGHTS.items())


def price_flags(m: dict) -> list[str]:
    """Warnings derivable from price alone (these are the ones the backtest can replay)."""
    out = []
    if m["pct_vs_200dma"] < 0 and m["off_high"] < -0.30 and m["st_dir"] < 0:
        out.append("Falling knife")
    if m["rsi"] > 75:
        out.append("Overbought (RSI > 75)")
    if m["pct_vs_200dma"] < 0:
        out.append("Below 200-DMA")
    if m["reward_risk"] < 1.5:
        out.append("Poor reward/risk")
    if m["extension"] > EXTENDED_ATR:
        out.append("Extended (chase risk)")
    return out


def fundamental_flags(m: dict) -> list[str]:
    out = []
    if m["pe"] >= LOSS_PE:
        out.append("Loss-making")
    if m["de"] > 2:
        out.append("High leverage")
    if m["piotroski_max"] >= 6 and m["piotroski"] / m["piotroski_max"] <= 0.34:
        out.append("Weak Piotroski")
    return out


def flags(m: dict) -> list[str]:
    """Human-readable warnings. Those in HARD_FLAGS veto any buy suggestion."""
    return price_flags(m) + fundamental_flags(m)


def check_gates(level: str, values: dict, warnings: list[str]) -> list[Gate]:
    gates = [Gate(f"{key.replace('_', '/').title()} >= {need}", f"{values[key]:.1f}", values[key] >= need)
             for key, need in GATES[level].items()]
    if level == "Strong Buy":  # top conviction never means chasing a spike
        extension = values.get("extension", float("nan"))
        gates.append(Gate(f"Not extended (< {EXTENDED_ATR:g} ATR above 20-EMA)", f"{extension:.1f} ATR",
                          not extension > EXTENDED_ATR))
    vetoes = [f for f in warnings if f in HARD_FLAGS]
    gates.append(Gate("No hard red flags", ", ".join(vetoes) or "none", not vetoes))
    return gates


def decide(values: dict, warnings: list[str]) -> tuple[str, list[Gate]]:
    """Return (verdict, gates of the level it qualified for -- or the Buy gates it failed)."""
    for level in ("Strong Buy", "Buy"):
        gates = check_gates(level, values, warnings)
        if all(g.passed for g in gates):
            return level, gates
    buy_gates = check_gates("Buy", values, warnings)
    vetoed = any(f in HARD_FLAGS for f in warnings)
    if not vetoed and all(values[key] >= need for key, need in WATCHLIST.items()):
        return "Watchlist", buy_gates
    if values["composite"] >= HOLD_COMPOSITE and not (vetoed and values["composite"] < 50):
        return "Hold", buy_gates
    return "Avoid", buy_gates
