"""Turn levels + volatility into a concrete entry / stop / target plan."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TradePlan:
    entry: float
    stop: float
    target1: float
    target2: float
    stop_basis: str

    @property
    def risk_pct(self) -> float:
        return (self.entry - self.stop) / self.entry

    @property
    def reward_risk(self) -> float:
        return (self.target1 - self.entry) / (self.entry - self.stop)


MAX_T1_R, MAX_T2_R = 3, 5  # beyond these, a "resistance" is too far away to be a realistic target


def make_plan(price: float, atr: float, support: float | None, resistances: list[float]) -> TradePlan:
    """Stop just under nearby support (else 2x ATR); targets = next significant resistances.

    No resistance within reach (blue sky, or the next level is miles away) -> 2R / 3R targets.
    """
    if support is not None and price - support <= 3 * atr:
        stop, basis = support - 0.5 * atr, "0.5 ATR below support"
    else:
        stop, basis = price - 2 * atr, "2x ATR below price"
    stop = min(stop, price - atr)  # always leave at least 1 ATR of breathing room
    risk = price - stop
    reachable_t1 = [r for r in resistances if r <= price + MAX_T1_R * risk]
    target1 = reachable_t1[0] if reachable_t1 else price + 2 * risk
    reachable_t2 = [r for r in resistances if target1 < r <= price + MAX_T2_R * risk]
    target2 = reachable_t2[0] if reachable_t2 else max(target1 + risk, price + 3 * risk)
    return TradePlan(price, stop, target1, target2, basis)
