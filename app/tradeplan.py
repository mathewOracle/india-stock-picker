"""Turn levels + volatility into a concrete entry / stop / target plan (long or short)."""

from dataclasses import dataclass

MAX_T1_R, MAX_T2_R = 3, 5  # beyond these, a level is too far away to be a realistic target


@dataclass(frozen=True)
class TradePlan:
    entry: float
    stop: float
    target1: float
    target2: float
    stop_basis: str

    @property
    def direction(self) -> int:
        return 1 if self.stop < self.entry else -1

    @property
    def risk_pct(self) -> float:
        return abs(self.entry - self.stop) / self.entry

    @property
    def reward_risk(self) -> float:
        return (self.target1 - self.entry) / (self.entry - self.stop)  # sign cancels for shorts


def pick_targets(entry: float, risk: float, levels: list[float], direction: int = 1) -> tuple[float, float]:
    """T1 = nearest level ahead within 3R (else 2R); T2 = next level within 5R (else 3R / T1 + 1R)."""
    ahead = sorted((lv for lv in levels if (lv - entry) * direction > 0), key=lambda lv: abs(lv - entry))
    reach1 = [lv for lv in ahead if abs(lv - entry) <= MAX_T1_R * risk]
    target1 = reach1[0] if reach1 else entry + direction * 2 * risk
    reach2 = [lv for lv in ahead if (lv - target1) * direction > 0 and abs(lv - entry) <= MAX_T2_R * risk]
    fallback = entry + direction * max(abs(target1 - entry) + risk, 3 * risk)
    return target1, (reach2[0] if reach2 else fallback)


def make_plan(price: float, atr: float, support: float | None, resistances: list[float]) -> TradePlan:
    """Swing (long) plan: stop just under nearby support (else 2x ATR); targets = reachable resistances."""
    if support is not None and price - support <= 3 * atr:
        stop, basis = support - 0.5 * atr, "0.5 ATR below support"
    else:
        stop, basis = price - 2 * atr, "2x ATR below price"
    stop = min(stop, price - atr)  # always leave at least 1 ATR of breathing room
    target1, target2 = pick_targets(price, price - stop, resistances)
    return TradePlan(price, stop, target1, target2, basis)
