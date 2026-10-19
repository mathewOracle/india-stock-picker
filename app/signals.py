"""Shared vocabulary for all three analysis pillars (technical, chart, fundamental).

Every check emits a Signal with points in [-1, +1] and a weight. A pillar's
score is simply 50 +/- 50 * (weighted average of points), so 50 = neutral,
100 = every check maximally bullish, 0 = every check maximally bearish.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    name: str
    reading: str   # what was measured, e.g. "RSI 62.4"
    points: float  # -1 (very bearish) .. +1 (very bullish)
    weight: float
    note: str      # plain-English interpretation

    @property
    def tone(self) -> str:
        if self.points > 0.15:
            return "bullish"
        return "bearish" if self.points < -0.15 else "neutral"

    @property
    def impact(self) -> float:
        return self.points * self.weight


def score(signals: list[Signal]) -> float:
    total_weight = sum(s.weight for s in signals)
    if not total_weight:
        return 50.0
    return 50 + 50 * sum(s.impact for s in signals) / total_weight


def best(signals: list[Signal]) -> Signal | None:
    positives = [s for s in signals if s.impact > 0]
    return max(positives, key=lambda s: s.impact) if positives else None


def worst(signals: list[Signal]) -> Signal | None:
    negatives = [s for s in signals if s.impact < 0]
    return min(negatives, key=lambda s: s.impact) if negatives else None


def grade(value: float, bands: list[tuple[float, float]], below: float) -> float:
    """Map a value to points via descending (threshold, points) bands.

    grade(0.22, [(0.20, 1), (0.15, 0.5)], below=-1) -> 1
    """
    for threshold, points in bands:
        if value >= threshold:
            return points
    return below
