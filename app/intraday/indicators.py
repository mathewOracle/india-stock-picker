"""Pure intraday indicator math on 5-minute OHLCV bars (index = IST wall-clock time).

VWAP (+/- 1 and 2 sigma bands), opening range, classic floor pivots with the
Central Pivot Range (CPR), relative volume by time of day, and gap stats.
"""

from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

from app.indicators import atr, ema, rsi, supertrend

SESSION_OPEN = time(9, 15)
OPENING_RANGE_END = time(9, 30)   # first 15 minutes
NO_NEW_ENTRIES = time(14, 45)     # brokers square off intraday (MIS) positions ~15:15-15:20
SESSION_CLOSE = time(15, 30)


@dataclass(frozen=True)
class Pivots:
    p: float
    bc: float
    tc: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float

    @property
    def cpr_width(self) -> float:
        """CPR width as a fraction of price. Narrow (< 0.25%) often precedes a trending day."""
        return (self.tc - self.bc) / self.p

    def levels(self) -> dict[str, float]:
        return {"R3": self.r3, "R2": self.r2, "R1": self.r1, "TC": self.tc, "P": self.p,
                "BC": self.bc, "S1": self.s1, "S2": self.s2, "S3": self.s3}


def floor_pivots(high: float, low: float, close: float) -> Pivots:
    """Classic floor-trader pivots + CPR from the previous session's high/low/close."""
    p = (high + low + close) / 3
    bc = (high + low) / 2
    tc = 2 * p - bc
    bc, tc = min(bc, tc), max(bc, tc)
    span = high - low
    return Pivots(p=p, bc=bc, tc=tc,
                  r1=2 * p - low, r2=p + span, r3=high + 2 * (p - low),
                  s1=2 * p - high, s2=p - span, s3=low - 2 * (high - p))


def sessions(bars: pd.DataFrame) -> list[date]:
    return sorted(set(bars.index.date))


def session(bars: pd.DataFrame, day: date) -> pd.DataFrame:
    return bars[bars.index.date == day]


def vwap(day: pd.DataFrame) -> pd.DataFrame:
    """Session VWAP with volume-weighted standard-deviation bands."""
    typical = (day["High"] + day["Low"] + day["Close"]) / 3
    volume = day["Volume"]
    cum_volume = volume.cumsum().replace(0, np.nan)
    value = (typical * volume).cumsum() / cum_volume
    variance = ((typical - value) ** 2 * volume).cumsum() / cum_volume
    std = np.sqrt(variance)
    return pd.DataFrame({"vwap": value, "vwap_u1": value + std, "vwap_l1": value - std,
                         "vwap_u2": value + 2 * std, "vwap_l2": value - 2 * std}, index=day.index)


def opening_range(day: pd.DataFrame) -> tuple[float, float] | None:
    """High/low of the first 15 minutes, or None while it is still forming."""
    first = day[day.index.time < OPENING_RANGE_END]
    if day.empty or day.index[-1].time() < OPENING_RANGE_END or first.empty:
        return None
    return float(first["High"].max()), float(first["Low"].min())


def relative_volume(bars: pd.DataFrame, day: date) -> float:
    """Today's cumulative volume vs the average of prior sessions *up to the same time of day*."""
    today = session(bars, day)
    if today.empty:
        return float("nan")
    cutoff = today.index[-1].time()
    prior = [session(bars, d) for d in sessions(bars) if d < day]
    baseline = [s[s.index.time <= cutoff]["Volume"].sum() for s in prior]
    baseline = [b for b in baseline if b > 0]
    return float(today["Volume"].sum() / np.mean(baseline)) if baseline else float("nan")


def with_intraday_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """Add rolling (cross-session) EMAs/RSI/ATR/Supertrend and per-session VWAP to 5m bars."""
    out = bars.copy()
    close = bars["Close"]
    out["ema9"], out["ema21"] = ema(close, 9), ema(close, 21)
    out["rsi"] = rsi(close)
    out["atr"] = atr(bars)
    out = pd.concat([out, supertrend(bars)], axis=1)
    vwaps = [vwap(session(bars, d)) for d in sessions(bars)]
    return out.join(pd.concat(vwaps)) if vwaps else out
