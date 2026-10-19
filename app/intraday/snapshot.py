"""One immutable view of 'where is this stock right now, intraday' shared by checks and setups."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.intraday.indicators import (Pivots, floor_pivots, opening_range, relative_volume, session,
                                     sessions, with_intraday_indicators)


@dataclass(frozen=True)
class Snapshot:
    bars: pd.DataFrame          # every 5m bar (multi-session) with indicators
    day: pd.DataFrame           # the latest session only
    session_date: date
    pivots: Pivots              # from the previous session -> today's levels
    next_pivots: Pivots         # from the latest session -> next session's levels
    opening_range: tuple[float, float] | None
    rvol: float
    prev_close: float
    pdh: float                  # previous day high / low
    pdl: float

    @property
    def last(self) -> pd.Series:
        return self.day.iloc[-1]

    @property
    def price(self) -> float:
        return float(self.last.Close)

    @property
    def atr(self) -> float:
        return float(self.last.atr)

    @property
    def open(self) -> float:
        return float(self.day["Open"].iloc[0])

    @property
    def day_high(self) -> float:
        return float(self.day["High"].max())

    @property
    def day_low(self) -> float:
        return float(self.day["Low"].min())

    @property
    def change(self) -> float:
        return self.price / self.prev_close - 1

    @property
    def gap(self) -> float:
        return self.open / self.prev_close - 1

    @property
    def above_vwap_share(self) -> float:
        """Fraction of today's bars that closed above VWAP: 1.0 = trend day up, 0.0 = trend day down."""
        valid = self.day.dropna(subset=["vwap"])
        return float((valid["Close"] > valid["vwap"]).mean()) if not valid.empty else 0.5


def build_snapshot(raw: pd.DataFrame) -> Snapshot | None:
    """None when there is not at least one full prior session to anchor levels on."""
    days = sessions(raw)
    if len(days) < 2:
        return None
    bars = with_intraday_indicators(raw)
    today, yesterday = days[-1], session(raw, days[-2])
    day = session(bars, today)
    return Snapshot(
        bars=bars, day=day, session_date=today,
        pivots=floor_pivots(yesterday["High"].max(), yesterday["Low"].min(), yesterday["Close"].iloc[-1]),
        next_pivots=floor_pivots(day["High"].max(), day["Low"].min(), day["Close"].iloc[-1]),
        opening_range=opening_range(day), rvol=relative_volume(raw, today),
        prev_close=float(yesterday["Close"].iloc[-1]),
        pdh=float(yesterday["High"].max()), pdl=float(yesterday["Low"].min()),
    )
