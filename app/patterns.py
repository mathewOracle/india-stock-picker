"""Chart-analysis pillar: price structure and patterns, the stuff you'd eyeball on a chart.

Swing pivots -> support/resistance levels, trend structure (HH/HL), regression
channel, 52-week breakouts, Bollinger squeeze, pullback setups, double
tops/bottoms, candlestick patterns and RSI divergences.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.signals import Signal

PIVOT_WINDOW = 5        # minor swings -> support/resistance levels
MAJOR_PIVOT_WINDOW = 10  # major swings -> trend structure & double tops/bottoms
LEVEL_TOLERANCE = 0.015  # pivots within 1.5% of each other are "the same level"


@dataclass(frozen=True)
class Level:
    price: float
    touches: int
    label: str = ""  # e.g. "52w high" for levels that matter regardless of touch count


@dataclass
class ChartReport:
    signals: list[Signal]
    levels: list[Level] = field(default_factory=list)
    support: float | None = None
    resistances: list[float] = field(default_factory=list)

    @property
    def resistance(self) -> float | None:
        return self.resistances[0] if self.resistances else None


# ---------- structure helpers ----------

def pivots(df: pd.DataFrame, k: int = PIVOT_WINDOW) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Swing highs/lows: bars that are the extreme of the surrounding 2k+1 window."""
    highs, lows = df["High"].to_numpy(), df["Low"].to_numpy()
    swing_highs, swing_lows = [], []
    for i in range(k, len(df) - k):
        if highs[i] == highs[i - k:i + k + 1].max():
            swing_highs.append((i, float(highs[i])))
        if lows[i] == lows[i - k:i + k + 1].min():
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


def cluster_levels(prices: list[float], tolerance: float = LEVEL_TOLERANCE) -> list[Level]:
    levels, group = [], []
    for price in sorted(prices):
        if group and price > group[0] * (1 + tolerance):
            levels.append(Level(float(np.mean(group)), len(group)))
            group = []
        group.append(price)
    if group:
        levels.append(Level(float(np.mean(group)), len(group)))
    return levels


def with_extremes(levels: list[Level], df: pd.DataFrame) -> list[Level]:
    """The 52-week high/low are the most-watched levels on any chart, even with a single touch."""
    out = list(levels)
    for price, label in ((float(df["High"].max()), "52w high"), (float(df["Low"].min()), "52w low")):
        if not any(abs(lv.price / price - 1) < LEVEL_TOLERANCE and lv.touches >= 2 for lv in out):
            out.append(Level(price, 2, label))
    return sorted(out, key=lambda lv: lv.price)


def _strong(levels: list[Level]) -> list[float]:
    return [lv.price for lv in levels if lv.touches >= 2]


def significant_levels(levels: list[Level], price: float, atr: float) -> tuple[float | None, list[float]]:
    """Support at least 0.5 ATR below, resistances at least 1 ATR above (closer ones are noise)."""
    strong = _strong(levels)
    below = [p for p in strong if p <= price - 0.5 * atr]
    above = sorted(p for p in strong if p >= price + atr)
    return (max(below) if below else None), above


# ---------- checks ----------

def _structure(highs, lows, close: float) -> Signal | None:
    if len(highs) < 2 or len(lows) < 2:
        return None
    hh, hl = highs[-1][1] > highs[-2][1], lows[-1][1] > lows[-2][1]
    pts, label, note = {
        (True, True): (1.0, "Higher highs & higher lows", "Classic uptrend structure (Dow theory)."),
        (False, False): (-1.0, "Lower highs & lower lows", "Classic downtrend structure."),
        (True, False): (0.0, "Expanding swings", "Volatile, broadening range."),
        (False, True): (0.2, "Contracting swings", "Coiling triangle: breakout pending."),
    }[(hh, hl)]
    # the latest price can already be breaking the pattern the confirmed pivots describe
    if pts < 1 and close > highs[-1][1]:
        return Signal("Swing structure", f"{label}, now above last swing high", max(pts + 0.8, 0.4), 2.0,
                      "Price cleared the last major swing high: structure turning up.")
    if pts > -1 and close < lows[-1][1]:
        return Signal("Swing structure", f"{label}, now below last swing low", min(pts - 0.8, -0.4), 2.0,
                      "Price lost the last major swing low: structure turning down.")
    return Signal("Swing structure", label, pts, 2.0, note)


def _channel(close: pd.Series, days: int = 60) -> Signal | None:
    if len(close) < days:
        return None
    y = np.log(close.tail(days).to_numpy())
    x = np.arange(days)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    annual = float(np.exp(slope * 252) - 1)
    reading = f"{annual:+.0%}/yr pace, R\u00b2 {r2:.2f}"
    if r2 < 0.5:
        return Signal("60-day trend channel", reading, 0.0, 1.0, "Choppy: no clean channel.")
    up = slope > 0
    return Signal("60-day trend channel", reading, 1.0 if up else -1.0, 1.0,
                  f"Clean {'rising' if up else 'falling'} channel.")


def _breakout(df: pd.DataFrame) -> Signal | None:
    if len(df) < 120:
        return None
    close = df["Close"].iloc[-1]
    prior_high = df["High"].iloc[-252:-5].max()
    year_high = df["High"].iloc[-252:].max()
    vol_ratio = df["Volume"].tail(5).mean() / df["Volume"].iloc[-55:-5].mean()
    if close >= prior_high:
        confirmed = vol_ratio >= 1.3
        return Signal("52-week breakout", f"new high, volume {vol_ratio:.1f}x", 1.0 if confirmed else 0.6, 1.5,
                      "Breakout to new 52-week highs" + (" on strong volume." if confirmed else ", volume not confirming."))
    gap = close / year_high - 1
    reading = f"{gap:.1%} from 52w intraday high"
    if gap >= -0.03:
        return Signal("52-week breakout", reading, 0.4, 1.5, "Knocking on the 52-week high.")
    if gap <= -0.25:
        return Signal("52-week breakout", reading, -0.6, 1.5, "Deep below its 52-week high: damaged chart.")
    return Signal("52-week breakout", reading, 0.0, 1.5, "Mid-range, no breakout yet.")


def _support_resistance(df, atr_now, levels) -> Signal:
    close = df["Close"].iloc[-1]
    close_10 = df["Close"].iloc[-11]
    broken = [lv for lv in levels if lv.touches >= 3 and close < lv.price < close_10]
    if broken:
        return Signal("Support / resistance", f"broke {broken[-1].price:,.1f}", -0.8, 1.0,
                      f"Broke a {broken[-1].touches}-touch support recently: former floor is now a ceiling.")
    strong = [lv for lv in levels if lv.touches >= 2]
    support = max((lv for lv in strong if lv.price < close), default=None, key=lambda lv: lv.price)
    resistance = min((lv for lv in strong if lv.price > close), default=None, key=lambda lv: lv.price)
    if support and (close - support.price) <= 1.5 * atr_now:
        return Signal("Support / resistance", f"{support.label or 'support'} {support.price:,.1f}", 0.7, 1.0,
                      "Sitting just above support: a tight stop is possible (good risk/reward).")
    if resistance and (resistance.price - close) <= 1.0 * atr_now:
        return Signal("Support / resistance", f"{resistance.label or 'resistance'} {resistance.price:,.1f}", -0.4, 1.0,
                      "Right under resistance: wait for a decisive breakout above it.")
    return Signal("Support / resistance", "between levels", 0.0, 1.0, "Trading between support and resistance.")


def _squeeze(ind: pd.DataFrame) -> Signal | None:
    width = ind["bandwidth"].dropna().tail(126)
    if len(width) < 60 or width.iloc[-1] > width.quantile(0.10):
        return None
    up = ind["Close"].iloc[-1] > ind["sma50"].iloc[-1]
    return Signal("Bollinger squeeze", f"bandwidth {width.iloc[-1]:.3f} (6-mo low)", 0.5 if up else -0.5, 0.75,
                  "Volatility is compressed: a big move is brewing, "
                  + ("bias up (above 50-DMA)." if up else "bias down (below 50-DMA)."))


def _pullback(ind: pd.DataFrame) -> Signal | None:
    last = ind.iloc[-1]
    sma200_rising = last.sma200 > ind["sma200"].iloc[-21]
    near_high = last.Close >= 0.85 * ind["High"].tail(252).max()
    in_uptrend = last.Close > last.sma200 and last.sma50 > last.sma200 and sma200_rising and near_high
    near_support_ma = abs(last.Close / last.sma50 - 1) < 0.03 or ind["Low"].tail(5).min() <= ind["ema20"].iloc[-1]
    if in_uptrend and near_support_ma and 38 <= last.rsi <= 55:
        return Signal("Pullback setup", f"RSI {last.rsi:.0f} near 20-EMA/50-DMA", 1.0, 1.5,
                      "Healthy pullback inside an uptrend: classic buy-the-dip entry.")
    return None


def _double_pattern(df, highs, lows) -> Signal | None:
    recent = len(df) - 120
    close = df["Close"].iloc[-1]
    lows_r, highs_r = [p for p in lows if p[0] >= recent], [p for p in highs if p[0] >= recent]
    if len(lows_r) >= 2:
        (i1, l1), (i2, l2) = lows_r[-2], lows_r[-1]
        neck = df["High"].iloc[i1:i2 + 1].max()
        if i2 - i1 >= 10 and abs(l2 / l1 - 1) < 0.03 and neck >= 1.05 * max(l1, l2):
            if close > neck:
                return Signal("Double bottom", f"neckline {neck:,.1f} cleared", 1.0, 1.5, "Confirmed W-bottom reversal.")
            if close > l2:
                return Signal("Double bottom", f"lows ~{l2:,.1f}, neckline {neck:,.1f}", 0.3, 1.5, "Possible W-bottom forming.")
    if len(highs_r) >= 2:
        (i1, h1), (i2, h2) = highs_r[-2], highs_r[-1]
        trough = df["Low"].iloc[i1:i2 + 1].min()
        if i2 - i1 >= 10 and abs(h2 / h1 - 1) < 0.03 and trough <= 0.95 * min(h1, h2):
            if close < trough:
                return Signal("Double top", f"neckline {trough:,.1f} broken", -1.0, 1.5, "Confirmed M-top reversal.")
            if close < h2:
                return Signal("Double top", f"highs ~{h2:,.1f}", -0.3, 1.5, "Possible M-top forming.")
    return None


def _candles(df: pd.DataFrame) -> Signal | None:
    """Most recent meaningful candlestick pattern in the last 3 sessions."""
    for back in range(1, 4):
        o, h, l, c = df[["Open", "High", "Low", "Close"]].iloc[-back]
        po, pc = df["Open"].iloc[-back - 1], df["Close"].iloc[-back - 1]
        body, rng = abs(c - o), h - l
        if rng <= 0:
            continue
        upper, lower = h - max(o, c), min(o, c) - l
        fell = c < df["Close"].iloc[-back - 6]
        when = "today" if back == 1 else f"{back - 1} session(s) ago"
        if pc < po and c > o and o <= pc and c >= po and fell:
            return Signal("Candlestick", f"Bullish engulfing ({when})", 1.0, 1.0, "Buyers overwhelmed sellers after a decline.")
        if pc > po and c < o and o >= pc and c <= po and not fell:
            return Signal("Candlestick", f"Bearish engulfing ({when})", -1.0, 1.0, "Sellers overwhelmed buyers after a rise.")
        if body > 0 and lower >= 2 * body and upper <= 0.3 * body and fell:
            return Signal("Candlestick", f"Hammer ({when})", 0.8, 1.0, "Rejection of lower prices after a decline.")
        if body > 0 and upper >= 2 * body and lower <= 0.3 * body and not fell:
            return Signal("Candlestick", f"Shooting star ({when})", -0.8, 1.0, "Rejection of higher prices after a rise.")
        if back == 1 and body <= 0.1 * rng:
            return Signal("Candlestick", "Doji (today)", 0.0, 1.0, "Indecision candle.")
    return None


def _divergence(ind: pd.DataFrame) -> Signal | None:
    if len(ind) < 60:
        return None
    recent, prior = ind.iloc[-20:], ind.iloc[-60:-20]
    if recent["Low"].min() < prior["Low"].min() * 0.99 and recent["rsi"].min() > prior["rsi"].min() + 3:
        return Signal("RSI divergence", "price lower low, RSI higher low", 1.0, 1.0,
                      "Bullish divergence: selling pressure is fading.")
    if recent["High"].max() > prior["High"].max() * 1.01 and recent["rsi"].max() < prior["rsi"].max() - 3:
        return Signal("RSI divergence", "price higher high, RSI lower high", -1.0, 1.0,
                      "Bearish divergence: momentum is fading under the rally.")
    return None


def analyze_chart(ind: pd.DataFrame) -> ChartReport:
    window = ind.tail(252).reset_index(drop=True)
    minor_highs, minor_lows = pivots(window, PIVOT_WINDOW)
    highs, lows = pivots(window, MAJOR_PIVOT_WINDOW)
    levels = with_extremes(cluster_levels([p for _, p in minor_highs + minor_lows]), window)
    close = float(window["Close"].iloc[-1])
    atr_now = float(window["atr"].iloc[-1])
    support, resistances = significant_levels(levels, close, atr_now)
    candidates = [
        _structure(highs, lows, close), _channel(window["Close"]), _breakout(window),
        _support_resistance(window, atr_now, levels),
        _squeeze(window), _pullback(window), _double_pattern(window, highs, lows),
        _candles(window), _divergence(window),
    ]
    return ChartReport([s for s in candidates if s is not None], levels, support, resistances)
