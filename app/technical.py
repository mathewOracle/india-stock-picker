"""Technical analysis pillar: trend, momentum, volatility, volume and relative strength.

Each `_check_*` function looks at the indicator frame and returns a Signal (or
None if it has nothing to say). Add a check = add a function to CHECKS.
"""

import math

import numpy as np
import pandas as pd

from app.signals import Signal, grade

NAN = float("nan")
EXTENDED_ATR = 3.0  # more than this many ATRs above the 20-EMA = chasing


def _ok(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def _ret(close: pd.Series, days: int) -> float:
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1) if len(close) > days else NAN


def _bars_since_cross(diff: pd.Series, lookback: int) -> int | None:
    """How many bars ago did `diff` change sign (within lookback)? None if it didn't."""
    signs = np.sign(diff.dropna().tail(lookback + 1).to_numpy())
    for back in range(1, len(signs)):
        if signs[-back] != signs[-back - 1]:
            return back - 1
    return None


def beta(close: pd.Series, benchmark: pd.Series | None, days: int = 252) -> float:
    """Sensitivity to the index from aligned daily returns (1.0 = moves with Nifty)."""
    if benchmark is None:
        return NAN
    returns = pd.concat([close.pct_change(), benchmark.pct_change()], axis=1, join="inner").dropna().tail(days)
    if len(returns) < 60:
        return NAN
    stock, index = returns.iloc[:, 0], returns.iloc[:, 1]
    return float(stock.cov(index) / index.var())


def traded_value_cr(df: pd.DataFrame, days: int = 20) -> float:
    """Average daily traded value in crore rupees (1 Cr = 10 million) -- the liquidity yardstick."""
    return float((df["Close"] * df["Volume"]).tail(days).mean() / 1e7)


def price_metrics(close: pd.Series) -> dict:
    """Headline return/trend/risk numbers from a close series (stocks and indices)."""
    last = float(close.iloc[-1])
    sma200 = close.tail(200).mean() if len(close) >= 200 else NAN
    year = close.tail(252)
    daily = close.pct_change().dropna().tail(252)
    return {
        "price": last,
        "ret_1d": _ret(close, 1), "ret_1m": _ret(close, 21), "ret_3m": _ret(close, 63),
        "ret_6m": _ret(close, 126), "ret_12m": _ret(close, 252),
        "pct_vs_200dma": float(last / sma200 - 1) if _ok(sma200) else NAN,
        "off_high": float(last / year.max() - 1),
        "vol": float(daily.std() * np.sqrt(252)) if len(daily) > 20 else NAN,
        "max_dd": float((year / year.cummax() - 1).min()),
    }


# ---------- checks ----------

def _check_trend_200(ind: pd.DataFrame, _bench) -> Signal | None:
    last = ind.iloc[-1]
    if not _ok(last.sma200):
        return None
    gap = last.Close / last.sma200 - 1
    pts = grade(gap, [(0.05, 1), (0, 0.5), (-0.05, -0.5)], below=-1)
    note = ("Long-term uptrend: price holds above its 200-day average." if gap > 0
            else "Long-term downtrend: price is below its 200-day average.")
    return Signal("200-DMA trend", f"{gap:+.1%} vs 200-DMA", pts, 2.0, note)


def _check_ma_alignment(ind, _bench) -> Signal | None:
    last = ind.iloc[-1]
    if not _ok(last.ema20, last.sma50, last.sma200):
        return None
    stacked = [last.Close > last.ema20, last.ema20 > last.sma50, last.sma50 > last.sma200]
    count = sum(stacked)
    notes = {3: "Perfect bullish stack: Price > 20-EMA > 50-DMA > 200-DMA.",
             0: "Fully bearish stack: Price < 20-EMA < 50-DMA < 200-DMA."}
    return Signal("Moving-average stack", f"{count}/3 bullish", (count - 1.5) / 1.5, 1.5,
                  notes.get(count, "Mixed moving averages: trend is transitioning."))


def _check_cross(ind, _bench) -> Signal | None:
    diff = ind["sma50"] - ind["sma200"]
    if not _ok(diff.iloc[-1]):
        return None
    golden = diff.iloc[-1] > 0
    recent = _bars_since_cross(diff, 20)
    label = "Golden cross" if golden else "Death cross"
    when = f" ({recent} bars ago - fresh!)" if recent is not None else ""
    pts = (1.0 if recent is not None else 0.6) * (1 if golden else -1)
    return Signal("50/200 cross", label + when, pts, 1.0,
                  "50-DMA above 200-DMA." if golden else "50-DMA below 200-DMA.")


def _check_adx(ind, _bench) -> Signal | None:
    last = ind.iloc[-1]
    if not _ok(last.adx, last.plus_di, last.minus_di):
        return None
    bullish = last.plus_di > last.minus_di
    reading = f"ADX {last.adx:.0f}, +DI {last.plus_di:.0f} / -DI {last.minus_di:.0f}"
    if last.adx < 20:
        return Signal("ADX trend strength", reading, 0.0, 1.0, "No real trend (ADX < 20): range-bound.")
    strength = 1.0 if last.adx >= 25 else 0.5
    return Signal("ADX trend strength", reading, strength * (1 if bullish else -1), 1.0,
                  f"{'Strong' if strength == 1 else 'Developing'} {'up' if bullish else 'down'}trend.")


def _check_supertrend(ind, _bench) -> Signal:
    up = ind["st_dir"].iloc[-1] > 0
    flip = _bars_since_cross(ind["st_dir"], 10)
    when = f", flipped {flip} bars ago" if flip is not None else ""
    return Signal("Supertrend (10,3)", f"{'BUY' if up else 'SELL'} @ {ind['st_line'].iloc[-1]:,.1f}{when}",
                  1.0 if up else -1.0, 1.5,
                  "Price above the Supertrend line: trend-followers are long." if up
                  else "Price below the Supertrend line: trend-followers are out.")


def _check_rsi(ind, _bench) -> Signal | None:
    value = ind["rsi"].iloc[-1]
    if not _ok(value):
        return None
    if value > 75:
        pts, note = -0.5, "Overbought: rally is stretched, pullback risk is high."
    elif value >= 55:
        pts, note = 1.0, "Bullish momentum zone (55-75)."
    elif value >= 45:
        pts, note = 0.0, "Neutral momentum."
    elif value >= 30:
        pts, note = -0.7, "Bearish momentum zone (30-45)."
    else:
        pts, note = 0.2, "Oversold: bounce possible, but trend is weak."
    return Signal("RSI (14)", f"{value:.1f}", pts, 1.5, note)


def _check_macd(ind, _bench) -> Signal | None:
    hist = ind["macd_hist"]
    if not _ok(hist.iloc[-1], hist.iloc[-2]):
        return None
    above, rising = hist.iloc[-1] > 0, hist.iloc[-1] > hist.iloc[-2]
    cross = _bars_since_cross(hist, 5)
    pts = {(True, True): 1.0, (True, False): 0.4, (False, True): -0.3, (False, False): -1.0}[(above, rising)]
    note = f"{'Bullish' if above else 'Bearish'} MACD, histogram {'rising' if rising else 'falling'}."
    if cross is not None:
        note += f" Fresh {'bullish' if above else 'bearish'} crossover {cross} bars ago."
    return Signal("MACD (12,26,9)", f"hist {hist.iloc[-1]:+.2f}", pts, 1.5, note)


def _check_stochastic(ind, _bench) -> Signal | None:
    k, d, k_prev = ind["stoch_k"].iloc[-1], ind["stoch_d"].iloc[-1], ind["stoch_k"].iloc[-3]
    if not _ok(k, d, k_prev):
        return None
    if k > d and k_prev < 20:
        pts, note = 1.0, "Bullish turn up from oversold."
    elif k > 80:
        pts, note = -0.5, "Overbought territory."
    else:
        pts, note = (0.3, "%K above %D.") if k > d else (-0.3, "%K below %D.")
    return Signal("Stochastic (14,3)", f"%K {k:.0f} / %D {d:.0f}", pts, 0.5, note)


def _check_bollinger(ind, _bench) -> Signal | None:
    pct_b = ind["pct_b"].iloc[-1]
    if not _ok(pct_b):
        return None
    if pct_b > 1:
        pts, note = -0.3, "Closed above the upper band: stretched."
    elif pct_b >= 0.5:
        pts, note = 0.5, "Riding the upper half of the bands."
    elif pct_b >= 0:
        pts, note = -0.3, "Lower half of the bands: sellers in control."
    else:
        pts, note = 0.2, "Below the lower band: oversold extreme."
    return Signal("Bollinger %B", f"{pct_b:.2f}", pts, 0.75, note)


def _check_obv(ind, _bench) -> Signal | None:
    if len(ind) < 25:
        return None
    obv_delta = ind["obv"].iloc[-1] - ind["obv"].iloc[-21]
    price_delta = ind["Close"].iloc[-1] / ind["Close"].iloc[-21] - 1
    obv_up, price_up = obv_delta > 0, price_delta > 0
    pts, note = {
        (True, True): (1.0, "Accumulation: volume confirms the rise."),
        (True, False): (0.5, "Quiet accumulation on dips (bullish divergence)."),
        (False, True): (-0.5, "Rally on weak volume: not confirmed."),
        (False, False): (-1.0, "Distribution: volume confirms the decline."),
    }[(obv_up, price_up)]
    return Signal("On-balance volume (20d)", f"OBV {'up' if obv_up else 'down'}, price {price_delta:+.1%}", pts, 1.5, note)


def _check_volume_surge(ind, _bench) -> Signal | None:
    avg50 = ind["vol_avg50"].iloc[-1]
    if not _ok(avg50) or avg50 == 0:
        return None
    ratio = ind["Volume"].tail(5).mean() / avg50
    if ratio < 1.5:
        return None
    up = ind["Close"].iloc[-1] > ind["Close"].iloc[-6]
    return Signal("Volume surge", f"{ratio:.1f}x 50-day avg", 0.6 if up else -0.6, 0.75,
                  "Heavy buying volume." if up else "Heavy selling volume.")


def _check_relative_strength(ind, bench: pd.Series | None) -> Signal | None:
    if bench is None or len(bench) < 130 or len(ind) < 130:
        return None
    rs3 = _ret(ind["Close"], 63) - _ret(bench, 63)
    rs6 = _ret(ind["Close"], 126) - _ret(bench, 126)
    pts = 1.0 if rs3 > 0 and rs6 > 0 else -1.0 if rs3 < 0 and rs6 < 0 else 0.0
    note = {1.0: "Outperforming Nifty over 3 and 6 months: market leader.",
            -1.0: "Lagging Nifty over 3 and 6 months: market laggard."}.get(pts, "Mixed vs Nifty.")
    return Signal("Relative strength vs Nifty", f"3M {rs3:+.1%}, 6M {rs6:+.1%}", pts, 2.0, note)


def extension_atr(ind: pd.DataFrame) -> float:
    """How many ATRs price sits above (+) or below (-) its 20-EMA."""
    last = ind.iloc[-1]
    return float((last.Close - last.ema20) / last.atr) if _ok(last.ema20, last.atr) and last.atr else NAN


def _check_extension(ind, _bench) -> Signal | None:
    gap = extension_atr(ind)
    if not _ok(gap) or abs(gap) <= 2:
        return None
    ema20 = ind["ema20"].iloc[-1]
    if gap > EXTENDED_ATR:
        return Signal("Extension vs 20-EMA", f"{gap:.1f} ATR above", -0.8, 1.0,
                      f"Over-extended after a sharp run: chasing risk. Better entry on a pullback toward {ema20:,.1f}.")
    if gap > 2:
        return Signal("Extension vs 20-EMA", f"{gap:.1f} ATR above", -0.3, 1.0, "Stretched above the 20-EMA.")
    return Signal("Extension vs 20-EMA", f"{gap:.1f} ATR below", 0.2, 1.0,
                  "Far below the 20-EMA: snap-back bounce possible.")


CHECKS = (_check_trend_200, _check_ma_alignment, _check_cross, _check_adx, _check_supertrend,
          _check_rsi, _check_macd, _check_stochastic, _check_bollinger, _check_obv,
          _check_volume_surge, _check_relative_strength, _check_extension)


def analyze_technical(ind: pd.DataFrame, benchmark: pd.Series | None) -> list[Signal]:
    return [s for check in CHECKS if (s := check(ind, benchmark)) is not None]
