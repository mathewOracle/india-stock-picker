"""Intraday checks, emitting the same Signal type as the swing pillars.

Technical: what the tape is doing *today* on 5-minute bars.
Context:   the slower, "fundamental" layer for day traders. Fundamentals don't move
           a stock in a day, so they act as filters (liquidity, volatility, event
           risk) and as a bias tilt (daily trend, business quality, sector flow).
"""

import math
from dataclasses import dataclass

from app.intraday.snapshot import Snapshot
from app.signals import Signal, grade

MIN_TRADED_VALUE_CR = 100   # below this, spreads/impact costs eat intraday edges
MIN_ATR_PCT = 0.01          # below 1% daily range there's not enough room to trade


@dataclass(frozen=True)
class DailyContext:
    traded_value_cr: float      # 20-day average daily traded value, in crore rupees
    atr_pct: float              # daily ATR(14) as % of price
    beta: float                 # vs Nifty, 1 year of daily returns
    daily_technical: float      # swing technical score (0-100)
    fundamental: float          # swing fundamental score (0-100)
    earnings_in_days: int | None
    sector_change: float        # average % change of sector peers today
    nifty_change: float


def _ok(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


# ---------- technical (today's tape) ----------

def _vwap(s: Snapshot) -> Signal | None:
    vw = s.last.vwap
    if not _ok(vw):
        return None
    gap = s.price / vw - 1
    pts = 1.0 if gap > 0.001 else -1.0 if gap < -0.001 else 0.0
    return Signal("Price vs VWAP", f"{gap:+.2%} vs VWAP {vw:,.1f}", pts, 2.0,
                  "Buyers in control (above VWAP): favour longs." if pts > 0
                  else "Sellers in control (below VWAP): favour shorts." if pts < 0 else "Sitting on VWAP: undecided.")


def _vwap_trend(s: Snapshot) -> Signal:
    share = s.above_vwap_share
    pts = 1.0 if share >= 0.8 else -1.0 if share <= 0.2 else 0.0
    return Signal("VWAP trend consistency", f"{share:.0%} of bars above VWAP", pts, 1.0,
                  "Trend day up: dips to VWAP get bought." if pts > 0
                  else "Trend day down: rallies to VWAP get sold." if pts < 0 else "Two-sided, choppy session.")


def _opening_range(s: Snapshot) -> Signal | None:
    if s.opening_range is None:
        return Signal("Opening range (15 min)", "still forming", 0.0, 1.5, "Wait for 09:30 before trading breakouts.")
    high, low = s.opening_range
    reading = f"OR {low:,.1f} - {high:,.1f}"
    if s.price > high:
        return Signal("Opening range (15 min)", reading + ", broken UP", 1.0, 1.5, "Opening-range breakout: bulls won the open.")
    if s.price < low:
        return Signal("Opening range (15 min)", reading + ", broken DOWN", -1.0, 1.5, "Opening-range breakdown: bears won the open.")
    return Signal("Opening range (15 min)", reading + ", inside", 0.0, 1.5, "Still inside the opening range: no edge yet.")


def _ema_cross(s: Snapshot) -> Signal | None:
    last = s.last
    if not _ok(last.ema9, last.ema21):
        return None
    up = last.ema9 > last.ema21
    aligned = (last.Close > last.ema9) if up else (last.Close < last.ema9)
    pts = (1.0 if aligned else 0.4) * (1 if up else -1)
    return Signal("EMA 9 / 21 (5m)", f"EMA9 {last.ema9:,.1f} {'>' if up else '<'} EMA21 {last.ema21:,.1f}", pts, 1.0,
                  f"Short-term momentum {'up' if up else 'down'}" + ("." if aligned else ", but price is pulling back."))


def _rsi(s: Snapshot) -> Signal | None:
    value = s.last.rsi
    if not _ok(value):
        return None
    if value > 75:
        pts, note = -0.5, "Overbought on 5m: don't chase, wait for a dip."
    elif value >= 55:
        pts, note = 0.8, "Bullish intraday momentum."
    elif value >= 45:
        pts, note = 0.0, "Neutral momentum."
    elif value >= 25:
        pts, note = -0.8, "Bearish intraday momentum."
    else:
        pts, note = 0.3, "Oversold on 5m: snap-back bounce possible."
    return Signal("RSI (14, 5m)", f"{value:.1f}", pts, 1.0, note)


def _supertrend(s: Snapshot) -> Signal:
    up = s.last.st_dir > 0
    return Signal("Supertrend (10,3, 5m)", f"{'BUY' if up else 'SELL'} @ {s.last.st_line:,.1f}", 1.0 if up else -1.0, 1.0,
                  "Intraday trend-followers long; the line doubles as a trailing stop." if up
                  else "Intraday trend-followers short; the line doubles as a trailing stop.")


def _cpr(s: Snapshot) -> Signal:
    pv = s.pivots
    width = pv.cpr_width
    kind = "narrow - trending day likely" if width < 0.0025 else "wide - range day likely" if width > 0.006 else "normal"
    reading = f"CPR {pv.bc:,.1f} - {pv.tc:,.1f} ({width:.2%}, {kind})"
    if s.price > pv.tc:
        return Signal("Central Pivot Range", reading, 1.0, 1.5, "Trading above CPR: bullish day structure.")
    if s.price < pv.bc:
        return Signal("Central Pivot Range", reading, -1.0, 1.5, "Trading below CPR: bearish day structure.")
    return Signal("Central Pivot Range", reading, 0.0, 1.5, "Inside CPR: sideways risk until it breaks.")


def _gap(s: Snapshot) -> Signal | None:
    gap = s.gap
    if abs(gap) < 0.005:
        return None
    holding = (s.price > s.open) if gap > 0 else (s.price < s.open)
    filled = (s.price <= s.prev_close) if gap > 0 else (s.price >= s.prev_close)
    direction = 1 if gap > 0 else -1
    if filled:
        return Signal("Opening gap", f"{gap:+.2%}, filled", -0.7 * direction, 0.75, "Gap fully filled: the open was faded.")
    return Signal("Opening gap", f"{gap:+.2%}, {'holding' if holding else 'fading'}", (0.7 if holding else -0.2) * direction,
                  0.75, "Gap-and-go: the gap is holding." if holding else "Gap is fading toward the prior close.")


def _prev_day_levels(s: Snapshot) -> Signal:
    reading = f"PDH {s.pdh:,.1f} / PDL {s.pdl:,.1f}"
    if s.price > s.pdh:
        return Signal("Previous-day high/low", reading, 1.0, 1.0, "Above yesterday's high: range expansion up.")
    if s.price < s.pdl:
        return Signal("Previous-day high/low", reading, -1.0, 1.0, "Below yesterday's low: range expansion down.")
    return Signal("Previous-day high/low", reading, 0.0, 1.0, "Inside yesterday's range.")


def _relative_volume(s: Snapshot) -> Signal | None:
    if not _ok(s.rvol):
        return None
    if s.rvol >= 1.5:
        direction = 1 if s.change > 0 else -1
        return Signal("Relative volume", f"{s.rvol:.1f}x normal for this time", 0.8 * direction, 1.0,
                      "Heavy participation: the move has conviction.")
    if s.rvol < 0.7:
        return Signal("Relative volume", f"{s.rvol:.1f}x normal for this time", 0.0, 1.0,
                      "Thin volume: breakouts are less reliable.")
    return Signal("Relative volume", f"{s.rvol:.1f}x normal for this time", 0.0, 1.0, "Normal participation.")


def _relative_strength(s: Snapshot, ctx: DailyContext) -> Signal | None:
    if not _ok(ctx.nifty_change):
        return None
    spread = s.change - ctx.nifty_change
    pts = 1.0 if spread > 0.005 else -1.0 if spread < -0.005 else 0.0
    return Signal("Strength vs Nifty (today)", f"{s.change:+.2%} vs Nifty {ctx.nifty_change:+.2%}", pts, 1.5,
                  "Outperforming the index today: a leader." if pts > 0
                  else "Underperforming the index today: a laggard." if pts < 0 else "Moving with the index.")


def intraday_technical(s: Snapshot, ctx: DailyContext) -> list[Signal]:
    candidates = [_vwap(s), _vwap_trend(s), _opening_range(s), _ema_cross(s), _rsi(s), _supertrend(s),
                  _cpr(s), _gap(s), _prev_day_levels(s), _relative_volume(s), _relative_strength(s, ctx)]
    return [c for c in candidates if c is not None]


# ---------- context (the 'fundamental' layer for day traders) ----------

def intraday_context(ctx: DailyContext) -> list[Signal]:
    out = []
    if _ok(ctx.traded_value_cr):
        out.append(Signal("Liquidity (avg daily traded value)", f"Rs {ctx.traded_value_cr:,.0f} Cr",
                          grade(ctx.traded_value_cr, [(500, 1), (MIN_TRADED_VALUE_CR, 0.5), (25, -0.3)], below=-1), 1.5,
                          "Deep liquidity: tight spreads, easy exits." if ctx.traded_value_cr >= MIN_TRADED_VALUE_CR
                          else "Thin liquidity: wide spreads and slippage."))
    if _ok(ctx.atr_pct):
        pts = 1.0 if 0.012 <= ctx.atr_pct <= 0.04 else -0.5 if ctx.atr_pct > 0.04 else -0.3
        out.append(Signal("Daily volatility (ATR%)", f"{ctx.atr_pct:.2%} average daily range", pts, 1.0,
                          "Enough range to trade." if pts > 0 else "Very volatile: cut size." if ctx.atr_pct > 0.04
                          else "Moves too little for intraday."))
    if ctx.earnings_in_days is not None and 0 <= ctx.earnings_in_days <= 3:
        today = ctx.earnings_in_days == 0
        out.append(Signal("Results / event risk", "results TODAY" if today else f"results in {ctx.earnings_in_days} days",
                          -1.0 if today else -0.3, 1.5,
                          "Results day: gaps and whipsaws. Avoid or trade tiny." if today else "Results soon: expect nervous moves."))
    if _ok(ctx.daily_technical):
        pts = 1.0 if ctx.daily_technical >= 60 else -1.0 if ctx.daily_technical <= 40 else 0.0
        out.append(Signal("Daily-chart trend alignment", f"daily technical score {ctx.daily_technical:.0f}", pts, 1.5,
                          "Daily trend up: intraday longs swim with the tide." if pts > 0
                          else "Daily trend down: favour intraday shorts." if pts < 0 else "Daily trend neutral."))
    if _ok(ctx.fundamental):
        pts = 0.3 if ctx.fundamental >= 65 else -0.3 if ctx.fundamental <= 40 else 0.0
        out.append(Signal("Business quality (tilt only)", f"fundamental score {ctx.fundamental:.0f}", pts, 0.5,
                          "Quality names tend to see dips bought. Small effect intraday." if pts > 0
                          else "Weak fundamentals: rallies tend to be sold. Small effect intraday." if pts < 0
                          else "Fundamentals neutral (they rarely matter within a day)."))
    if _ok(ctx.sector_change):
        pts = 0.7 if ctx.sector_change > 0.005 else -0.7 if ctx.sector_change < -0.005 else 0.0
        out.append(Signal("Sector flow today", f"sector peers {ctx.sector_change:+.2%}", pts, 1.0,
                          "Sector tailwind." if pts > 0 else "Sector headwind." if pts < 0 else "Sector flat."))
    if _ok(ctx.beta):
        out.append(Signal("Beta vs Nifty", f"{ctx.beta:.2f}", 0.0, 0.0,  # informational: sizes, doesn't score
                          "Moves more than the index: size smaller." if ctx.beta > 1.2 else "Moves roughly with or less than the index."))
    return out


def tradeability_issues(ctx: DailyContext) -> list[str]:
    """Hard filters: any issue here means 'no intraday trade' regardless of setup quality."""
    issues = []
    if _ok(ctx.traded_value_cr) and ctx.traded_value_cr < MIN_TRADED_VALUE_CR:
        issues.append(f"Illiquid (< Rs {MIN_TRADED_VALUE_CR} Cr/day)")
    if _ok(ctx.atr_pct) and ctx.atr_pct < MIN_ATR_PCT:
        issues.append("Too little daily range (< 1% ATR)")
    if ctx.earnings_in_days == 0:
        issues.append("Results today")
    return issues
