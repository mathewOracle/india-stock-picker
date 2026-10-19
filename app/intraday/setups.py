"""The intraday playbook: each setup's rules are written ONCE and used both to detect it and to explain it."""

from collections.abc import Callable
from dataclasses import dataclass

from app.intraday.snapshot import Snapshot
from app.tradeplan import MAX_T1_R, TradePlan, pick_targets

MIN_STOP_ATR = 1.0      # stops tighter than one 5m-ATR get hit by ordinary noise
LEVEL_NOISE_ATR = 0.75  # levels closer than this to entry aren't meaningful targets


@dataclass(frozen=True)
class Playbook:
    key: str
    name: str
    style: str
    when: str
    rules: tuple[str, ...]
    stop: str
    target: str


@dataclass(frozen=True)
class Setup:
    playbook: Playbook
    side: str        # "Long" | "Short"
    trigger: str     # what fired, in plain words
    plan: TradePlan


PLAYBOOKS: dict[str, Playbook] = {p.key: p for p in [
    Playbook("orb", "Opening Range Breakout (ORB)", "Breakout",
             "First 15 minutes set the battle lines; a decisive break shows who won the open.",
             ("Price closes beyond the 09:15-09:30 high (long) or low (short)",
              "Price on the same side of VWAP", "Relative volume >= 1.0x",
              "Not more than 1.5 ATR past the breakout level (no chasing)"),
             "Middle of the opening range", "Next pivot / previous-day level, else 2R"),
    Playbook("vwap_pullback", "VWAP Pullback", "Trend continuation",
             "On trend days institutions defend VWAP; the first touch is a low-risk entry.",
             (">= 70% of today's bars on one side of VWAP (trend day)", "EMA 9 vs 21 agrees with the trend",
              "Price dipped to within 0.3 ATR of VWAP and is bouncing away"),
             "0.2 ATR beyond the pullback extreme / VWAP", "Day high/low, pivots, else 2R"),
    Playbook("gap_and_go", "Gap and Go", "Momentum",
             "A gap of 1%+ that holds above the opening range signals strong overnight demand.",
             ("Gap >= 1% from the previous close", "Price beyond the opening range in the gap's direction",
              "Price on the gap's side of VWAP"),
             "0.3 ATR on the other side of VWAP", "Next pivot level, else 2R"),
    Playbook("gap_fill", "Gap Fill (Fade)", "Reversal",
             "Gaps that can't hold VWAP often get pulled back to the previous close.",
             ("Gap >= 1% from the previous close", "Price back through VWAP and the opening range against the gap",
              "Previous close not yet reached"),
             "0.3 ATR beyond VWAP", "Previous close (the gap fill)"),
    Playbook("cpr_trend", "CPR Trend Day", "Trend",
             "A narrow Central Pivot Range (< 0.3%) often precedes a strong one-directional day.",
             ("CPR width < 0.3% of price", "Price beyond the CPR (above TC for longs, below BC for shorts)",
              "Price on the same side of VWAP for most of the session"),
             "0.3 ATR back inside the CPR", "R1/R2 (longs) or S1/S2 (shorts)"),
    Playbook("pdh_break", "Previous-Day High/Low Breakout", "Breakout",
             "Clearing yesterday's extreme expands the range and triggers stop orders.",
             ("Fresh break above PDH (long) or below PDL (short) within the last 6 bars",
              "Relative volume >= 1.0x"),
             "0.5 ATR back inside yesterday's range", "Next pivot level, else 2R"),
    Playbook("vwap_reversion", "VWAP Mean Reversion", "Counter-trend",
             "Stretched 2 standard deviations from VWAP with extreme RSI, price tends to snap back.",
             ("Price beyond the VWAP 2-sigma band", "RSI(14, 5m) < 30 for longs or > 70 for shorts",
              "Counter-trend: take smaller size"),
             "0.3 ATR beyond the day's extreme", "VWAP"),
]}


def _levels(s: Snapshot) -> list[float]:
    return [*s.pivots.levels().values(), s.pdh, s.pdl, s.day_high, s.day_low, s.prev_close]


def _setup(s: Snapshot, key: str, side: str, stop: float, trigger: str, target: float | None = None) -> Setup | None:
    direction = 1 if side == "Long" else -1
    entry = s.price
    stop = min(stop, entry - MIN_STOP_ATR * s.atr) if direction > 0 else max(stop, entry + MIN_STOP_ATR * s.atr)
    risk = abs(entry - stop)
    levels = [lv for lv in _levels(s) if (lv - entry) * direction >= LEVEL_NOISE_ATR * s.atr]
    target1, target2 = pick_targets(entry, risk, levels, direction)
    if target is not None and (target - entry) * direction > 0:
        if abs(target - entry) <= MAX_T1_R * risk:
            target1, target2 = target, target + direction * risk  # fixed objective, T2 = 1R beyond it
        else:  # objective is far away: bank at 3R, let the rest run to it (same no-fantasy-R:R rule as swing)
            target1, target2 = entry + direction * MAX_T1_R * risk, target
    plan = TradePlan(entry, stop, target1, target2, PLAYBOOKS[key].stop)
    return Setup(PLAYBOOKS[key], side, trigger, plan)


def _orb(s: Snapshot) -> Setup | None:
    if s.opening_range is None or not s.rvol >= 1.0:
        return None
    high, low = s.opening_range
    mid, vwap = (high + low) / 2, s.last.vwap
    if high < s.price <= high + 1.5 * s.atr and s.price > vwap:
        return _setup(s, "orb", "Long", mid, f"Broke opening-range high {high:,.1f} above VWAP")
    if low - 1.5 * s.atr <= s.price < low and s.price < vwap:
        return _setup(s, "orb", "Short", mid, f"Broke opening-range low {low:,.1f} below VWAP")
    return None


def _vwap_pullback(s: Snapshot) -> Setup | None:
    recent, last, prev = s.day.tail(3), s.last, s.day.iloc[-2] if len(s.day) > 1 else s.last
    vwap, share = last.vwap, s.above_vwap_share
    if share >= 0.7 and last.ema9 > last.ema21 and recent["Low"].min() <= vwap + 0.3 * s.atr \
            and last.Close > vwap and last.Close > prev.Close:
        return _setup(s, "vwap_pullback", "Long", min(recent["Low"].min(), vwap) - 0.2 * s.atr,
                      f"Trend day up ({share:.0%} above VWAP), bounced off VWAP {vwap:,.1f}")
    if share <= 0.3 and last.ema9 < last.ema21 and recent["High"].max() >= vwap - 0.3 * s.atr \
            and last.Close < vwap and last.Close < prev.Close:
        return _setup(s, "vwap_pullback", "Short", max(recent["High"].max(), vwap) + 0.2 * s.atr,
                      f"Trend day down ({1 - share:.0%} below VWAP), rejected at VWAP {vwap:,.1f}")
    return None


def _gap_and_go(s: Snapshot) -> Setup | None:
    if s.opening_range is None or abs(s.gap) < 0.01:
        return None
    high, low = s.opening_range
    vwap = s.last.vwap
    if s.gap > 0 and s.price > high and s.price > vwap:
        return _setup(s, "gap_and_go", "Long", vwap - 0.3 * s.atr, f"Gap {s.gap:+.1%} holding above opening range")
    if s.gap < 0 and s.price < low and s.price < vwap:
        return _setup(s, "gap_and_go", "Short", vwap + 0.3 * s.atr, f"Gap {s.gap:+.1%} extending below opening range")
    return None


def _gap_fill(s: Snapshot) -> Setup | None:
    if s.opening_range is None or abs(s.gap) < 0.01:
        return None
    high, low = s.opening_range
    vwap = s.last.vwap
    if s.gap > 0 and s.price < vwap and s.price < low and s.price > s.prev_close:
        return _setup(s, "gap_fill", "Short", vwap + 0.3 * s.atr, f"Gap {s.gap:+.1%} failing below VWAP",
                      target=s.prev_close)
    if s.gap < 0 and s.price > vwap and s.price > high and s.price < s.prev_close:
        return _setup(s, "gap_fill", "Long", vwap - 0.3 * s.atr, f"Gap {s.gap:+.1%} recovering above VWAP",
                      target=s.prev_close)
    return None


def _cpr_trend(s: Snapshot) -> Setup | None:
    pv, vwap, share = s.pivots, s.last.vwap, s.above_vwap_share
    if pv.cpr_width >= 0.003:
        return None
    if s.price > pv.tc and s.price > vwap and share >= 0.6:
        return _setup(s, "cpr_trend", "Long", pv.tc - 0.3 * s.atr, f"Narrow CPR ({pv.cpr_width:.2%}), holding above TC")
    if s.price < pv.bc and s.price < vwap and share <= 0.4:
        return _setup(s, "cpr_trend", "Short", pv.bc + 0.3 * s.atr, f"Narrow CPR ({pv.cpr_width:.2%}), holding below BC")
    return None


def _pdh_break(s: Snapshot) -> Setup | None:
    if not s.rvol >= 1.0:
        return None
    closes = s.day["Close"].tail(6)
    if s.price > s.pdh and closes.min() <= s.pdh:
        return _setup(s, "pdh_break", "Long", s.pdh - 0.5 * s.atr, f"Fresh break above previous-day high {s.pdh:,.1f}")
    if s.price < s.pdl and closes.max() >= s.pdl:
        return _setup(s, "pdh_break", "Short", s.pdl + 0.5 * s.atr, f"Fresh break below previous-day low {s.pdl:,.1f}")
    return None


def _vwap_reversion(s: Snapshot) -> Setup | None:
    last = s.last
    if last.Close < last.vwap_l2 and last.rsi < 30:
        return _setup(s, "vwap_reversion", "Long", s.day_low - 0.3 * s.atr,
                      f"2-sigma below VWAP with RSI {last.rsi:.0f}", target=last.vwap)
    if last.Close > last.vwap_u2 and last.rsi > 70:
        return _setup(s, "vwap_reversion", "Short", s.day_high + 0.3 * s.atr,
                      f"2-sigma above VWAP with RSI {last.rsi:.0f}", target=last.vwap)
    return None


DETECTORS: tuple[Callable[[Snapshot], Setup | None], ...] = (
    _orb, _vwap_pullback, _gap_and_go, _gap_fill, _cpr_trend, _pdh_break, _vwap_reversion)
COUNTER_TREND = {"vwap_reversion", "gap_fill"}  # allowed to fight the intraday bias


def detect_setups(s: Snapshot) -> list[Setup]:
    return [setup for detect in DETECTORS if (setup := detect(s)) is not None]
