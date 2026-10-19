"""Intraday orchestration: snapshot + context + setups -> one verdict per stock, plus market phase."""

import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from app import analysis as swing
from app import data
from app.intraday.checks import DailyContext, intraday_context, intraday_technical, tradeability_issues
from app.intraday.indicators import NO_NEW_ENTRIES, OPENING_RANGE_END, SESSION_CLOSE, SESSION_OPEN
from app.intraday.setups import COUNTER_TREND, Setup, detect_setups
from app.intraday.snapshot import Snapshot, build_snapshot
from app.signals import Signal, score
from app.universe import BENCHMARK, STOCKS

IST = ZoneInfo("Asia/Kolkata")
MIN_RR = 1.5
BIAS_WEIGHTS = {"technical": 0.7, "context": 0.3}
LONG_BIAS, SHORT_BIAS = 55, 45
STALE_MINUTES = 10


@dataclass(frozen=True)
class MarketStatus:
    phase: str
    is_open: bool
    accepting_entries: bool
    session_date: date | None
    last_bar: datetime | None
    note: str
    nifty_change: float = float("nan")


@dataclass
class IntradayReport:
    symbol: str
    name: str
    sector: str
    snapshot: Snapshot
    context_data: DailyContext
    technical: list[Signal]
    context: list[Signal]
    setups: list[Setup]
    issues: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    verdict: str = ""
    reason: str = ""
    best: Setup | None = None


@dataclass
class IntradayScan:
    reports: list[IntradayReport] = field(default_factory=list)
    status: MarketStatus | None = None
    refreshed_at: float | None = None

    def get(self, symbol: str) -> IntradayReport | None:
        return next((r for r in self.reports if r.symbol == symbol), None)


def market_phase(now: datetime) -> tuple[str, bool, bool]:
    """(label, is_open, accepting_new_entries) for NSE cash market hours. Holidays aren't known."""
    if now.weekday() >= 5:
        return "Closed (weekend)", False, False
    t = now.time()
    if t < time(9, 0):
        return "Pre-market", False, False
    if t < SESSION_OPEN:
        return "Pre-open auction", False, False
    if t < OPENING_RANGE_END:
        return "Opening range forming", True, False
    if t < NO_NEW_ENTRIES:
        return "Open: trading window", True, True
    if t < SESSION_CLOSE:
        return "Open: no new entries (square-off time)", True, False
    return "Closed", False, False


def _status(now: datetime, nifty: Snapshot | None, last_bar: datetime | None) -> MarketStatus:
    phase, is_open, accepting = market_phase(now)
    session_date = nifty.session_date if nifty else None
    if not is_open:
        note = (f"Market closed. Showing how the last session ({session_date:%d %b}) ended. "
                "Use the next-session levels to plan tomorrow." if session_date else "Market closed.")
    elif session_date != now.date():
        note = "Market hours, but no bars for today yet (holiday, or data not refreshed)."
    elif last_bar and (now.replace(tzinfo=None) - last_bar).total_seconds() > STALE_MINUTES * 60:
        note = f"Data is from {last_bar:%H:%M}. Refresh for the latest bars (Yahoo can lag a few minutes)."
    else:
        note = f"Live session. Last bar {last_bar:%H:%M} IST." if last_bar else "Live session."
    return MarketStatus(phase, is_open and session_date == now.date(), accepting and session_date == now.date(),
                        session_date, last_bar, note, nifty.change if nifty else float("nan"))


def _earnings_in_days(ts: float, session_date: date) -> int | None:
    if not math.isfinite(ts):
        return None
    days = (datetime.fromtimestamp(ts, IST).date() - session_date).days
    return days if days >= 0 else None


def _context(metrics: dict, scores: dict, session_date: date, sector_change: float, nifty_change: float) -> DailyContext:
    return DailyContext(
        traded_value_cr=metrics["traded_value_cr"], atr_pct=metrics["atr"] / metrics["price"],
        beta=metrics["beta"], daily_technical=scores["technical"], fundamental=scores["fundamental"],
        earnings_in_days=_earnings_in_days(metrics["earnings_ts"], session_date),
        sector_change=sector_change, nifty_change=nifty_change,
    )


def decide(report: IntradayReport, status: MarketStatus) -> tuple[str, str, Setup | None]:
    """Return (verdict, reason, best setup). Every 'No trade' says why."""
    if report.issues:
        return "No trade", "; ".join(report.issues), None
    if status.is_open and status.phase == "Opening range forming":
        return "Wait", "Opening range forms until 09:30", None
    if not report.setups:
        return "No trade", "No playbook setup active", None
    bias = report.scores["bias"]
    aligned = [s for s in report.setups if s.playbook.key in COUNTER_TREND
               or (s.side == "Long" and bias >= LONG_BIAS) or (s.side == "Short" and bias <= SHORT_BIAS)]
    valid = [s for s in aligned if s.plan.reward_risk >= MIN_RR]
    if not valid:
        why = "fights the intraday bias" if not aligned else f"reward/risk below {MIN_RR}"
        return "No trade", f"{report.setups[0].playbook.name} found, but it {why}", None
    best = max(valid, key=lambda s: s.plan.reward_risk)
    if status.is_open and not status.accepting_entries:
        return "No trade", "After 14:45: no new entries, manage open positions only", best
    return f"{best.side} setup", best.trigger, best


def _build(now: datetime) -> IntradayScan:
    scan = IntradayScan(refreshed_at=data.last_intraday_refresh())
    nifty_bars = data.load_intraday(BENCHMARK)
    nifty = build_snapshot(nifty_bars) if nifty_bars is not None else None
    swing_result = swing.build()
    snapshots = {}
    for symbol in STOCKS:
        bars, report = data.load_intraday(symbol), swing_result.get(symbol)
        if bars is not None and report is not None and (snap := build_snapshot(bars)) is not None:
            snapshots[symbol] = (snap, report)
    last_bar = max((s.day.index[-1] for s, _ in snapshots.values()), default=None)
    scan.status = _status(now, nifty, last_bar.to_pydatetime() if last_bar is not None else None)

    by_sector: dict[str, dict[str, float]] = {}
    for symbol, (snap, report) in snapshots.items():
        by_sector.setdefault(report.metrics["sector"], {})[symbol] = snap.change
    for symbol, (snap, sw) in snapshots.items():
        sector = sw.metrics["sector"]
        peers = [change for peer, change in by_sector[sector].items() if peer != symbol]
        ctx = _context(sw.metrics, sw.scores, snap.session_date,
                       statistics.mean(peers) if peers else float("nan"), scan.status.nifty_change)
        report = IntradayReport(symbol, STOCKS[symbol], sector, snap, ctx, intraday_technical(snap, ctx),
                                intraday_context(ctx), detect_setups(snap), tradeability_issues(ctx))
        report.scores = {"technical": score(report.technical), "context": score(report.context)}
        report.scores["bias"] = sum(report.scores[k] * w for k, w in BIAS_WEIGHTS.items())
        report.verdict, report.reason, report.best = decide(report, scan.status)
        scan.reports.append(report)

    rank = {"Long setup": 0, "Short setup": 0, "Wait": 1, "No trade": 2}
    scan.reports.sort(key=lambda r: (rank.get(r.verdict, 3), -abs(r.scores["bias"] - 50)))
    return scan


_memo: tuple[tuple, IntradayScan] | None = None


def build(now: datetime | None = None) -> IntradayScan:
    """Recomputed when either data set refreshes or the market phase changes."""
    global _memo
    now = now or datetime.now(IST)
    key = (data.last_intraday_refresh(), data.last_refresh(), market_phase(now)[0], now.strftime("%H:%M"))
    if _memo is None or _memo[0] != key:
        _memo = (key, _build(now))
    return _memo[1]


def _epoch(index: pd.DatetimeIndex) -> list[int]:
    """IST wall-clock -> epoch seconds as if UTC, so the chart's UTC axis reads in IST."""
    return [int(ts.timestamp()) for ts in index.tz_localize("UTC")]


def chart_payload(report: IntradayReport) -> dict:
    s = report.snapshot
    days = sorted(set(s.bars.index.date))[-2:]
    bars = s.bars[pd.Index(s.bars.index.date).isin(days)]
    times = _epoch(bars.index)

    def line(col: str) -> list[dict]:
        return [{"time": t, "value": round(float(v), 2)} for t, v in zip(times, bars[col]) if math.isfinite(v)]

    levels = [{"price": v, "title": k} for k, v in s.pivots.levels().items() if k in ("R2", "R1", "TC", "BC", "S1", "S2")]
    levels += [{"price": s.pdh, "title": "PDH"}, {"price": s.pdl, "title": "PDL"}]
    if s.opening_range:
        levels += [{"price": s.opening_range[0], "title": "OR high"}, {"price": s.opening_range[1], "title": "OR low"}]
    plan = report.best.plan if report.best else None
    return {
        "candles": [{"time": t, "open": r.Open, "high": r.High, "low": r.Low, "close": r.Close}
                    for t, r in zip(times, bars.itertuples())],
        "volume": [{"time": t, "value": float(r.Volume), "color": "#16a34a80" if r.Close >= r.Open else "#dc262680"}
                   for t, r in zip(times, bars.itertuples())],
        **{col: line(col) for col in ("vwap", "vwap_u2", "vwap_l2", "ema9", "ema21", "rsi")},
        "levels": levels,
        "plan": {"entry": plan.entry, "stop": plan.stop, "target1": plan.target1, "target2": plan.target2} if plan else None,
    }
