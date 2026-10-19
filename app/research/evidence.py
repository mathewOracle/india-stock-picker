"""Backtest rows -> evidence: universe and per-stock statistics with PRE-REGISTERED pass rules.

The thresholds below were fixed before the first backtest run. Tuning them after looking at
results would be data snooping and would make the backtest meaningless.
"""

import math
from collections import Counter
from datetime import date, timedelta
from statistics import mean

import numpy as np

from app.research.backtest import non_overlapping

MIN_UNIVERSE_TRADES = 30
UNIVERSE_MIN_T = 1.65   # one-sided 95%: the rule must beat random entry, not just ride a bull market
SHRINK_K = 20           # weight (in trades) of the universe result when judging a single stock
MIN_STOCK_TRADES = 8    # below this, label the stock's evidence as "mostly universe"
RECENT_TRADES = 12
RECENT_YEARS = 3         # "does it still work lately?" window, reported next to the full history
MIN_YEAR_TRADES = 20     # a calendar year needs this many trades to count in years_positive


def trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    rets = np.array([t["net_return"] for t in trades])
    wins, losses = rets[rets > 0].sum(), -rets[rets <= 0].sum()
    return {
        "n": len(trades), "win_rate": float((rets > 0).mean()), "avg_return": float(rets.mean()),
        "median_return": float(np.median(rets)), "expectancy_r": float(mean(t["r_multiple"] for t in trades)),
        "profit_factor": float(wins / losses) if losses > 0 else None,
        "avg_bars": float(mean(t["bars"] for t in trades)), "best": float(rets.max()), "worst": float(rets.min()),
        "exits": dict(Counter(t["reason"] for t in trades)),
    }


def edge_test(trades: list[dict], control_mean: float) -> tuple[float, float]:
    """(mean excess over the control group, t-statistic of that excess)."""
    if not trades or not math.isfinite(control_mean):
        return float("nan"), float("nan")
    excess = np.array([t["net_return"] for t in trades]) - control_mean
    if len(excess) < 2 or excess.std(ddof=1) == 0:
        return float(excess.mean()), float("nan")
    return float(excess.mean()), float(excess.mean() / (excess.std(ddof=1) / math.sqrt(len(excess))))


def _control_mean(rows: list[dict]) -> float:
    outcomes = [r["outcome"]["net_return"] for r in rows if r["outcome"]]
    return float(mean(outcomes)) if outcomes else float("nan")


def _block(trades: list[dict], control_rows: list[dict]) -> dict:
    control = _control_mean(control_rows)
    edge, t = edge_test(trades, control)
    return {**trade_stats(trades), "control_return": control, "edge": edge, "t_stat": t}


def _shrink(n: int, own: float, prior: float) -> float:
    own = own if math.isfinite(own) else prior
    return (n * own + SHRINK_K * prior) / (n + SHRINK_K)


def summarize(rows_by_symbol: dict[str, list[dict]]) -> dict:
    all_rows = [r for rows in rows_by_symbol.values() for r in rows]
    trades = {flag: [t for rows in rows_by_symbol.values() for t in non_overlapping(rows, flag)]
              for flag in ("buy", "strong")}
    universe = {flag: _block(trades[flag], all_rows) for flag in trades}
    buy = universe["buy"]

    dates = sorted({r["date"] for r in all_rows})
    middle = dates[len(dates) // 2] if dates else ""
    halves = [{"label": label, **_block([t for t in trades["buy"] if keep(t["signal_date"])],
                                        [r for r in all_rows if keep(r["date"])])}
              for label, keep in ((f"{dates[0] if dates else ''} to {middle}", lambda d: d < middle),
                                  (f"{middle} to {dates[-1] if dates else ''}", lambda d: d >= middle))]
    years = sorted({t["signal_date"][:4] for t in trades["buy"]})
    by_year = [{"year": y, **_block([t for t in trades["buy"] if t["signal_date"].startswith(y)],
                                    [r for r in all_rows if r["date"].startswith(y)])} for y in years]

    cutoff = (date.fromisoformat(dates[-1]) - timedelta(days=365 * RECENT_YEARS)).isoformat() if dates else ""
    recent = _block([t for t in trades["buy"] if t["signal_date"] >= cutoff],
                    [r for r in all_rows if r["date"] >= cutoff])
    rated = [y for y in by_year if y["n"] >= MIN_YEAR_TRADES]
    years_positive = sum(y["edge"] > 0 for y in rated) / len(rated) if rated else float("nan")

    passed = (buy["n"] >= MIN_UNIVERSE_TRADES and buy["avg_return"] > 0 and buy["edge"] > 0
              and math.isfinite(buy["t_stat"]) and buy["t_stat"] >= UNIVERSE_MIN_T)

    stocks = {}
    for symbol, rows in rows_by_symbol.items():
        own = non_overlapping(rows, "buy")
        block = _block(own, rows)
        n = block["n"]
        shrunk_return = _shrink(n, block.get("avg_return", float("nan")), buy.get("avg_return", 0.0))
        shrunk_edge = _shrink(n, block["edge"], buy["edge"] if math.isfinite(buy["edge"]) else 0.0)
        stocks[symbol] = {
            **block, "shrunk_return": shrunk_return, "shrunk_edge": shrunk_edge,
            "evidence": "own history" if n >= MIN_STOCK_TRADES else "mostly universe (few own trades)",
            "passed": bool(passed and shrunk_return > 0 and shrunk_edge > 0),
            "recent": sorted(own, key=lambda t: t["signal_date"])[-RECENT_TRADES:],
        }
    return {"universe": universe, "halves": halves, "by_year": by_year, "passed": bool(passed),
            "recent": {"years": RECENT_YEARS, **recent}, "years_positive": years_positive,
            "period": [dates[0], dates[-1]] if dates else [], "evaluations": len(all_rows), "stocks": stocks}
