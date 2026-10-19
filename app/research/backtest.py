"""Walk-forward replay of the live price rules over 10 years -- with no lookahead.

One evaluation (every EVAL_EVERY sessions, per stock):
  1. Slice the indicator frame so it ENDS at the evaluation date (indicators are causal; tested).
  2. Run the exact live price assessment (app.pricecheck) on that slice; benchmark sliced too.
  3. Simulate the trade the plan describes: enter at the NEXT session's open, stop / target 1 /
     time-exit. If stop and target are both touched on one day, assume the stop (conservative);
     gaps through a level fill at the open; costs are charged on both sides.
Every evaluation records that hypothetical outcome whether or not the rule fired: the non-signal
rows are the control group the rule has to beat.
"""

from dataclasses import asdict, dataclass

import pandas as pd

from app.indicators import compute_indicators
from app.pricecheck import assess_price, price_signal
from app.research.features import features

EVAL_EVERY = 5           # weekly re-evaluation, like someone running the screener every week
WARMUP = 260             # 200-DMA + 52-week range need a year of bars first
WINDOW = 300             # bars handed to the analysers (live uses ~2 years; 300 covers every lookback)
MAX_HOLD = 40            # time stop, in sessions (~2 months)
COST_PER_SIDE = 0.0015   # brokerage + STT + stamp + slippage, deliberately conservative
HORIZON = 20             # forward-return horizon for ML labels (sessions)
FINAL_BARS = MAX_HOLD + 2  # this many future bars => the row's outcome can never change


@dataclass(frozen=True)
class Outcome:
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    reason: str
    bars: int
    net_return: float
    r_multiple: float


def simulate(ind: pd.DataFrame, i: int, stop: float, target: float) -> Outcome | None:
    """Trade signalled at the close of bar i. None = never entered, or still open when data ends."""
    if i + 1 >= len(ind):
        return None
    o, h, l, c = (ind[col].to_numpy() for col in ("Open", "High", "Low", "Close"))
    entry = float(o[i + 1])
    if not stop < entry < target:  # gapped through a level before we could get in
        return None
    last = i + MAX_HOLD
    if last >= len(ind):
        last = len(ind) - 1
    exit_price, reason, j = None, "", i + 1
    for j in range(i + 1, last + 1):
        gap_day = j > i + 1  # on the entry day the open IS our entry
        if gap_day and o[j] <= stop:
            exit_price, reason = float(o[j]), "stop (gap)"
        elif l[j] <= stop:
            exit_price, reason = stop, "stop"  # checked before target: conservative
        elif gap_day and o[j] >= target:
            exit_price, reason = float(o[j]), "target (gap)"
        elif h[j] >= target:
            exit_price, reason = target, "target"
        if exit_price is not None:
            break
    if exit_price is None:
        if last < i + MAX_HOLD:  # data ran out before the time stop: outcome unknown, exclude
            return None
        exit_price, reason = float(c[last]), "time"
    risk = entry - stop
    net = exit_price / entry - 1 - 2 * COST_PER_SIDE
    r_multiple = (exit_price - entry) / risk - 2 * COST_PER_SIDE * entry / risk
    return Outcome(ind.index[i + 1].strftime("%Y-%m-%d"), ind.index[j].strftime("%Y-%m-%d"), entry,
                   float(exit_price), reason, int(j - i), float(net), float(r_multiple))


def forward_return(ind: pd.DataFrame, i: int, horizon: int = HORIZON) -> float:
    """Next open -> close `horizon` sessions later (what a trader acting on the signal could earn)."""
    if i + horizon >= len(ind):
        return float("nan")
    return float(ind["Close"].iloc[i + horizon] / ind["Open"].iloc[i + 1] - 1)


def replay_stock(symbol: str, history: pd.DataFrame, bench: pd.Series | None,
                 eval_dates: list[pd.Timestamp]) -> list[dict]:
    """Every evaluation row for one stock. Pure function of its inputs (safe to run in a subprocess)."""
    ind = compute_indicators(history)
    rows = []
    for date, i in zip(eval_dates, ind.index.get_indexer(eval_dates).tolist()):  # plain ints, not numpy
        if i < WARMUP:  # also catches -1 (date missing for this stock)
            continue
        # max(0, ..): a negative start would make Python slice from the END -- i.e. future bars
        window = ind.iloc[max(0, i + 1 - WINDOW): i + 1]
        bench_now = bench.loc[:date] if bench is not None else None
        pa = assess_price(window, bench_now)
        outcome = simulate(ind, i, pa.plan.stop, pa.plan.target1)
        rows.append({
            "date": date.strftime("%Y-%m-%d"), "symbol": symbol,
            "buy": bool(price_signal(pa, "Buy")), "strong": bool(price_signal(pa, "Strong Buy")),
            "fwd": forward_return(ind, i), "outcome": asdict(outcome) if outcome else None,
            "final": bool(i + FINAL_BARS < len(ind)),
            "features": features(pa, window, bench_now),
        })
    return rows


def non_overlapping(rows: list[dict], flag: str) -> list[dict]:
    """Signal trades one stock could actually take: no new entry while a position is open."""
    trades, busy_until = [], ""
    for row in sorted(rows, key=lambda r: r["date"]):
        out = row["outcome"]
        if row[flag] and out and out["entry_date"] > busy_until:
            trades.append({**out, "symbol": row["symbol"], "signal_date": row["date"]})
            busy_until = out["exit_date"]
    return trades


def eval_dates(bench_index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    return list(bench_index[::EVAL_EVERY])
