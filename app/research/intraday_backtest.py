"""Intraday backtest: replay the 7 playbook setups over the archived 5-minute history, exactly as the live scanner sees them.

Parity with live: for each stock and session we compute indicators once on the same 5-session window the live
scanner uses, then take a decision every 15 minutes (09:30-14:45) from PREFIXES of that window (every indicator only
looks back, and a test proves a prefix equals recomputing on truncated data).

Trade rules (fixed in advance): signal at a bar's close -> entry at the NEXT bar's open -> the setup's own stop and
first target -> stop wins if both are touched in one bar -> gaps fill at the open -> forced exit at the 15:15
square-off. One trade per stock, session and (setup, side), at its first qualifying trigger. Costs charged both sides.
Only COMPLETE sessions are replayed (a half-finished day has no outcome yet).
"""

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import time as clock

import numpy as np
import pandas as pd

from app import data
from app.intraday import archive
from app.intraday.analysis import MIN_RR as LIVE_MIN_RR
from app.intraday.indicators import with_intraday_indicators
from app.intraday.setups import PLAYBOOKS, detect_setups
from app.intraday.snapshot import assemble
from app.universe import STOCKS

LOOKBACK_SESSIONS = 5        # what Yahoo's 5-day view gives the live scanner
FIRST_DECISION_START = clock(9, 25)    # the bar that CLOSES at 09:30, when the opening range is complete
LAST_DECISION_START = clock(14, 40)    # closes 14:45: the live no-new-entries cutoff
SQUARE_OFF_START = clock(15, 10)       # last bar used; closes 15:15
DECISION_EVERY_MIN = 15
COST_PER_SIDE = 0.0006       # brokerage + STT + slippage, intraday
MIN_RR = LIVE_MIN_RR         # the same reward/risk bar as the live verdict (imported, never copied)
LEDGER_PATH = data.DB_PATH.parent / "research" / "intraday_trades.csv.gz"
WORKERS = max(1, min(10, (os.cpu_count() or 2) - 2))

COLUMNS = ["symbol", "date", "decision", "setup", "side", "entry", "exit", "reason", "bars", "net_return", "r_multiple"]


def simulate(day: pd.DataFrame, pos: int, side: str, stop: float, target: float) -> dict | None:
    """Trade signalled at the close of day's bar `pos`. None = no entry, or the session has no usable end."""
    sign = 1 if side == "Long" else -1
    starts = np.array([t.time() for t in day.index])
    last = int(np.searchsorted([s > SQUARE_OFF_START for s in starts], True)) - 1   # last bar allowed (<= 15:10)
    if pos + 1 > last or starts[last] < clock(15, 0):    # no next bar, or the session's end is missing
        return None
    o, h, l, c = (day[col].to_numpy() for col in ("Open", "High", "Low", "Close"))
    entry = float(o[pos + 1])
    if not (stop < entry < target if sign > 0 else target < entry < stop):   # gapped through a level: no trade
        return None
    exit_price, reason, j = None, "", pos + 1
    for j in range(pos + 1, last + 1):
        first_bar = j == pos + 1                      # on the entry bar the open IS our entry
        hit_stop = l[j] <= stop if sign > 0 else h[j] >= stop
        hit_target = h[j] >= target if sign > 0 else l[j] <= target
        if not first_bar and (o[j] <= stop if sign > 0 else o[j] >= stop):
            exit_price, reason = float(o[j]), "stop (gap)"
        elif hit_stop:
            exit_price, reason = stop, "stop"                 # checked before the target: conservative
        elif not first_bar and (o[j] >= target if sign > 0 else o[j] <= target):
            exit_price, reason = float(o[j]), "target (gap)"
        elif hit_target:
            exit_price, reason = target, "target"
        if exit_price is not None:
            break
    if exit_price is None:
        exit_price, reason, j = float(c[last]), "square-off", last
    risk = abs(entry - stop)
    return {"entry": entry, "exit": float(exit_price), "reason": reason, "bars": int(j - pos),
            "net_return": float(sign * (exit_price / entry - 1) - 2 * COST_PER_SIDE),
            "r_multiple": float(sign * (exit_price - entry) / risk - 2 * COST_PER_SIDE * entry / risk)}


def decision_positions(day: pd.DataFrame) -> list[int]:
    """Positions of today's bars after which we take a decision (every 15 minutes, 09:30 to 14:45)."""
    out = []
    for pos, ts in enumerate(day.index):
        minutes = (ts.hour * 60 + ts.minute) - (FIRST_DECISION_START.hour * 60 + FIRST_DECISION_START.minute)
        if FIRST_DECISION_START <= ts.time() <= LAST_DECISION_START and minutes % DECISION_EVERY_MIN == 0:
            out.append(pos)
    return out


def session_complete(day: pd.DataFrame) -> bool:
    return bool(len(day)) and day.index[-1].time() >= SQUARE_OFF_START


def replay_symbol(symbol: str, raw: pd.DataFrame, only_after: str | None = None) -> dict:
    """Trades for one stock over its complete sessions after `only_after`, plus the last session it covered."""
    dates = sorted(set(raw.index.date))
    trades, through = [], only_after
    for k in range(1, len(dates)):
        today = dates[k]
        if only_after is not None and today.isoformat() <= only_after:
            continue
        window = raw[pd.Index(raw.index.date).isin(dates[max(0, k - LOOKBACK_SESSIONS + 1): k + 1])]
        today_mask = pd.Index(window.index.date) == today
        if not session_complete(window[today_mask]):
            continue
        through = today.isoformat()
        bars = with_intraday_indicators(window)
        first_today = int(np.argmax(today_mask))
        seen: set[tuple[str, str]] = set()
        for rel in decision_positions(window[today_mask]):
            pos = first_today + rel
            snap = assemble(window.iloc[: pos + 1], bars.iloc[: pos + 1])   # prefix only: nothing from the future
            if snap is None:
                continue
            for setup in detect_setups(snap):
                key = (setup.playbook.key, setup.side)
                if key in seen or setup.plan.reward_risk < MIN_RR:
                    continue
                seen.add(key)
                result = simulate(window[today_mask], rel, setup.side, setup.plan.stop, setup.plan.target1)
                if result:
                    trades.append({"symbol": symbol, "date": today.isoformat(),
                                   "decision": window.index[pos + 1].strftime("%H:%M"),
                                   "setup": setup.playbook.key, "side": setup.side, **result})
    return {"trades": trades, "through": through}


# ---------- evidence ----------

MIN_TRADES, MIN_DAYS, MIN_HALF_TRADES = 100, 20, 30
ALPHA = 0.05
N_TESTS = len(PLAYBOOKS) * 2                 # every setup, both sides: Bonferroni over all of them


def bonferroni_t() -> float:
    from scipy.stats import norm
    return float(norm.isf(ALPHA / N_TESTS))   # one-sided: we only care about a POSITIVE edge


def day_clustered_t(trades: pd.DataFrame) -> tuple[float, int]:
    """Stocks move together on a given day, so trades on one day are not independent: test the daily means."""
    daily = trades.groupby("date")["net_return"].mean()
    n = len(daily)
    if n < 3 or daily.std(ddof=1) == 0:
        return float("nan"), n
    return float(daily.mean() / (daily.std(ddof=1) / math.sqrt(n))), n


def stats_block(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "days": 0, "status": "insufficient data"}
    t, days = day_clustered_t(trades)
    ordered = trades.sort_values("date")
    half = len(ordered) // 2
    halves = [float(ordered.iloc[:half]["net_return"].mean()), float(ordered.iloc[half:]["net_return"].mean())]
    block = {"n": int(len(trades)), "days": int(days), "win_rate": float((trades["net_return"] > 0).mean()),
             "avg_return": float(trades["net_return"].mean()), "avg_r": float(trades["r_multiple"].mean()),
             "t_stat": t, "halves": halves, "best_exit": trades["reason"].value_counts().to_dict(),
             "first": ordered["date"].iloc[0], "last": ordered["date"].iloc[-1]}
    enough = block["n"] >= MIN_TRADES and days >= MIN_DAYS
    if not enough:
        block["status"] = "insufficient data"
    else:
        steady = len(ordered) // 2 >= MIN_HALF_TRADES and all(h > 0 for h in halves)
        supported = block["avg_return"] > 0 and t >= bonferroni_t() and steady
        block["status"] = "supported" if supported else "not supported"
    return block


def summarize(trades: pd.DataFrame) -> dict:
    setups = {f"{key}:{side}": stats_block(trades[(trades["setup"] == key) & (trades["side"] == side)])
              for key in PLAYBOOKS for side in ("Long", "Short")}
    return {"setups": setups, "all": stats_block(trades), "trades": int(len(trades)),
            "bonferroni_t": bonferroni_t(), "min_trades": MIN_TRADES, "min_days": MIN_DAYS,
            "requirements": {"cost_per_side": COST_PER_SIDE, "min_rr": MIN_RR, "tests": N_TESTS}}


# ---------- incremental ledger ----------

def load_ledger() -> pd.DataFrame | None:
    return pd.read_csv(LEDGER_PATH) if LEDGER_PATH.exists() else None


def save_ledger(df: pd.DataFrame) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".tmp.gz")
    df.sort_values(["date", "symbol", "decision"]).to_csv(tmp, index=False)
    os.replace(tmp, LEDGER_PATH)


def run(progress=data._no_progress) -> dict:
    """Replay only sessions newer than each stock's progress marker, then re-summarise the whole ledger."""
    started = time.time()
    stored = load_ledger()
    markers = (data.load_result("intraday_backtest_meta") or {}).get("through", {}) if stored is not None else {}
    raws = {s: archive.load(s) for s in STOCKS}
    raws = {s: r for s, r in raws.items() if r is not None and len(set(r.index.date)) >= 2}
    if not raws:
        raise RuntimeError("No intraday archive yet: refresh intraday data first.")
    rows, failures, through = [], {}, dict(markers)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(replay_symbol, s, r, markers.get(s)): s for s, r in raws.items()}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
                rows += result["trades"]
                through[symbol] = result["through"]
            except Exception as exc:
                failures[symbol] = f"{type(exc).__name__}: {exc}"      # this stock keeps its old marker
            progress(done, len(futures), f"intraday backtest {symbol}")
    if len(failures) == len(raws):
        raise RuntimeError(f"Intraday backtest failed for every stock, e.g. {next(iter(failures.values()))}")
    fresh = pd.DataFrame(rows, columns=COLUMNS)
    full = pd.concat([stored, fresh], ignore_index=True) if stored is not None else fresh
    full = full.drop_duplicates(["symbol", "date", "setup", "side"], keep="last")
    summary = summarize(full)
    summary.update(failures=failures, seconds=round(time.time() - started, 1), generated_at=time.time(),
                   increment={"new_trades": int(len(fresh)), "through": max((d for d in through.values() if d), default=None)},
                   sessions=archive.sessions_covered())
    save_ledger(full)
    data.save_result("intraday_backtest_meta", {"through": through})
    data.save_result("intraday_backtest", summary)
    return summary


def evidence() -> dict | None:
    return data.load_result("intraday_backtest")
