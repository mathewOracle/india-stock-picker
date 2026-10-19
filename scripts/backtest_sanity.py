"""Sanity checks for the backtest engine (run manually: python -m scripts.backtest_sanity).

1. Oracle: a signal that cheats with future returns MUST show a large edge -- proves the engine
   can detect an edge, so a 'no edge' verdict for the real rules isn't just an engine bug.
2. Information coefficient: per-date Spearman rank correlation between each feature and the
   forward return, averaged over dates (standard factor-research diagnostic).
"""

import numpy as np
import pandas as pd

from app import data
from app.research.backtest import eval_dates, replay_stock
from app.research.evidence import summarize
from app.research.pipeline import load_dataset
from app.universe import BENCHMARK

ORACLE_STOCKS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "SBIN.NS", "TITAN.NS"]


def oracle() -> None:
    bench = data.load_history(BENCHMARK)
    dates = eval_dates(bench.index)
    rows = {s: replay_stock(s, data.load_history(s), bench["Close"], dates) for s in ORACLE_STOCKS}
    for symbol_rows in rows.values():
        for r in symbol_rows:  # cheat: "buy" exactly when the next 20 sessions go up
            r["buy"] = bool(np.isfinite(r["fwd"]) and r["fwd"] > 0)
    u = summarize(rows)["universe"]["buy"]
    print(f"ORACLE  n={u['n']} avg={u['avg_return']:+.2%} control={u['control_return']:+.2%} "
          f"edge={u['edge']:+.2%} t={u['t_stat']:.1f}   (must be strongly positive)")


def information_coefficients() -> None:
    df = load_dataset()
    df = df[np.isfinite(df["fwd"])]
    df["excess"] = df["fwd"] - df.groupby("date")["fwd"].transform("mean")
    cols = ["score_price", "score_technical", "score_chart", "ret_12m", "ret_6m", "ret_3m", "ret_1m",
            "off_high", "pct_vs_200dma", "rsi", "vol", "extension", "reward_risk"]
    out = []
    for col in cols:
        # Spearman = Pearson on ranks (computed directly; no scipy needed)
        per_date = df.groupby("date").apply(lambda g: g[col].rank().corr(g["excess"].rank()), include_groups=False)
        per_date = per_date.dropna()
        out.append((col, per_date.mean(), per_date.mean() / per_date.std() * np.sqrt(len(per_date))))
    print("\nINFORMATION COEFFICIENT vs 20-day excess return (|t| > 2 ~ significant; overlapping labels inflate t)")
    for col, ic, t in sorted(out, key=lambda x: -abs(x[1])):
        print(f"  {col:16} IC={ic:+.3f}  t={t:+.1f}")


if __name__ == "__main__":
    oracle()
    information_coefficients()
