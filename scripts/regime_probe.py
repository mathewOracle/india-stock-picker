"""Why did the swing rule's edge fade after HOLDOUT_START? (diagnostic, not a strategy change)

Every split below was fixed BEFORE looking at results, and all of them are printed -- no cherry-picking:
  * time: half-years, and selection period vs hold-out
  * market regime: Nifty above/below its 200-DMA, Nifty 3-month return up/down
  * stock volatility: terciles of the stock's own volatility
Edge = average net trade return of non-overlapping rule trades minus the control group (every evaluation's
simulated trade in the same slice), exactly as the backtest and the meta-model measure it.

Usage: .venv/bin/python -m scripts.regime_probe
"""

import pandas as pd

from app.research import ledger, meta
from app.research.evidence import edge_test, trade_stats


def block(rows: pd.DataFrame, gate: pd.Series | None = None) -> dict:
    """Rule trades vs the control group inside one slice. `gate` limits which signals may trade (control unchanged)."""
    if rows.empty:
        return {"n": 0}
    control = float(rows["o_net_return"].mean())
    signals = rows["buy"] if gate is None else rows["buy"] & gate.loc[rows.index]
    trades = meta.trades(rows[signals])
    edge, t = edge_test(trades, control)
    stats = trade_stats(trades)
    if not stats["n"]:
        return {"n": 0}
    return {"n": stats["n"], "win": stats.get("win_rate", float("nan")), "avg": stats.get("avg_return", float("nan")),
            "control": control, "edge": edge, "t": t, "stop_share": stats.get("exits", {}).get("stop", 0) / max(stats["n"], 1)}


def show(title: str, slices: dict[str, pd.DataFrame], gate: pd.Series | None = None) -> None:
    print(f"\n{title}")
    print(f"  {'slice':28}{'trades':>7}{'win%':>7}{'avg':>8}{'control':>9}{'edge':>8}{'t':>7}{'stops':>7}")
    for label, rows in slices.items():
        b = block(rows, gate)
        if not b["n"]:
            print(f"  {label:28}{0:>7}")
            continue
        print(f"  {label:28}{b['n']:>7}{b['win']:>7.0%}{b['avg']:>+8.2%}{b['control']:>+9.2%}{b['edge']:>+8.2%}"
              f"{b['t']:>7.2f}{b['stop_share']:>7.0%}")


def main() -> None:
    raw, _ = ledger.load()
    df = meta.prepare(raw)
    df["period"] = df["date"].dt.year.astype(str) + "-H" + ((df["date"].dt.month > 6) + 1).astype(str)
    holdout = df["date"] >= meta.HOLDOUT_START
    print(f"{len(df)} evaluations, {int(df['buy'].sum())} rule signals, hold-out from {meta.HOLDOUT_START:%Y-%m-%d}")

    show("By half-year", {p: g for p, g in df.groupby("period")})
    show("Selection vs hold-out", {"selection": df[~holdout], "hold-out": df[holdout]})

    for title, mask in (("Nifty above 200-DMA", df["nifty_vs_200dma"] > 0), ("Nifty 3m return > 0", df["nifty_ret_3m"] > 0)):
        show(f"Regime: {title}", {f"{side} / {period}": df[(mask == (side == 'yes')) & (holdout == (period == 'hold-out'))]
                                  for side in ("yes", "no") for period in ("selection", "hold-out")})
        print(f"  share of hold-out evaluations in '{title}' = yes: {mask[holdout].mean():.0%} "
              f"(selection: {mask[~holdout].mean():.0%})")

    tercile = pd.qcut(df["vol"], 3, labels=["low vol", "mid vol", "high vol"])   # descriptive only
    show("Stock volatility tercile", {f"{v} / {p}": df[(tercile == v) & (holdout == (p == 'hold-out'))]
                                      for v in ("low vol", "mid vol", "high vol") for p in ("selection", "hold-out")})

    # Follow-up suggested BY the regime split above, so the hold-out is no longer an independent test of it.
    gate = df["nifty_vs_200dma"] > 0
    periods = {"selection": df[~holdout], "hold-out": df[holdout], "all": df}
    show("Rule as-is (reference)", periods)
    show("Rule + market gate (signals only while Nifty > 200-DMA; control unchanged)", periods, gate)
    show("Gated rule by year", {y: g for y, g in df.groupby(df["date"].dt.year.astype(str))}, gate)


if __name__ == "__main__":
    main()
