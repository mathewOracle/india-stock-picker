"""Final-verdict validation: a rule-based Buy only stays a Buy if the evidence backs it.

Checks (all visible as gates on the stock page):
  1. Backtest, universe level: did the price rules beat random entries at 95% one-sided?
  2. Backtest, this stock: shrunk history is positive (few trades => leans on the universe).
  3. News: not strongly negative (informational when coverage is thin -- never blocks on no data).
  4. ML model (added last): only counts when it beat its baselines out-of-sample.
A missing backtest is NOT treated as a pass: unvalidated means unvalidated.
"""

from dataclasses import dataclass

from app.research.sentiment import STRONG_NEGATIVE_AT
from app.scoring import BUY_LEVELS, Gate

REQUIRE_VALIDATION = True
DOWNGRADE_TO = "Watchlist"


@dataclass(frozen=True)
class Validation:
    status: str                  # "validated" | "unvalidated" | "contradicted" | "not applicable"
    gates: list[Gate]
    news: dict | None
    backtest: dict | None        # this stock's backtest block
    summary: str


def news_gate(news: dict | None) -> Gate:
    if not news or news.get("status") != "ok":
        why = "no news fetched yet" if not news else news.get("note") or "thin coverage"
        return Gate("News not strongly negative", f"not assessed ({why})", True)  # missing data never blocks
    score = news["score"]
    return Gate(f"News sentiment > {STRONG_NEGATIVE_AT:+.2f}", f"{score:+.2f} ({news['label']}, {news['n']} headlines)",
                score > STRONG_NEGATIVE_AT)


def backtest_gates(backtest: dict | None, symbol: str) -> list[Gate]:
    if not backtest:
        return [Gate("Backtest validates the rules", "not run yet", False)]
    buy = backtest["universe"]["buy"]
    t = buy.get("t_stat")
    shown = (f"edge {buy['edge']:+.2%}, t={t:.2f}, n={buy['n']}" if buy.get("n") and t == t and t is not None
             else "insufficient trades")
    gates = [Gate("Backtest: rules beat random entry (t >= 1.65)", shown, bool(backtest["passed"]))]
    stock = backtest["stocks"].get(symbol)
    if stock:
        gates.append(Gate("Backtest: this stock's history supports it",
                          f"shrunk return {stock['shrunk_return']:+.2%} ({stock['n']} own trades)",
                          bool(stock["passed"])))
    else:
        gates.append(Gate("Backtest: this stock's history supports it", "no backtest for this stock", False))
    return gates


def validate(symbol: str, rule_verdict: str, backtest: dict | None, news: dict | None,
             model_gate: Gate | None = None) -> tuple[str, Validation]:
    """Return (final verdict, validation details)."""
    stock_bt = (backtest or {}).get("stocks", {}).get(symbol)
    if rule_verdict not in BUY_LEVELS:
        return rule_verdict, Validation("not applicable", [], news, stock_bt, "Not a buy candidate; nothing to validate.")
    gates = backtest_gates(backtest, symbol) + [news_gate(news)] + ([model_gate] if model_gate else [])
    failed = [g for g in gates if not g.passed]
    if not failed:
        return rule_verdict, Validation("validated", gates, news, stock_bt, "Rules, backtest and news agree.")
    if not REQUIRE_VALIDATION:
        return rule_verdict, Validation("unvalidated", gates, news, stock_bt, "Validation failed but is not enforced.")
    contradicted = any("News" in g.name or "ML" in g.name for g in failed) and all(
        g.passed for g in gates if g.name.startswith("Backtest"))
    return DOWNGRADE_TO, Validation(
        "contradicted" if contradicted else "unvalidated", gates, news, stock_bt,
        "Rule-based candidate, but " + "; ".join(g.name.lower() for g in failed) + " not satisfied.")
