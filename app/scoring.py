"""Pure scoring engine: metrics in, explainable 0-100 scores out. No I/O here.

Every metric is turned into a percentile rank *within the universe*, so scores
answer "how does this stock look versus its peers?" -- not "is it cheap in
absolute terms?". Factor score = mean of its metric ranks; composite = weighted
sum of factor scores.
"""

import math

import numpy as np
import pandas as pd

NAN = float("nan")
LOSS_PE = 1_000.0  # loss-making companies get a sentinel P/E so they rank dead last on value

# factor -> {metric: higher_is_better}
FACTORS: dict[str, dict[str, bool]] = {
    "momentum": {"ret_3m": True, "ret_6m": True, "mom_12_1": True},
    "trend": {"pct_vs_200dma": True, "off_high": True, "golden": True},
    "quality": {"roe": True, "margin": True, "de": False},
    "growth": {"rev_growth": True, "earn_growth": True},
    "valuation": {"pe": False, "pb": False},
    "risk": {"vol": False, "max_dd": True},
}
WEIGHTS: dict[str, float] = {
    "momentum": 0.25, "trend": 0.10, "quality": 0.25,
    "growth": 0.15, "valuation": 0.15, "risk": 0.10,
}
assert math.isclose(sum(WEIGHTS.values()), 1.0)

STRONG_BUY, BUY, HOLD, AVOID = 65, 57, 47, 0


# ---------- raw metrics ----------

def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NAN
    return number if math.isfinite(number) else NAN


def _ret(close: pd.Series, days: int) -> float:
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1) if len(close) > days else NAN


def price_metrics(close: pd.Series) -> dict:
    """Trend/momentum/risk metrics from a daily close series (works for indices too)."""
    last = float(close.iloc[-1])
    sma50 = close.tail(50).mean() if len(close) >= 50 else NAN
    sma200 = close.tail(200).mean() if len(close) >= 200 else NAN
    year = close.tail(252)
    daily = close.pct_change().dropna().tail(252)
    return {
        "price": last,
        "ret_1d": _ret(close, 1),
        "ret_1m": _ret(close, 21),
        "ret_3m": _ret(close, 63),
        "ret_6m": _ret(close, 126),
        "ret_12m": _ret(close, 252),
        # classic "12-1" momentum: last year's gain, skipping the most recent month
        "mom_12_1": float(close.iloc[-22] / close.iloc[-253] - 1) if len(close) >= 253 else NAN,
        "pct_vs_200dma": float(last / sma200 - 1) if sma200 == sma200 else NAN,
        "golden": float(sma50 > sma200) if sma50 == sma50 and sma200 == sma200 else NAN,
        "off_high": float(last / year.max() - 1),
        "vol": float(daily.std() * np.sqrt(252)) if len(daily) > 20 else NAN,
        "max_dd": float((year / year.cummax() - 1).min()),
    }


def fundamental_metrics(f: dict) -> dict:
    pe = _num(f.get("trailingPE"))
    if pe <= 0:  # loss-making
        pe = LOSS_PE
    financial = f.get("sector") == "Financial Services"
    return {
        "sector": f.get("sector") or "Other",
        "market_cap": _num(f.get("marketCap")),
        "pe": pe,
        "pb": _num(f.get("priceToBook")),
        "roe": _num(f.get("returnOnEquity")),
        "margin": _num(f.get("profitMargins")),
        # leverage is the business model for banks/NBFCs, so it says nothing about quality
        "de": NAN if financial else _num(f.get("debtToEquity")),
        "rev_growth": _num(f.get("revenueGrowth")),
        "earn_growth": _num(f.get("earningsGrowth")),
        "div_yield": _num(f.get("dividendYield")),
    }


# ---------- scoring ----------

def score_universe(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add factor columns, composite `score`, `verdict`, `flags` and `reasons`."""
    out = metrics.copy()
    for factor, spec in FACTORS.items():
        ranks = pd.DataFrame({
            metric: metrics[metric].rank(pct=True, ascending=higher) * 100
            for metric, higher in spec.items()
        })
        out[factor] = ranks.mean(axis=1, skipna=True).fillna(50.0)  # no data -> neutral
    out["score"] = sum(out[factor] * weight for factor, weight in WEIGHTS.items())
    out["flags"] = out.apply(_flags, axis=1)
    out["verdict"] = out.apply(lambda row: _verdict(row["score"], row["flags"]), axis=1)
    out["strengths"] = out.apply(lambda row: _extremes(row, lambda v: v >= 70), axis=1)
    out["weaknesses"] = out.apply(lambda row: _extremes(row, lambda v: v <= 30), axis=1)
    return out.sort_values("score", ascending=False)


def _flags(row: pd.Series) -> list[str]:
    flags = []
    if row["pct_vs_200dma"] < 0:
        flags.append("Below 200-DMA")
    if row["off_high"] < -0.30 and row["pct_vs_200dma"] < 0:
        flags.append("Falling knife")
    if row["vol"] > 0.35:
        flags.append("High volatility")
    if row["pe"] >= LOSS_PE:
        flags.append("Loss-making")
    elif row["pe"] > 60:
        flags.append("Expensive (P/E > 60)")
    if row["ret_1m"] > 0.15:
        flags.append("Short-term overheated")
    return flags


def _verdict(score: float, flags: list[str]) -> str:
    if score >= STRONG_BUY:
        label = "Strong Buy"
    elif score >= BUY:
        label = "Buy"
    elif score >= HOLD:
        label = "Hold"
    else:
        label = "Avoid"
    # never recommend catching a falling knife or buying losses, however nice the rest looks
    if label in ("Strong Buy", "Buy") and ("Falling knife" in flags or "Loss-making" in flags):
        return "Hold"
    # a broken trend caps conviction: "Strong Buy" demands price above the 200-DMA
    if label == "Strong Buy" and "Below 200-DMA" in flags:
        return "Buy"
    return label


def _extremes(row: pd.Series, test) -> list[str]:
    return [factor for factor in FACTORS if test(row[factor])]
