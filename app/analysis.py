"""Glue: cached data -> scored picks + market context. This is what the web layer calls."""

from dataclasses import dataclass, field

import pandas as pd

from app import data
from app.scoring import price_metrics, fundamental_metrics, score_universe
from app.universe import BENCHMARK, INDICES, STOCKS, VIX


@dataclass
class Analysis:
    picks: list[dict] = field(default_factory=list)
    indices: list[dict] = field(default_factory=list)
    sectors: list[dict] = field(default_factory=list)
    breadth_200dma: float | None = None
    regime: dict = field(default_factory=dict)
    as_of: str | None = None
    refreshed_at: float | None = None


def market_regime(nifty: dict | None, vix: float | None) -> dict:
    """A deliberately simple read of the tape: trend of Nifty + fear gauge."""
    if not nifty:
        return {"label": "Unknown", "detail": "No index data yet.", "tone": "slate"}
    below_200 = nifty["pct_vs_200dma"] < 0
    drawdown = nifty["off_high"]
    fearful = vix is not None and vix > 20
    if below_200 and drawdown < -0.10:
        return {"label": "Correction / Bearish",
                "detail": f"Nifty is {drawdown:.0%} off its 52-week high and below its 200-DMA. "
                          "Be selective, stagger entries.", "tone": "red"}
    if below_200:
        return {"label": "Weak / Range-bound",
                "detail": "Nifty is under its 200-DMA. Favour quality over momentum.", "tone": "amber"}
    if fearful:
        return {"label": "Uptrend, but nervous",
                "detail": f"Trend is up but India VIX is {vix:.1f}. Expect chop.", "tone": "amber"}
    return {"label": "Healthy Uptrend", "detail": "Above 200-DMA with calm volatility.", "tone": "green"}


def _sector_table(scored: pd.DataFrame) -> list[dict]:
    grouped = scored.groupby("sector").agg(
        stocks=("score", "size"), avg_score=("score", "mean"),
        avg_ret_3m=("ret_3m", "mean"), avg_ret_12m=("ret_12m", "mean"),
    )
    return grouped.sort_values("avg_score", ascending=False).reset_index().to_dict("records")


def build() -> Analysis:
    stock_data = data.load_stocks()
    result = Analysis(refreshed_at=data.last_refresh())

    index_metrics: dict[str, dict] = {}
    for symbol, name in INDICES.items():
        close = data.load_close(symbol)
        if close is not None and len(close) > 30:
            index_metrics[symbol] = price_metrics(close)
            result.indices.append({"symbol": symbol, "name": name, **index_metrics[symbol]})
    nifty = index_metrics.get(BENCHMARK)
    vix = index_metrics.get(VIX, {}).get("price")
    result.regime = market_regime(nifty, vix)
    if not stock_data:
        return result

    rows = {
        s.symbol: {"symbol": s.symbol, "name": STOCKS[s.symbol],
                   **price_metrics(s.close), **fundamental_metrics(s.fundamentals)}
        for s in stock_data if len(s.close) > 60
    }
    scored = score_universe(pd.DataFrame.from_dict(rows, orient="index"))
    result.picks = scored.reset_index(drop=True).to_dict("records")
    result.sectors = _sector_table(scored)
    result.breadth_200dma = float((scored["pct_vs_200dma"] > 0).mean())
    result.as_of = max(s.close.index[-1] for s in stock_data).strftime("%d %b %Y")
    return result
