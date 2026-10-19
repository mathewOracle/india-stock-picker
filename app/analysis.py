"""Glue: cached data -> full three-pillar report per stock + market context."""

import math
import statistics
from dataclasses import dataclass, field

import pandas as pd

from app import data
from app.fundamentals import LOSS_PE, analyze_fundamentals, fundamental_metrics
from app.indicators import compute_indicators
from app.patterns import ChartReport
from app.pricecheck import assess_price
from app.research import ml
from app.research.features import features
from app.research.verdict import Validation, validate
from app.scoring import Gate, composite, decide, fundamental_flags
from app.signals import Signal, best, score, worst
from app.technical import beta, price_metrics, traded_value_cr
from app.tradeplan import TradePlan
from app.universe import BENCHMARK, INDICES, STOCKS, VIX

MIN_SECTOR_PEERS = 3
CHART_LEVELS_EACH_SIDE = 3  # more S/R lines than this just clutters the chart


@dataclass
class StockReport:
    symbol: str
    name: str
    metrics: dict
    technical: list[Signal]
    chart: ChartReport
    fundamental: list[Signal]
    plan: TradePlan
    scores: dict[str, float] = field(default_factory=dict)
    verdict: str = ""            # FINAL verdict (after backtest / news validation)
    rule_verdict: str = ""       # what the price + fundamental rules alone said
    validation: Validation | None = None
    gates: list[Gate] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def pillars(self) -> list[tuple[str, float, list[Signal]]]:
        return [("Fundamental", self.scores["fundamental"], self.fundamental),
                ("Technical", self.scores["technical"], self.technical),
                ("Chart", self.scores["chart"], self.chart.signals)]

    @property
    def blockers(self) -> list[str]:
        """Which Buy gates failed -- i.e. what we're waiting for before suggesting it."""
        return [g.name for g in self.gates if not g.passed]

    @property
    def thesis(self) -> list[str]:
        """One line per pillar: strongest positive and the main risk."""
        lines = []
        for label, value, signals in self.pillars:
            good, bad = best(signals), worst(signals)
            parts = [f"{label} {value:.0f}"]
            if good:
                parts.append(f"+ {good.name}: {good.reading}")
            if bad:
                parts.append(f"- {bad.name}: {bad.reading}")
            lines.append(" | ".join(parts))
        return lines


@dataclass
class Analysis:
    reports: list[StockReport] = field(default_factory=list)
    indices: list[dict] = field(default_factory=list)
    sectors: list[dict] = field(default_factory=list)
    breadth_200dma: float | None = None
    regime: dict = field(default_factory=dict)
    backtest: dict | None = None
    ml_note: str = ""
    as_of: str | None = None
    refreshed_at: float | None = None

    def get(self, symbol: str) -> StockReport | None:
        return next((r for r in self.reports if r.symbol == symbol), None)


def market_regime(nifty: dict | None, vix: float | None) -> dict:
    """A deliberately simple read of the tape: trend of Nifty + fear gauge."""
    if not nifty:
        return {"label": "Unknown", "detail": "No index data yet.", "tone": "slate"}
    below_200, drawdown = nifty["pct_vs_200dma"] < 0, nifty["off_high"]
    if below_200 and drawdown < -0.10:
        return {"label": "Correction / Bearish", "tone": "red",
                "detail": f"Nifty is {drawdown:.0%} off its 52-week high and below its 200-DMA. "
                          "Be selective, size small, stagger entries."}
    if below_200:
        return {"label": "Weak / Range-bound", "tone": "amber",
                "detail": "Nifty is under its 200-DMA. Favour quality and relative strength."}
    if vix is not None and vix > 20:
        return {"label": "Uptrend, but nervous", "tone": "amber",
                "detail": f"Trend is up but India VIX is {vix:.1f}. Expect chop."}
    return {"label": "Healthy Uptrend", "tone": "green", "detail": "Above 200-DMA with calm volatility."}


def _sector_pe(metrics: dict[str, dict]) -> dict[str, float]:
    """Median P/E per sector (falls back to the universe median for thin sectors)."""
    valid = {s: m for s, m in metrics.items() if math.isfinite(m["pe"]) and 0 < m["pe"] < LOSS_PE}
    universe = statistics.median(m["pe"] for m in valid.values()) if valid else float("nan")
    by_sector: dict[str, list[float]] = {}
    for m in valid.values():
        by_sector.setdefault(m["sector"], []).append(m["pe"])
    return {sector: statistics.median(pes) if len(pes) >= MIN_SECTOR_PEERS else universe
            for sector, pes in by_sector.items()} | {"__universe__": universe}


def _ml_gate(model, pa, ind: pd.DataFrame, bench: pd.Series | None, metrics: dict) -> Gate | None:
    """Only exists when an ADOPTED model is loaded; any scoring problem means 'no ML', never a crash."""
    if model is None:
        return None
    try:
        p = ml.predict(model, features(pa, ind, bench))
    except Exception:
        return None
    metrics["ml_prob"] = p
    return Gate(f"ML model agrees (P(outperform) >= {ml.PROBA_GATE:.0%})", f"{p:.0%}", p >= ml.PROBA_GATE)


def _report(stock: data.StockData, ind: pd.DataFrame, bench: pd.Series | None,
            fund: dict, sector_pe: float, research: dict) -> StockReport:
    pa = assess_price(ind, bench)  # the same code path the backtest replays
    metrics = {**pa.metrics, **fund, "sector_pe": sector_pe, "beta": beta(ind["Close"], bench),
               "traded_value_cr": traded_value_cr(ind)}
    report = StockReport(stock.symbol, STOCKS[stock.symbol], metrics, pa.technical, pa.chart,
                         analyze_fundamentals(fund, sector_pe, pa.metrics["price"]), pa.plan)
    report.scores = {"fundamental": score(report.fundamental), "technical": pa.scores["technical"],
                     "chart": pa.scores["chart"]}
    report.scores["composite"] = composite(report.scores)
    report.flags = pa.flags + fundamental_flags(metrics)
    report.verdict, report.gates = decide(
        {**report.scores, "reward_risk": pa.plan.reward_risk, "extension": metrics["extension"]}, report.flags)
    report.rule_verdict = report.verdict
    model_gate = _ml_gate(research["model"], pa, ind, bench, metrics)
    report.verdict, report.validation = validate(
        stock.symbol, report.rule_verdict, research["backtest"], research["news"].get(stock.symbol), model_gate)
    report.gates = report.gates + report.validation.gates  # failed ones show up as "waiting for"
    return report


def _indices() -> list[dict]:
    out = []
    for symbol, name in INDICES.items():
        frame = data.load_ohlcv(symbol)
        if frame is not None and len(frame) > 30:
            out.append({"symbol": symbol, "name": name, **price_metrics(frame["Close"])})
    return out


def _build() -> Analysis:
    result = Analysis(refreshed_at=data.last_refresh(), indices=_indices())
    by_symbol = {i["symbol"]: i for i in result.indices}
    result.regime = market_regime(by_symbol.get(BENCHMARK), by_symbol.get(VIX, {}).get("price"))
    stocks = [s for s in data.load_stocks() if len(s.ohlcv) > 130]
    if not stocks:
        return result

    bench_frame = data.load_ohlcv(BENCHMARK)
    bench = bench_frame["Close"] if bench_frame is not None else None
    funds = {s.symbol: fundamental_metrics(s.fundamentals) for s in stocks}
    sector_pe = _sector_pe(funds)
    research = {"backtest": data.load_result("backtest"), "news": (data.load_result("news") or {}).get("stocks", {})}
    research["model"], result.ml_note = ml.active_model()
    result.backtest = research["backtest"]
    result.reports = sorted(
        (_report(s, compute_indicators(s.ohlcv), bench, funds[s.symbol],
                 sector_pe.get(funds[s.symbol]["sector"], sector_pe["__universe__"]), research) for s in stocks),
        key=lambda r: r.scores["composite"], reverse=True,
    )
    frame = pd.DataFrame([{"sector": r.metrics["sector"], "score": r.scores["composite"],
                           "ret_3m": r.metrics["ret_3m"], "ret_12m": r.metrics["ret_12m"],
                           "above_200": r.metrics["pct_vs_200dma"] > 0} for r in result.reports])
    result.breadth_200dma = float(frame["above_200"].mean())
    result.sectors = (frame.groupby("sector")
                      .agg(stocks=("score", "size"), avg_score=("score", "mean"),
                           avg_ret_3m=("ret_3m", "mean"), avg_ret_12m=("ret_12m", "mean"))
                      .sort_values("avg_score", ascending=False).reset_index().to_dict("records"))
    result.as_of = max(s.ohlcv.index[-1] for s in stocks).strftime("%d %b %Y")
    return result


_memo: tuple[tuple, Analysis] | None = None


def build() -> Analysis:
    """Recomputed only when prices OR research results (backtest/news) change."""
    global _memo
    stamp = (data.last_refresh(), data.last_research())
    if _memo is None or _memo[0] != stamp:
        _memo = (stamp, _build())
    return _memo[1]


def _chart_levels(report: StockReport) -> list[dict]:
    price = report.metrics["price"]
    strong = [lv for lv in report.chart.levels if lv.touches >= 2]
    below = [lv for lv in strong if lv.price < price][-CHART_LEVELS_EACH_SIDE:]
    above = [lv for lv in strong if lv.price >= price][:CHART_LEVELS_EACH_SIDE]
    return [{"price": lv.price, "title": lv.label or f"S/R x{lv.touches}"} for lv in below + above]


def chart_payload(symbol: str, report: StockReport) -> dict:
    """Series for the candlestick + indicator charts on the detail page."""
    ind = compute_indicators(data.load_ohlcv(symbol)).tail(250)
    times = [d.strftime("%Y-%m-%d") for d in ind.index]

    def line(col: str) -> list[dict]:
        return [{"time": t, "value": round(float(v), 2)} for t, v in zip(times, ind[col]) if math.isfinite(v)]

    return {
        "candles": [{"time": t, "open": r.Open, "high": r.High, "low": r.Low, "close": r.Close}
                    for t, r in zip(times, ind.itertuples())],
        "volume": [{"time": t, "value": float(r.Volume), "color": "#16a34a80" if r.Close >= r.Open else "#dc262680"}
                   for t, r in zip(times, ind.itertuples())],
        **{col: line(col) for col in ("ema20", "sma50", "sma200", "bb_upper", "bb_lower", "st_line",
                                      "rsi", "macd", "macd_signal", "macd_hist")},
        "levels": _chart_levels(report),
        "plan": {"entry": report.plan.entry, "stop": report.plan.stop,
                 "target1": report.plan.target1, "target2": report.plan.target2},
    }
