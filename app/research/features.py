"""Point-in-time feature vector for one stock on one date.

Used for backtest rows, ML training AND live ML scoring -- one function, so the model is never
fed differently-computed features in production than it was trained on.
"""

import math
import re

import pandas as pd

from app.pricecheck import PriceAssessment

# Pinned on purpose: these become model columns. A renamed signal must fail tests, not silently
# turn into an all-zero feature.
TECHNICAL_SIGNALS = ("200-DMA trend", "Moving-average stack", "50/200 cross", "ADX trend strength",
                     "Supertrend (10,3)", "RSI (14)", "MACD (12,26,9)", "Stochastic (14,3)", "Bollinger %B",
                     "On-balance volume (20d)", "Volume surge", "Relative strength vs Nifty", "Extension vs 20-EMA")
CHART_SIGNALS = ("Swing structure", "60-day trend channel", "52-week breakout", "Support / resistance",
                 "Bollinger squeeze", "Pullback setup", "Double bottom", "Double top", "Candlestick", "RSI divergence")
NUMERIC = ("score_technical", "score_chart", "score_price", "reward_risk", "ret_1m", "ret_3m", "ret_6m", "ret_12m",
           "vol", "off_high", "pct_vs_200dma", "rsi", "extension", "adx", "pct_b", "bandwidth", "macd_hist_atr",
           "volume_ratio", "st_dir", "nifty_vs_200dma", "nifty_ret_3m")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


FEATURES = tuple([f"t_{slug(n)}" for n in TECHNICAL_SIGNALS] + [f"c_{slug(n)}" for n in CHART_SIGNALS] + list(NUMERIC))


def _finite(value) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def features(pa: PriceAssessment, ind: pd.DataFrame, bench: pd.Series | None) -> dict[str, float]:
    """`ind` and `bench` must end at the evaluation date (callers guarantee no lookahead)."""
    out = {name: 0.0 for name in FEATURES[: len(TECHNICAL_SIGNALS) + len(CHART_SIGNALS)]}  # absent signal = 0 pts
    for prefix, signals in (("t", pa.technical), ("c", pa.chart.signals)):
        for s in signals:
            key = f"{prefix}_{slug(s.name)}"
            if key not in out:
                raise KeyError(f"Unknown signal {s.name!r}: add it to features.py (model columns are pinned)")
            out[key] = float(s.points)
    m, last = pa.metrics, ind.iloc[-1]
    out.update({
        "score_technical": pa.scores["technical"], "score_chart": pa.scores["chart"], "score_price": pa.scores["price"],
        "reward_risk": min(pa.plan.reward_risk, 10.0),  # cap: blue-sky fallbacks shouldn't dominate
        **{k: _finite(m[k]) for k in ("ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol", "off_high",
                                      "pct_vs_200dma", "rsi", "extension", "st_dir")},
        "adx": _finite(last.adx), "pct_b": _finite(last.pct_b), "bandwidth": _finite(last.bandwidth),
        "macd_hist_atr": _finite(last.macd_hist / last.atr) if last.atr else float("nan"),
        "volume_ratio": _finite(last.Volume / last.vol_avg50) if last.vol_avg50 else float("nan"),
        "nifty_vs_200dma": float("nan"), "nifty_ret_3m": float("nan"),
    })
    if bench is not None and len(bench) > 200:
        out["nifty_vs_200dma"] = _finite(bench.iloc[-1] / bench.tail(200).mean() - 1)
        out["nifty_ret_3m"] = _finite(bench.iloc[-1] / bench.iloc[-64] - 1)
    return out
