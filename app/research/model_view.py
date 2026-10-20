"""Information-only view of the swing meta-model: its score, how similar scores fared out of sample, and why.

This NEVER changes a verdict. The only way the model influences a verdict is ml.active_model(), which hands
it to the validation gate solely when the saved report says it was adopted. Kept apart from metrics["ml_prob"]
on purpose: that key means "adopted gate score" (the intraday ML column reads it).
"""

import math

import numpy as np
import pandas as pd

from app import data
from app.research import ml
from app.research.features import CHART_SIGNALS, TECHNICAL_SIGNALS, slug

TOP_DRIVERS = 3
LABELS = {f"t_{slug(n)}": n for n in TECHNICAL_SIGNALS} | {f"c_{slug(n)}": n for n in CHART_SIGNALS} | {
    "score_technical": "Technical score", "score_chart": "Chart score", "score_price": "Price score",
    "reward_risk": "Reward/risk of the plan", "ret_1m": "1-month return", "ret_3m": "3-month return",
    "ret_6m": "6-month return", "ret_12m": "12-month return", "vol": "Volatility", "off_high": "Distance from 52w high",
    "pct_vs_200dma": "Price vs its 200-DMA", "rsi": "RSI", "extension": "Extension vs 20-EMA", "adx": "ADX",
    "pct_b": "Bollinger %B", "bandwidth": "Bollinger bandwidth", "macd_hist_atr": "MACD histogram",
    "volume_ratio": "Volume vs 50-day average", "st_dir": "Supertrend direction",
    "nifty_vs_200dma": "Nifty vs its 200-DMA", "nifty_ret_3m": "Nifty 3-month return"}


def load() -> tuple[dict | None, dict | None]:
    """(saved model bundle, its training report). A broken or mismatched file means 'no view', never a crash."""
    report = data.load_result("model")
    if not report or report.get("target") != "trade_win":
        return None, None
    try:
        bundle = ml.load_model(ml.MODEL_PATH)
    except Exception:
        return None, report
    if not bundle or bundle.get("columns") != report.get("columns"):   # file and report from different runs
        return None, report
    return bundle, report


def similar(calibration: dict | None, p: float) -> dict | None:
    """The calibration bin this score falls in: how trades with similar scores really did."""
    for b in (calibration or {}).get("bins", []):
        if b["lo"] <= p < b["hi"] or (b["hi"] >= 1.0 and p >= b["lo"]):
            return b
    return None


def drivers(bundle: dict, row: dict) -> list[dict]:
    """Per-feature push on the log-odds for linear models (coef x standardised value). Empty for tree models."""
    model = bundle["model"]
    final = model.steps[-1][1] if hasattr(model, "steps") else None
    if final is None or not hasattr(final, "coef_"):
        return []
    frame = pd.DataFrame([row])[bundle["columns"]]
    effects = np.asarray(model[:-1].transform(frame))[0] * final.coef_[0]
    out = [{"feature": c, "label": LABELS.get(c, c.replace("_", " ")), "value": row.get(c), "effect": float(e)}
           for c, e in zip(bundle["columns"], effects) if math.isfinite(e) and abs(e) > 1e-6]
    return sorted(out, key=lambda d: -abs(d["effect"]))


def view(bundle: dict | None, report: dict | None, row: dict | None) -> dict | None:
    if not bundle or not report or row is None:
        return None
    try:
        p = ml.predict(bundle, row)
        pushes = drivers(bundle, row)
    except Exception:
        return None
    calib = report.get("calibration", {})
    return {
        "p": p, "cut": bundle["cut"], "adopted": bool(report["adopted"]), "why": report["why"], "model": report["best"],
        "columns": len(bundle["columns"]), "trained_period": report.get("period"),
        "selection": calib.get("selection"), "holdout": calib.get("holdout"),
        "similar_selection": similar(calib.get("selection"), p), "similar_holdout": similar(calib.get("holdout"), p),
        "up": [d for d in pushes if d["effect"] > 0][:TOP_DRIVERS],
        "down": [d for d in pushes if d["effect"] < 0][:TOP_DRIVERS],
    }
