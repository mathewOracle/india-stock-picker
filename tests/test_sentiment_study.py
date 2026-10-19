import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.research import sentiment_study as ss


def ts(day: str, hour: int = 0) -> float:
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc, hour=hour).timestamp()


def history(n=120, start="2025-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1e6}, index=idx)


def test_only_headlines_published_strictly_before_the_decision_date_are_used():
    h = history()
    date = h.index[60]                                    # decision date D
    d = date.strftime("%Y-%m-%d")
    prev = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    nxt = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    base = [{"t": f"old news {i}", "s": "x", "p": ts(prev) - i * 3600, "sc": 0.5} for i in range(3)]   # day before: allowed
    same_day = {"t": "same day", "s": "x", "p": ts(d, 0), "sc": -1.0}          # exactly at D 00:00 UTC: NOT allowed
    after = {"t": "after", "s": "x", "p": ts(nxt, 5), "sc": -1.0}              # future: NOT allowed
    panel = ss.symbol_panel(base + [same_day, after], h, [date])
    assert panel["n7"].iloc[0] == 3 and panel["sent_7"].iloc[0] == pytest.approx(0.5)


def test_window_edges_and_minimum_headlines():
    h = history()
    date = h.index[60]
    d = date.strftime("%Y-%m-%d")
    inside = [{"t": f"in{i}", "s": "x", "p": ts(d) - (6 * 86400) + i, "sc": 1.0} for i in range(3)]      # 6 days back
    outside = [{"t": f"out{i}", "s": "x", "p": ts(d) - (8 * 86400) - i, "sc": -1.0} for i in range(5)]    # 8 days back
    p = ss.symbol_panel(inside + outside, h, [date])
    assert p["n7"].iloc[0] == 3 and p["sent_7"].iloc[0] == pytest.approx(1.0) and p["n30"].iloc[0] == 8
    thin = ss.symbol_panel(inside[:2], h, [date])
    assert math.isnan(thin["sent_7"].iloc[0])                                  # 2 < MIN_N7: no made-up average


def test_targets_start_at_the_next_open_not_the_decision_close():
    h = history()
    date = h.index[60]
    p = ss.symbol_panel([], h, [date])
    i = 60
    assert p["fwd5"].iloc[0] == pytest.approx(h["Close"].iloc[i + 5] / h["Open"].iloc[i + 1] - 1)
    assert p["past20"].iloc[0] == pytest.approx(h["Close"].iloc[i] / h["Close"].iloc[i - 20] - 1)
    end = ss.symbol_panel([], h, [h.index[-3]])
    assert math.isnan(end["fwd20"].iloc[0])                                    # future not known yet -> NaN, not a guess


def test_event_flags_are_built_from_the_same_window():
    h = history()
    date = h.index[60]
    d = date.strftime("%Y-%m-%d")
    items = [{"t": "SEBI probe into promoter", "s": "x", "p": ts(d) - 86400, "sc": -0.6},
             {"t": "Q2 results beat estimates", "s": "y", "p": ts(d) - 2 * 86400, "sc": 0.7}]
    p = ss.symbol_panel(items, h, [date])
    assert p["ev_regulatory"].iloc[0] == 1 and p["ev_results"].iloc[0] == 1 and p["ev_deal"].iloc[0] == 0
    assert p["evs_regulatory"].iloc[0] == pytest.approx(-0.6) and math.isnan(p["evs_deal"].iloc[0])


def synthetic_panel(relation: float, seed=0, dates=80, stocks=40, target="x_fwd20") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = dates * stocks
    panel = pd.DataFrame({"date": np.repeat(pd.date_range("2024-01-05", periods=dates, freq="7D"), stocks),
                          "symbol": np.tile([f"S{i}" for i in range(stocks)], dates),
                          "sent_7": rng.normal(size=n)})
    panel[target] = relation * panel["sent_7"] * 0.02 + rng.normal(0, 0.05, n)
    return panel


def test_planted_relationship_is_detected_and_noise_is_not():
    strong = ss.summarize_series(ss._per_date(synthetic_panel(1.0), "sent_7", "x_fwd20", "ic"))
    noise = ss.summarize_series(ss._per_date(synthetic_panel(0.0, seed=3), "sent_7", "x_fwd20", "ic"))
    assert strong["mean"] > 0.1 and strong["t"] > 3
    assert abs(noise["t"]) < 2.5 and abs(noise["mean"]) < 0.05


def test_t_statistic_uses_only_non_overlapping_dates():
    out = ss.summarize_series(ss._per_date(synthetic_panel(1.0), "sent_7", "x_fwd20", "ic"))
    assert out["dates"] == 80 and out["indep_dates"] == math.ceil(80 / ss.THIN)


def test_top_vs_bottom_quintile_spread_has_the_right_sign():
    out = ss.summarize_series(ss._per_date(synthetic_panel(1.0), "sent_7", "x_fwd20", "spread"))
    assert out["mean"] > 0.01


def test_dates_with_too_few_stocks_are_skipped_not_extrapolated():
    panel = synthetic_panel(1.0, stocks=ss.MIN_CROSS - 1)
    assert ss._per_date(panel, "sent_7", "x_fwd20", "ic").isna().all()


def test_event_flag_effect_compares_flagged_to_unflagged():
    rng = np.random.default_rng(1)
    panel = synthetic_panel(0.0)
    panel["ev_deal"] = (rng.random(len(panel)) < 0.3).astype(float)
    panel["x_fwd20"] = panel["ev_deal"] * 0.05 + rng.normal(0, 0.05, len(panel))
    assert ss.summarize_series(ss._per_date(panel, "ev_deal", "x_fwd20", "flag"))["mean"] == pytest.approx(0.05, abs=0.01)


def test_full_study_flags_a_planted_signal_but_not_the_placebo_alone():
    panel = synthetic_panel(1.0)
    rng = np.random.default_rng(2)
    for h in ss.HORIZONS:
        panel[f"x_fwd{h}"] = rng.normal(0, 0.05, len(panel))
    panel["x_past20"] = panel["sent_7"] * 0.02 + rng.normal(0, 0.05, len(panel))   # news FOLLOWS price only
    for e in ss.EVENTS:
        panel[f"ev_{e}"] = (rng.random(len(panel)) < 0.2).astype(float)
        panel[f"evs_{e}"] = rng.normal(size=len(panel))
    panel["sent_30"] = rng.normal(size=len(panel))
    panel["n7"] = 5
    result = ss.run_study(panel)
    row = next(r for r in result["tests"] if r["feature"] == "sent_7" and r["how"] == "ic")
    assert row["x_past20"]["t"] > 3                 # placebo: sentiment matches the PAST ...
    assert row["significant_future"] == []          # ... but predicts nothing about the future
    assert result["n_primary_tests"] == 15 * len(ss.HORIZONS)             # 3 sentiment tests + 12 event tests, per horizon
    assert result["bonferroni_t"] == pytest.approx(3.26, abs=0.01)        # P(Z > 3.26) = 0.00056 = 5% / 2 / 45
    assert result["coverage"][0]["share_with_sentiment"] == 1.0
