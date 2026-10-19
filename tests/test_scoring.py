import numpy as np
import pandas as pd
import pytest

from app.scoring import (
    FACTORS, LOSS_PE, WEIGHTS, fundamental_metrics, price_metrics, score_universe,
)


def _series(daily_return: float, days: int = 300, start: float = 100.0) -> pd.Series:
    return pd.Series(start * (1 + daily_return) ** np.arange(days))


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert set(WEIGHTS) == set(FACTORS)


def test_price_metrics_uptrend_is_above_200dma_with_golden_cross():
    m = price_metrics(_series(0.001))
    assert m["pct_vs_200dma"] > 0
    assert m["golden"] == 1.0
    assert m["off_high"] == pytest.approx(0.0)
    assert m["max_dd"] == pytest.approx(0.0)


def test_price_metrics_short_history_gives_nan_not_crash():
    m = price_metrics(_series(0.001, days=100))
    assert np.isnan(m["pct_vs_200dma"]) and np.isnan(m["ret_12m"])
    assert m["ret_3m"] > 0


def test_loss_making_company_gets_sentinel_pe():
    assert fundamental_metrics({"trailingPE": -5})["pe"] == LOSS_PE


def test_banks_are_exempt_from_debt_to_equity():
    bank = fundamental_metrics({"sector": "Financial Services", "debtToEquity": 800})
    assert np.isnan(bank["de"])
    assert fundamental_metrics({"sector": "Energy", "debtToEquity": 40})["de"] == 40


def _universe() -> pd.DataFrame:
    def row(daily, pe, roe, flag_loss=False):
        base = price_metrics(_series(daily))
        base.update(sector="Tech", pe=LOSS_PE if flag_loss else pe, pb=3.0, roe=roe,
                    margin=0.2, de=30.0, rev_growth=0.1, earn_growth=0.1)
        return base

    return pd.DataFrame({
        "good": row(0.002, 15, 0.30),
        "meh": row(0.0005, 30, 0.12),
        "bad": row(-0.002, 80, 0.02),
        "loser": row(0.002, 10, 0.30, flag_loss=True),
    }).T.infer_objects()


def test_better_fundamentals_and_trend_rank_higher():
    scored = score_universe(_universe())
    assert scored.index[0] in ("good", "loser")
    assert scored.loc["good", "score"] > scored.loc["meh", "score"] > scored.loc["bad", "score"]


def test_scores_are_bounded():
    scored = score_universe(_universe())
    assert scored["score"].between(0, 100).all()


def test_loss_making_stock_never_gets_a_buy_verdict():
    scored = score_universe(_universe())
    assert "Loss-making" in scored.loc["loser", "flags"]
    assert scored.loc["loser", "verdict"] not in ("Buy", "Strong Buy")


def test_missing_data_is_neutral_not_fatal():
    universe = _universe()
    universe["roe"] = np.nan
    scored = score_universe(universe)
    assert (scored["quality"] >= 0).all()
