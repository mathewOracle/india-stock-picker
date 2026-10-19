import math

import pytest

from app.fundamentals import LOSS_PE, analyze_fundamentals, cagr, fundamental_metrics, piotroski


def _statements(improving: bool = True) -> dict:
    """Two-year statement set where every Piotroski test passes (or fails)."""
    new, old = (1.0, 0.0) if improving else (0.0, 1.0)

    def pair(good, bad):  # newest first
        return [good, bad] if improving else [bad, good]

    return {
        "income": {"rows": {
            "Net Income": [120 if improving else -10, 100],
            "Total Revenue": pair(1200, 1000),
            "Gross Profit": pair(600, 450),
            "Operating Income": pair(200, 150),
            "EBIT": [210, 160], "Interest Expense": [10, 10], "Diluted EPS": pair(12, 10),
        }},
        "balance": {"rows": {
            "Total Assets": [1000, 1000],
            "Long Term Debt": pair(100, 200),
            "Current Assets": pair(400, 300),
            "Current Liabilities": [200, 200],
            "Ordinary Shares Number": pair(100, 110),
            "Common Stock Equity": [600, 550], "Total Debt": [150, 250],
        }},
        "cashflow": {"rows": {"Operating Cash Flow": [150 if improving else -5, 90],
                              "Free Cash Flow": [100, 80, 60, 40]}},
    }


def test_cagr_basic_and_guards():
    assert cagr([133.1, 121, 110, 100]) == pytest.approx(0.10, abs=1e-3)
    assert math.isnan(cagr([-5, 10]))  # negative endpoint -> undefined
    assert cagr([121, float("nan"), 100]) == pytest.approx(0.10, abs=1e-3)  # skips missing middle year


def test_piotroski_perfect_and_terrible():
    assert piotroski(_statements(True)) == (9, 9)
    passed, available = piotroski(_statements(False))
    assert available == 9 and passed <= 1


def test_metrics_and_bank_exemptions():
    metrics = fundamental_metrics({"info": {"sector": "Industrials", "trailingPE": 20}, "statements": _statements()})
    assert metrics["de"] == pytest.approx(0.25)
    assert metrics["interest_cover"] == pytest.approx(21)
    bank = fundamental_metrics({"info": {"sector": "Financial Services"}, "statements": _statements()})
    assert math.isnan(bank["de"]) and bank["piotroski_max"] == 0


def test_loss_maker_gets_sentinel_pe():
    metrics = fundamental_metrics({"info": {"trailingPE": -4}, "statements": {}})
    assert metrics["pe"] == LOSS_PE


def test_strong_company_scores_bullish():
    metrics = fundamental_metrics({"info": {"sector": "Industrials", "trailingPE": 15}, "statements": _statements()})
    signals = analyze_fundamentals(metrics, sector_pe=25, price=100)
    assert sum(s.impact for s in signals) > 0
    assert any(s.name == "P/E vs sector" and s.tone == "bullish" for s in signals)
