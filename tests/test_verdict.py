import pytest

from app.research.verdict import validate


def backtest(passed=True, stock_passed=True, t=2.5):
    return {"passed": passed,
            "universe": {"buy": {"n": 100, "edge": 0.01, "t_stat": t}},
            "stocks": {"X.NS": {"passed": stock_passed, "shrunk_return": 0.01, "n": 12}}}


def news(score=0.2, status="ok"):
    return {"status": status, "score": score, "label": "Positive", "n": 20, "note": "thin" if status != "ok" else ""}


def test_non_buy_verdicts_pass_through_untouched():
    for verdict in ("Watchlist", "Hold", "Avoid"):
        assert validate("X.NS", verdict, None, None)[0] == verdict


def test_buy_confirmed_when_backtest_and_news_agree():
    final, v = validate("X.NS", "Buy", backtest(), news())
    assert final == "Buy" and v.status == "validated"
    assert validate("X.NS", "Strong Buy", backtest(), news())[0] == "Strong Buy"


def test_missing_backtest_is_never_a_pass():
    final, v = validate("X.NS", "Buy", None, news())
    assert final == "Watchlist" and v.status == "unvalidated"


def test_universe_without_edge_blocks_even_a_strong_stock_history():
    final, v = validate("X.NS", "Strong Buy", backtest(passed=False, t=0.3), news())
    assert final == "Watchlist" and "beat random entry" in v.summary


def test_stock_missing_from_backtest_blocks():
    assert validate("Y.NS", "Buy", backtest(), news())[0] == "Watchlist"


def test_strongly_negative_news_contradicts_a_validated_buy():
    final, v = validate("X.NS", "Buy", backtest(), news(score=-0.5))
    assert final == "Watchlist" and v.status == "contradicted"


def test_thin_or_missing_news_never_blocks_and_is_labelled():
    for n in (None, news(status="insufficient", score=-0.9)):
        final, v = validate("X.NS", "Buy", backtest(), n)
        assert final == "Buy"
        assert any("not assessed" in g.shown for g in v.gates)


def test_ml_gate_can_veto():
    from app.scoring import Gate
    final, v = validate("X.NS", "Buy", backtest(), news(), Gate("ML model agrees", "p=0.31", False))
    assert final == "Watchlist" and v.status == "contradicted"
