"""Research pages and the research job: empty states, partial failures, no network."""

import pytest
from fastapi.testclient import TestClient

from app import analysis, data, main
from app.jobs import RefreshJob
from app.research import pipeline

client = TestClient(main.app)


def test_research_page_has_honest_empty_states(monkeypatch):
    monkeypatch.setattr(analysis, "_memo", None)   # restored on teardown: no leaking research-free analysis
    monkeypatch.setattr(data, "load_result", lambda key: None)
    page = client.get("/research")
    assert page.status_code == 200
    assert page.text.count("Not run yet") >= 2 and "Not trained yet" in page.text
    assert "How we avoid fooling ourselves" in page.text


def test_research_page_renders_real_results_when_present():
    if data.load_result("backtest") is None:
        pytest.skip("run `python -m app.research all` first")
    page = client.get("/research").text
    for text in ("Buy gates (price only)", "By year", "HOLD-OUT, unseen stocks", "Pruning:", "Ledger through"):
        assert text in page
    import re
    assert re.search(r"Ledger through <strong>20\d\d-\d\d-\d\d</strong>", page)   # a real date, never blank


def test_dashboard_states_validation_status():
    page = client.get("/").text
    assert "Validation:" in page and "/research" in page


def test_research_job_is_registered_and_reports_errors(monkeypatch):
    assert "research" in main.jobs

    def broken(progress):
        raise pipeline.ResearchError("backtest failed (no history)")

    monkeypatch.setitem(main.jobs, "research", RefreshJob(broken))
    client.post("/refresh/research")
    for _ in range(100):
        text = client.get("/refresh/research/status").text
        if "failed" in text:
            break
    assert "Refresh failed" in text and "no history" in text


def _stub_history(monkeypatch, fail=False):
    """The history-news + study step does network and takes minutes: never run it for real in tests."""
    def harvest(progress):
        if fail:
            raise RuntimeError("archive down")
        return {"windows_fetched": 3, "headlines_scored": 10}
    monkeypatch.setattr(pipeline.news_history, "run", harvest)
    monkeypatch.setattr(pipeline.sentiment_study, "run", lambda progress: {"headlines": 99})


def _stub_intraday(monkeypatch, fail=False):
    """The intraday backtest needs the 5-minute archive on disk: never run it for real in tests."""
    def run(progress):
        if fail:
            raise RuntimeError("no intraday archive")
        return {"trades": 50, "setups": {"orb:Long": {"status": "supported"}, "orb:Short": {"status": "not supported"}}}
    monkeypatch.setattr(pipeline.intraday_backtest, "run", run)


def test_run_all_keeps_going_when_one_step_fails(monkeypatch):
    _stub_history(monkeypatch)
    _stub_intraday(monkeypatch)
    bt = {"universe": {"buy": {"n": 10, "t_stat": 0.5}}, "passed": False}
    monkeypatch.setattr(pipeline, "run_backtest", lambda progress: bt)
    monkeypatch.setattr(pipeline, "run_news", lambda progress: (_ for _ in ()).throw(RuntimeError("feed down")))
    monkeypatch.setattr(pipeline, "run_model", lambda progress: {"best": "logistic", "adopted": False})
    result = pipeline.run_all()
    assert "Backtest" in result.message and "Model" in result.message and "News history" in result.message
    assert "Intraday backtest" in result.message
    assert result.errors == ["news failed (feed down)"] and "Problems" in result.message


def test_history_step_failure_is_reported_but_does_not_stop_the_model(monkeypatch):
    _stub_history(monkeypatch, fail=True)
    _stub_intraday(monkeypatch)
    monkeypatch.setattr(pipeline, "run_backtest", lambda progress: {"universe": {"buy": {"n": 1, "t_stat": 0.1}}, "passed": False})
    monkeypatch.setattr(pipeline, "run_news", lambda progress: {"stocks": {}, "model": "loughran-mcdonald"})
    monkeypatch.setattr(pipeline, "run_model", lambda progress: {"best": "logistic", "adopted": False})
    result = pipeline.run_all()
    assert result.errors == ["news history/study failed (archive down)"] and "Model" in result.message


def test_intraday_backtest_failure_is_reported_but_does_not_stop_the_model(monkeypatch):
    """A fresh install has no intraday archive yet: that step fails alone, everything else still runs."""
    _stub_history(monkeypatch)
    _stub_intraday(monkeypatch, fail=True)
    monkeypatch.setattr(pipeline, "run_backtest", lambda progress: {"universe": {"buy": {"n": 1, "t_stat": 0.1}}, "passed": False})
    monkeypatch.setattr(pipeline, "run_news", lambda progress: {"stocks": {}, "model": "loughran-mcdonald"})
    monkeypatch.setattr(pipeline, "run_model", lambda progress: {"best": "logistic", "adopted": False})
    result = pipeline.run_all()
    assert result.errors == ["intraday backtest failed (no intraday archive)"] and "Model" in result.message


def test_run_all_raises_only_when_everything_fails(monkeypatch):
    _stub_history(monkeypatch, fail=True)
    _stub_intraday(monkeypatch, fail=True)
    def boom(progress):
        raise RuntimeError("no network")

    for step in ("run_backtest", "run_news"):
        monkeypatch.setattr(pipeline, step, boom)
    with pytest.raises(pipeline.ResearchError, match="backtest failed"):
        pipeline.run_all()


def test_model_is_skipped_if_backtest_failed(monkeypatch):
    _stub_history(monkeypatch)
    _stub_intraday(monkeypatch)
    called = []
    monkeypatch.setattr(pipeline, "run_backtest", lambda progress: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(pipeline, "run_news", lambda progress: {"stocks": {}, "model": "loughran-mcdonald"})
    monkeypatch.setattr(pipeline, "run_model", lambda progress: called.append(1))
    pipeline.run_all()
    assert called == []   # never train on a dataset that doesn't match the stored backtest
