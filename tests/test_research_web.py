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
    for text in ("Buy gates (price only)", "By year", "Mean AUC (out-of-sample)", "walk-forward"):
        assert text in page


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


def test_run_all_keeps_going_when_one_step_fails(monkeypatch):
    bt = {"universe": {"buy": {"n": 10, "t_stat": 0.5}}, "passed": False}
    monkeypatch.setattr(pipeline, "run_backtest", lambda progress: bt)
    monkeypatch.setattr(pipeline, "run_news", lambda progress: (_ for _ in ()).throw(RuntimeError("feed down")))
    monkeypatch.setattr(pipeline, "run_model", lambda progress: {"best": "logistic", "adopted": False})
    result = pipeline.run_all()
    assert "Backtest" in result.message and "Model" in result.message
    assert result.errors == ["news failed (feed down)"] and "Problems" in result.message


def test_run_all_raises_only_when_everything_fails(monkeypatch):
    def boom(progress):
        raise RuntimeError("no network")

    for step in ("run_backtest", "run_news"):
        monkeypatch.setattr(pipeline, step, boom)
    with pytest.raises(pipeline.ResearchError, match="backtest failed"):
        pipeline.run_all()


def test_model_is_skipped_if_backtest_failed(monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, "run_backtest", lambda progress: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(pipeline, "run_news", lambda progress: {"stocks": {}, "model": "wordlist"})
    monkeypatch.setattr(pipeline, "run_model", lambda progress: called.append(1))
    pipeline.run_all()
    assert called == []   # never train on a dataset that doesn't match the stored backtest
