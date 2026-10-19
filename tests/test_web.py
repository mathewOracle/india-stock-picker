"""Smoke tests for the web layer. They use the SQLite cache if present, never the network."""

import pytest
from fastapi.testclient import TestClient

from app import data, main
from app.data import RefreshResult
from app.jobs import RefreshJob
from app.main import app, crore, pct, pe

client = TestClient(app)
needs_cache = pytest.mark.skipif(data.last_refresh() is None, reason="run a refresh first")


def test_formatters_handle_nan_and_loss():
    assert pct(float("nan")) == "–"
    assert pct(0.1234) == "+12.3%"
    assert pe(1000.0) == "loss"
    assert crore(None) == "–"


def test_unknown_symbol_is_404():
    assert client.get("/stock/NOPE.NS").status_code == 404


def test_vendored_assets_are_served():
    for path in ("/static/vendor/htmx.min.js", "/static/vendor/lightweight-charts.js", "/static/charts.js"):
        assert client.get(path).status_code == 200


def test_refresh_failure_shows_error(monkeypatch):
    def broken(progress):
        raise RuntimeError("Could not reach Yahoo Finance")

    monkeypatch.setattr(main, "refresh_job", RefreshJob(broken))
    client.post("/refresh")
    for _ in range(100):
        response = client.get("/refresh/status")
        if "failed" in response.text:
            break
    assert "Refresh failed" in response.text and "HX-Refresh" not in response.headers


def test_refresh_success_triggers_reload(monkeypatch):
    monkeypatch.setattr(main, "refresh_job", RefreshJob(lambda progress: RefreshResult(1, [], 0.1)))
    client.post("/refresh")
    for _ in range(100):
        response = client.get("/refresh/status")
        if response.headers.get("HX-Refresh"):
            break
    assert response.headers.get("HX-Refresh") == "true"


@needs_cache
def test_dashboard_and_table_render():
    page = client.get("/")
    assert page.status_code == 200 and "Suggested buys" in page.text and "Watchlist" in page.text
    assert client.get("/table", params={"verdict": "Avoid"}).status_code == 200


@needs_cache
def test_stock_detail_renders_all_pillars():
    page = client.get("/stock/RELIANCE.NS")
    assert page.status_code == 200
    for section in ("Fundamental analysis", "Technical analysis", "Chart analysis", "Trade plan", "Decision checklist"):
        assert section in page.text
