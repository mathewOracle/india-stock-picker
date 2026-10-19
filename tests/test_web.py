"""Smoke tests for the web layer. They use the SQLite cache if present, never the network."""

import pytest
from fastapi.testclient import TestClient

from app import data
from app.main import app, crore, pct, pe

client = TestClient(app)
has_cache = data.last_refresh() is not None
needs_cache = pytest.mark.skipif(not has_cache, reason="run a refresh first")


def test_formatters_handle_nan_and_loss():
    assert pct(float("nan")) == "–"
    assert pct(0.1234) == "+12.3%"
    assert pe(1000.0) == "loss"
    assert crore(None) == "–"


def test_unknown_symbol_is_404():
    assert client.get("/stock/NOPE.NS").status_code == 404


@needs_cache
def test_dashboard_and_table_render():
    page = client.get("/")
    assert page.status_code == 200 and "Top picks" in page.text
    assert client.get("/table", params={"verdict": "Avoid"}).status_code == 200


@needs_cache
def test_stock_detail_renders():
    assert client.get("/stock/RELIANCE.NS").status_code == 200
