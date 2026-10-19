"""Refresh logic against a throwaway database, with Yahoo mocked out (no network)."""

import pandas as pd
import pytest

from app import data
from tests.conftest import make_ohlcv


@pytest.fixture
def fake_yahoo(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DB_PATH", tmp_path / "cache.db")
    monkeypatch.setattr(data, "_fresh_session", lambda: None)
    calls = {"download": 0, "fundamentals": 0}

    def download(symbols, session, **kwargs):
        calls["download"] += 1
        return {s: make_ohlcv([100 + i for i in range(30)]) for s in symbols}

    def fundamentals(symbol, session):
        calls["fundamentals"] += 1
        return {"info": {"sector": "Tech"}, "statements": {}}

    monkeypatch.setattr(data, "_download", download)
    monkeypatch.setattr(data, "_fetch_fundamentals", fundamentals)
    return calls


def test_fundamentals_are_cached_between_refreshes(fake_yahoo):
    first = data.refresh()
    assert first.fundamentals_updated == len(data.STOCKS)
    second = data.refresh()
    assert second.fundamentals_updated == 0               # the optimisation: no refetch within the TTL
    assert fake_yahoo["download"] == 2                     # but prices always refresh
    assert fake_yahoo["fundamentals"] == len(data.STOCKS)


def test_stale_fundamentals_are_refetched(fake_yahoo, monkeypatch):
    data.refresh()
    monkeypatch.setattr(data, "FUNDAMENTALS_TTL", -1)      # everything is now "stale"
    assert data.refresh().fundamentals_updated == len(data.STOCKS)


def test_total_failure_raises_and_keeps_cache(fake_yahoo, monkeypatch):
    data.refresh()
    monkeypatch.setattr(data, "_download", lambda symbols, session, **kw: {})
    with pytest.raises(data.RefreshError):
        data.refresh()
    assert data.load_ohlcv("RELIANCE.NS") is not None      # old data survives


def test_intraday_refresh_does_not_bump_daily_timestamp(fake_yahoo):
    data.refresh()
    daily_stamp = data.last_refresh()
    data.refresh_intraday()
    assert data.last_refresh() == daily_stamp
    assert data.last_intraday_refresh() is not None
    assert isinstance(data.load_intraday("RELIANCE.NS"), pd.DataFrame)
