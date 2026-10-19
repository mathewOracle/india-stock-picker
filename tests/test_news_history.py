from datetime import date, datetime, timezone

import pytest

from app.research import news, news_history as nh
from app.research.news import Headline

NOW = datetime(2026, 10, 19, tzinfo=timezone.utc).timestamp()
TODAY = date(2026, 10, 19)


def H(title, day="2024-01-10", source="Mint"):
    ts = datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()
    return Headline(title, source, ts)


def test_month_windows_cover_start_to_current_month_contiguously():
    windows = nh.month_windows(TODAY)
    assert windows[0][0] == nh.ARCHIVE_START and windows[-1][0] == date(2026, 10, 1)
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))            # no gaps, no overlaps
    assert nh.month_windows(date(2024, 1, 15), start=date(2023, 11, 1))[-1] == (date(2024, 1, 1), date(2024, 2, 1))
    assert nh.month_windows(date(2023, 12, 5), start=date(2023, 12, 1)) == [(date(2023, 12, 1), date(2024, 1, 1))]


def test_closed_month_is_done_once_fetched_after_grace_but_open_month_never():
    end = date(2024, 2, 1)
    ts = lambda d: datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp()
    assert not nh.window_done(None, end, NOW)
    assert not nh.window_done({"fetched": ts("2024-02-02")}, end, NOW)         # fetched inside the grace period
    assert nh.window_done({"fetched": ts("2024-02-10")}, end, NOW)
    assert not nh.window_done({"fetched": NOW}, date(2026, 11, 1), NOW)        # current month: always refetch


def test_truncated_windows_are_split_until_small_enough(monkeypatch):
    calls = []

    def fake(symbol, start, end, proxy):
        calls.append((start, end))
        span = (end - start).days
        title = f"Reliance story {start.isoformat()}"
        return [H(title, start.isoformat())], span > 10          # "truncated" while the window is wider than 10 days

    monkeypatch.setattr(news, "fetch_range", fake)
    items = nh.fetch_adaptive("RELIANCE.NS", date(2024, 1, 1), date(2024, 2, 1), None)
    assert len(calls) > 1 and all((e - s).days <= 31 for s, e in calls)
    assert len({h.key for h in items}) == len(items) > 1                      # merged without duplicates


def test_split_stops_at_the_minimum_window(monkeypatch):
    monkeypatch.setattr(news, "fetch_range", lambda *a: ([H("x")], True))      # always "truncated"
    nh.fetch_adaptive("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 4), None)  # must terminate


def test_harvest_is_incremental_closed_months_are_not_refetched(monkeypatch):
    store = {}
    monkeypatch.setattr(nh.data, "load_result", lambda key: store.get(key))
    fetched = []

    def fake(symbol, start, end, proxy):
        fetched.append(start)
        return [H(f"Reliance news {start}", start.isoformat())], False

    monkeypatch.setattr(news, "fetch_range", fake)
    first = nh.harvest_symbol("RELIANCE.NS", None, TODAY, NOW)
    store[nh.KEY.format("RELIANCE.NS")] = first["payload"]
    n_first = len(fetched)
    assert n_first == len(nh.month_windows(TODAY)) and not first["errors"]

    fetched.clear()
    later = NOW + 3600                                                         # an hour later: only the open month is redone
    second = nh.harvest_symbol("RELIANCE.NS", None, TODAY, later)
    assert fetched == [date(2026, 10, 1)]
    assert len(second["payload"]["headlines"]) == len(first["payload"]["headlines"]) + 0   # same stories, no duplicates


def test_failed_window_is_left_undone_and_reported(monkeypatch):
    monkeypatch.setattr(nh.data, "load_result", lambda key: None)
    monkeypatch.setattr(nh.time, "sleep", lambda s: None)

    def flaky(symbol, start, end, proxy):
        if start == date(2024, 3, 1):
            raise RuntimeError("HTTP 429")
        return [H(f"Reliance {start}", start.isoformat())], False

    monkeypatch.setattr(news, "fetch_range", flaky)
    out = nh.harvest_symbol("RELIANCE.NS", None, TODAY, NOW)
    assert "2024-03" in out["errors"] and "2024-03" not in out["payload"]["windows"]
    assert "2024-04" in out["payload"]["windows"]                              # the other months still succeeded


def test_scores_never_mix_models():
    class Fake:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def classify(self, titles):
            return [(self.value, 0.0, 1 - self.value)] * len(titles)

    payload = {"headlines": [{"t": "a", "s": "x", "p": 1.0}, {"t": "b", "s": "x", "p": 2.0}]}
    assert nh.score_payload(payload, Fake("m1", 0.9)) == 2
    assert nh.score_payload(payload, Fake("m1", 0.1)) == 0                     # already scored by m1: untouched
    assert payload["headlines"][0]["sc"] == pytest.approx(0.9)
    assert nh.score_payload(payload, Fake("m2", 0.2)) == 2                     # different model: rescored, labelled
    assert {h["m"] for h in payload["headlines"]} == {"m2"} and payload["headlines"][0]["sc"] == pytest.approx(0.2)


@pytest.mark.parametrize("title,expected", [
    ("Q2 net profit rises 12%", {"results"}), ("SEBI notice to promoter", {"regulatory"}),
    ("Brokerage upgrades target price", {"rating"}), ("Wins Rs 500 crore order", {"deal"}),
    ("Board approves dividend and buyback", {"payout"}), ("CEO resigns", {"management"}),
    ("Stock moves sideways today", set())])
def test_event_classification(title, expected):
    assert set(nh.classify_events(title)) == expected
