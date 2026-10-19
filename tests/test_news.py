import pytest

from app.research import news, sentiment
from app.research.news import Headline, mentions, parse_feed
from app.research.sentiment import MIN_HEADLINES, WordList, aggregate, score_headlines

NOW = 1_800_000_000.0  # fixed clock so tests don't age


def rss(*items: tuple[str, str, float]) -> str:
    from email.utils import formatdate
    from html import escape
    body = "".join(f"<item><title>{escape(t)} - {s}</title><source url='x'>{s}</source>"
                   f"<pubDate>{formatdate(ts, usegmt=True)}</pubDate></item>" for t, s, ts in items)
    return f"<rss><channel>{body}</channel></rss>"


DAY = 86400


def test_parse_keeps_only_relevant_recent_unique_headlines():
    xml = rss(("Titan Company shares surge on jewellery demand", "Mint", NOW - 1 * DAY),
              ("Titan Company shares surge on jewellery demand", "Reuters", NOW - 1 * DAY),   # duplicate
              ("Saturn's moon Titan has methane lakes", "BBC", NOW - 1 * DAY),               # wrong Titan
              ("Titan Company old story", "Mint", NOW - 30 * DAY),                            # too old
              ("Titan Company results preview", "ET", NOW - 2 * DAY))
    out = parse_feed(xml, news.aliases("TITAN.NS"), NOW)
    assert [h.title for h in out] == ["Titan Company shares surge on jewellery demand", "Titan Company results preview"]
    assert out[0].source == "Mint"  # source suffix stripped from the title, newest first


def test_short_uppercase_aliases_are_case_exact():
    assert mentions("ITC Ltd posts record profit", ("ITC Ltd",))
    assert not mentions("Switch to ITCHY sweaters", ("ITC Ltd",))
    assert not mentions("I love itc", ("ITC",))        # short all-caps alias: case-exact
    assert mentions("ITC rallies", ("ITC",))
    assert mentions("HAL bags order", ("HAL",))
    assert not mentions("Hal Varian says", ("HAL",))
    assert not mentions("CHALLENGER bank", ("HAL",))      # whole word only


def test_quote_pages_are_not_news():
    xml = rss(("Trent Ltd. Share Price Today - Trent Ltd. Stock Price Live NSE/BSE", "CNBC", NOW - DAY),
              ("ITC LTD (ITC6.BO) Stock Price, News, Quote & History", "Yahoo", NOW - DAY),
              ("Trent Ltd profit jumps 30% on store expansion", "Mint", NOW - DAY))
    assert [h.source for h in parse_feed(xml, ("Trent Ltd", "ITC Ltd"), NOW)] == ["Mint"]


def test_bad_xml_and_bad_dates_are_handled_not_crashed():
    with pytest.raises(ValueError):
        parse_feed("<rss><channel><item>", ("X",), NOW)
    bad_date = "<rss><channel><item><title>Titan Company up - Mint</title><pubDate>garbage</pubDate></item></channel></rss>"
    assert parse_feed(bad_date, ("Titan Company",), NOW) == []   # no trustworthy date -> dropped


def test_every_stock_has_a_usable_query():
    from app.universe import STOCKS
    for symbol in STOCKS:
        assert news.aliases(symbol) and all(news.aliases(symbol)) and news.query(symbol).count("(") >= 1


def test_wordlist_classifier_basics():
    pos, neg, neu = zip(*WordList().classify(["Profit surges to record high", "Shares plunge on fraud probe", "Board meeting on Monday"]))
    assert pos[0] > neg[0] and neg[1] > pos[1] and neu[2] == 1.0


def _scored(scores, sources=("A", "B", "C", "D", "E", "F"), age_days=1.0):
    return [sentiment.Scored(f"h{i}", sources[i % len(sources)], NOW - age_days * DAY, s) for i, s in enumerate(scores)]


def test_aggregate_labels_and_thresholds():
    ok = aggregate(_scored([0.8] * 6), "finbert", NOW)
    assert ok["status"] == "ok" and ok["label"] == "Positive"
    assert aggregate(_scored([-0.8] * 6), "finbert", NOW)["label"] == "Negative"
    assert aggregate(_scored([0.1, -0.1] * 3), "finbert", NOW)["label"] == "Neutral / mixed"


def test_thin_evidence_is_flagged_not_trusted():
    few = aggregate(_scored([0.9] * (MIN_HEADLINES - 1)), "finbert", NOW)
    assert few["status"] == "insufficient" and "too thin" in few["note"]
    one_outlet = aggregate(_scored([0.9] * 8, sources=("Only",)), "finbert", NOW)
    assert one_outlet["status"] == "insufficient"
    empty = aggregate([], "finbert", NOW)
    assert empty["status"] == "insufficient" and empty["score"] is None


def test_recent_headlines_outweigh_old_ones():
    scored = [sentiment.Scored(f"new{i}", f"S{i}", NOW - 0.2 * DAY, -0.9) for i in range(3)] + \
             [sentiment.Scored(f"old{i}", f"T{i}", NOW - 13 * DAY, +0.9) for i in range(3)]
    assert aggregate(scored, "finbert", NOW)["score"] < -0.5


def test_cached_scores_are_reused_and_new_ones_computed():
    class Spy:
        name = "spy"
        calls: list[list[str]] = []
        def classify(self, titles):
            self.calls.append(list(titles))
            return [(0.9, 0.0, 0.1)] * len(titles)
    spy = Spy()
    a, b = Headline("Alpha beats estimates", "S", NOW), Headline("Beta misses estimates", "S", NOW)
    out = score_headlines([a, b], spy, cache={a.key: -0.5})
    assert spy.calls == [["Beta misses estimates"]]                      # only the new headline hit the model
    assert [round(s.score, 2) for s in out] == [-0.5, 0.9]
