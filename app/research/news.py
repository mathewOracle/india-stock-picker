"""Headline collection from Google News RSS (free, no key) with strict relevance + dedupe.

Honest limits: RSS only reaches back a couple of weeks, so news can NOT be backtested -- it is
used as a live caution layer (see app.research.verdict), never as a proven predictor.
"""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from curl_cffi import requests as curl_requests

from app.universe import STOCKS

LOOKBACK_DAYS = 14
FEED = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
FINANCE_TERMS = "(shares OR stock OR results OR earnings OR NSE)"
TIMEOUT = 20

# Headline names that differ from the display name (or are ambiguous words).
ALIASES: dict[str, tuple[str, ...]] = {
    "HINDUNILVR.NS": ("Hindustan Unilever", "HUL"), "LT.NS": ("Larsen & Toubro", "L&T"),
    "M&M.NS": ("Mahindra & Mahindra", "M&M"), "TITAN.NS": ("Titan Company", "Titan Co"),
    "ETERNAL.NS": ("Eternal", "Zomato"), "TMPV.NS": ("Tata Motors Passenger", "Tata Motors"),
    "DMART.NS": ("Avenue Supermarts", "DMart", "D-Mart"), "BEL.NS": ("Bharat Electronics",),
    "HAL.NS": ("Hindustan Aeronautics", "HAL"), "LICI.NS": ("Life Insurance Corporation", "LIC of India", "LIC"),
    "INDIGO.NS": ("IndiGo", "InterGlobe Aviation"), "IOC.NS": ("Indian Oil", "IOC"),
    "JIOFIN.NS": ("Jio Financial",), "SIEMENS.NS": ("Siemens India", "Siemens Ltd"),
    "TRENT.NS": ("Trent Ltd", "Trent share", "Tata Trent"), "ITC.NS": ("ITC Ltd", "ITC share", "ITC shares"),
    "NESTLEIND.NS": ("Nestle India",), "TCS.NS": ("Tata Consultancy Services", "TCS"),
    "VEDL.NS": ("Vedanta",), "SBILIFE.NS": ("SBI Life",), "HDFCLIFE.NS": ("HDFC Life",),
    "BAJAJ-AUTO.NS": ("Bajaj Auto",), "DRREDDY.NS": ("Dr Reddy's", "Dr. Reddy's"),
}


def aliases(symbol: str) -> tuple[str, ...]:
    if symbol in ALIASES:
        return ALIASES[symbol]
    return (re.sub(r"\s*\(.*?\)", "", STOCKS[symbol]).strip(),)


def _quoted(symbol: str) -> str:
    return " OR ".join(f'"{a}"' for a in aliases(symbol)[:2])


def query(symbol: str) -> str:
    return f"({_quoted(symbol)}) {FINANCE_TERMS} when:{LOOKBACK_DAYS}d"


def query_range(symbol: str, start: date, end: date) -> str:
    """Google News date operators: results published on/after `start` and before `end`."""
    return f"({_quoted(symbol)}) {FINANCE_TERMS} after:{start:%Y-%m-%d} before:{end:%Y-%m-%d}"


@dataclass(frozen=True)
class Headline:
    title: str
    source: str
    published: float   # unix seconds (UTC)

    @property
    def key(self) -> str:
        return normalize(self.title)


def normalize(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


# Quote/landing pages that Google News indexes alongside real stories: no information, so drop.
BOILERPLATE = re.compile(r"share price( today)?\b|stock price\b|price today|live nse|quote\b|\bnews, quote", re.I)


def mentions(title: str, names: tuple[str, ...]) -> bool:
    """Whole-word match; short all-caps aliases (ITC, HAL, LIC) must match case-exactly."""
    for name in names:
        flags = 0 if (len(name) <= 4 and name.isupper()) else re.IGNORECASE
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", title, flags):
            return True
    return False


def parse_feed(xml_text: str, names: tuple[str, ...], now: float | None = None,
               max_age_days: float | None = LOOKBACK_DAYS) -> list[Headline]:
    """RSS -> relevant, dated, de-duplicated headlines (newest first). `max_age_days=None` keeps old items
    (historical harvesting). Raises ValueError on bad XML."""
    now = time.time() if now is None else now
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Unparseable news feed: {exc}") from exc
    seen, out = set(), []
    for item in root.iter("item"):
        raw_title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        title = raw_title.removesuffix(f" - {source}").strip() if source else raw_title.rsplit(" - ", 1)[0].strip()
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc).timestamp()
        except (TypeError, ValueError):
            continue  # no trustworthy date -> can't weight it by recency -> drop
        age_days = (now - published) / 86400
        if not title or age_days < -0.1 or (max_age_days is not None and age_days > max_age_days) \
                or not mentions(title, names) or BOILERPLATE.search(title):
            continue
        headline = Headline(title, source or "unknown", published)
        if headline.key not in seen:
            seen.add(headline.key)
            out.append(headline)
    return sorted(out, key=lambda h: -h.published)


def _get(url: str, proxy: str | None) -> str:
    response = curl_requests.get(url, impersonate="chrome", proxy=proxy, timeout=TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"news feed HTTP {response.status_code}")
    return response.text


def fetch(symbol: str, proxy: str | None, now: float | None = None) -> list[Headline]:
    """One network call per stock. Raises on HTTP / parse problems so callers can report them."""
    text = _get(FEED.format(q=quote_plus(query(symbol))), proxy)
    return parse_feed(text, aliases(symbol), now)


TRUNCATED_AT = 90   # Google caps a feed at ~100 items: this many raw items means the window was cut off


def fetch_range(symbol: str, start: date, end: date, proxy: str | None) -> tuple[list[Headline], bool]:
    """Headlines for [start, end). Second value: True if Google probably truncated the window."""
    text = _get(FEED.format(q=quote_plus(query_range(symbol, start, end))), proxy)
    return parse_feed(text, aliases(symbol), max_age_days=None), text.count("<item>") >= TRUNCATED_AT


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M")
