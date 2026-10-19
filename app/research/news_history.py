"""Historical news archive: monthly windows per stock, fetched ONCE, then only extended.

* A closed month is done once fetched (after a 5-day grace for late-indexed stories); the current
  month is re-fetched every run, so the archive grows on top of itself.
* Windows Google truncates (~100 items) are split in halves recursively, so busy months are not cut off.
* Every headline keeps its publish time (UTC) so studies can use only news published BEFORE a decision date.
* Scores are stored with the model name; a different model rescoring never mixes with old scores.
"""

import re
import time
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

from app import data
from app.research import news
from app.research.news import Headline, normalize

ARCHIVE_START = date(2023, 10, 1)   # ~3 years: enough for a meaningful test, ~2.3k requests once
GRACE_DAYS = 5
MIN_SPLIT_DAYS = 3
ATTEMPTS = 3
WORKERS = 6
KEY = "newsarch:{}"

# Event taxonomy (keyword rules, deliberately transparent). Order matters only for readability.
EVENTS = {
    "results": r"\b(q[1-4]\b|quarterly|results?|earnings|net profit|revenue|ebitda|margins?)\b",
    "deal": r"\b(order|contract|wins?|bags?|acqui\w+|merger|stake|takeover|buyout|partnership|ipo|listing)\b",
    "regulatory": r"\b(sebi|probe|raid|fraud|penalty|fine[ds]?|lawsuit|court|ban|notice|investigation|default|violation)\b",
    "rating": r"\b(upgrade[ds]?|downgrade[ds]?|target price|initiates? coverage|brokerage|buy rating|sell rating|overweight|underweight)\b",
    "payout": r"\b(dividend|buyback|bonus issue|stock split)\b",
    "management": r"\b(ceo|cfo|chairman|md|resigns?|appoints?|steps down|successor)\b",
}
EVENT_RES = {name: re.compile(pattern, re.I) for name, pattern in EVENTS.items()}


def classify_events(title: str) -> list[str]:
    return [name for name, regex in EVENT_RES.items() if regex.search(title)]


def month_windows(today: date, start: date = ARCHIVE_START) -> list[tuple[date, date]]:
    """[first of month, first of next month) windows from `start` up to and including the current month."""
    out, y, m = [], start.year, start.month
    while (y, m) <= (today.year, today.month):
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        out.append((date(y, m, 1), nxt))
        y, m = nxt.year, nxt.month
    return out


def window_done(info: dict | None, end: date, now: float) -> bool:
    """A closed month counts as done once fetched after its end + grace; open months are always redone."""
    if not info:
        return False
    closes = datetime(end.year, end.month, end.day, tzinfo=timezone.utc).timestamp() + GRACE_DAYS * 86400
    return info["fetched"] >= closes


def _retry(fn, *args):
    last = None
    for attempt in range(ATTEMPTS):
        try:
            return fn(*args)
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{type(last).__name__}: {last}")


def fetch_adaptive(symbol: str, start: date, end: date, proxy: str | None) -> list[Headline]:
    headlines, truncated = _retry(news.fetch_range, symbol, start, end, proxy)
    if truncated and (end - start).days > MIN_SPLIT_DAYS:
        mid = start + (end - start) / 2
        merged = {h.key: h for h in headlines}
        for lo, hi in ((start, mid), (mid, end)):
            merged.update({h.key: h for h in fetch_adaptive(symbol, lo, hi, proxy)})
        return list(merged.values())
    return headlines


def harvest_symbol(symbol: str, proxy: str | None, today: date | None = None, now: float | None = None) -> dict:
    """Fetch only the windows not yet done; returns {'payload', 'fetched', 'errors'} (not yet saved)."""
    today, now = today or date.today(), now or time.time()
    payload = data.load_result(KEY.format(symbol)) or {"windows": {}, "headlines": []}
    seen = {normalize(h["t"]) for h in payload["headlines"]}
    errors, fetched = {}, 0
    for start, end in month_windows(today):
        label = f"{start:%Y-%m}"
        if window_done(payload["windows"].get(label), end, now):
            continue
        try:
            items = fetch_adaptive(symbol, start, end, proxy)
        except Exception as exc:    # leave the window undone: it is retried next run
            errors[label] = str(exc)
            continue
        for h in items:
            if h.key not in seen:
                seen.add(h.key)
                payload["headlines"].append({"t": h.title, "s": h.source, "p": h.published})
        payload["windows"][label] = {"fetched": now, "n": len(items)}
        fetched += 1
    return {"payload": payload, "fetched": fetched, "errors": errors}


def score_payload(payload: dict, classifier) -> int:
    """Score headlines lacking a score from THIS model. Returns how many were (re)scored."""
    todo = [h for h in payload["headlines"] if h.get("m") != classifier.name]
    if todo:
        for h, (p, n, _) in zip(todo, classifier.classify([h["t"] for h in todo])):
            h["sc"], h["m"] = round(float(p - n), 4), classifier.name
    return len(todo)


def run(progress=lambda d, t, label: None, classifier=None, symbols=None) -> dict:
    """Harvest + score every stock. Windows fail independently; saved results are never lost."""
    import os

    from app.research.sentiment import load_classifier
    from app.universe import STOCKS
    symbols = symbols or list(STOCKS)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    classifier = classifier or load_classifier()
    started, errors, fetched_windows, scored = time.time(), {}, 0, 0
    saved = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(harvest_symbol, s, proxy): s for s in symbols}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                result = future.result()
                fetched_windows += result["fetched"]
                errors.update({f"{symbol} {w}": e for w, e in result["errors"].items()})
                scored += score_payload(result["payload"], classifier)
                data.save_result(KEY.format(symbol), result["payload"])   # saved NOW: a later crash loses nothing
                saved += 1
            except Exception as exc:
                errors[symbol] = str(exc)
            progress(done, len(futures), f"news history {symbol}")
    if not saved:
        raise RuntimeError("Historical news harvest failed for every stock.")
    return {"symbols": saved, "windows_fetched": fetched_windows, "headlines_scored": scored,
            "model": classifier.name, "errors": errors, "seconds": round(time.time() - started, 1)}


def load_all(symbols=None) -> dict[str, list[dict]]:
    from app.universe import STOCKS
    out = {}
    for symbol in symbols or STOCKS:
        payload = data.load_result(KEY.format(symbol))
        if payload and payload["headlines"]:
            out[symbol] = payload["headlines"]
    return out
