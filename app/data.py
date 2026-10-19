"""Fetch market data from Yahoo Finance and cache it in SQLite.

Reads never hit the network; only the `refresh*()` functions do.

* Each refresh builds a brand new HTTP session: yfinance keeps ONE global
  session, and in a long-running server it can go stale and fail forever.
* Prices come down in one batched call (fast); fundamentals change quarterly,
  so they are only re-fetched when older than FUNDAMENTALS_TTL.
"""

import json
import math
import os
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests as curl_requests

from app.universe import BENCHMARK, INDICES, STOCKS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.db"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
ATTEMPTS = 2
FUNDAMENTALS_TTL = 20 * 3600
FUNDAMENTALS_KEY = "fundamentals:v2:{}"  # bump the version when INFO_FIELDS/STATEMENT_ROWS change
DAILY_FMT, INTRADAY_FMT = "%Y-%m-%d", "%Y-%m-%d %H:%M"

INFO_FIELDS = (
    "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToBook",
    "enterpriseToEbitda", "dividendYield", "returnOnEquity", "recommendationMean",
    "recommendationKey", "numberOfAnalystOpinions", "targetMeanPrice", "heldPercentInsiders",
    "earningsTimestampStart", "exDividendDate",
)
STATEMENT_ROWS = {
    "income": ("financials", (
        "Total Revenue", "Net Income Common Stockholders", "Net Income", "Gross Profit",
        "Operating Income", "EBIT", "Interest Expense", "Diluted EPS")),
    "balance": ("balance_sheet", (
        "Total Assets", "Common Stock Equity", "Stockholders Equity", "Current Assets",
        "Current Liabilities", "Long Term Debt", "Total Debt", "Ordinary Shares Number")),
    "cashflow": ("cashflow", ("Operating Cash Flow", "Free Cash Flow")),
}

Progress = Callable[[int, int, str], None]


def _no_progress(done: int, total: int, label: str) -> None:
    pass


class RefreshError(RuntimeError):
    """Raised when a refresh fetched nothing usable (cache is left untouched)."""


@dataclass
class StockData:
    symbol: str
    ohlcv: pd.DataFrame
    fundamentals: dict = field(default_factory=dict)


@dataclass
class RefreshResult:
    ok: int
    failed: list[str]
    seconds: float
    fundamentals_updated: int = 0

    @property
    def message(self) -> str:
        msg = f"Updated {self.ok} price series"
        msg += f" and {self.fundamentals_updated} fundamentals" if self.fundamentals_updated else ""
        msg += f" in {self.seconds:.0f}s."
        return msg + (f" Failed: {', '.join(self.failed)}" if self.failed else "")


# ---------- tiny key/value cache ----------

def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache "
        "(key TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at REAL NOT NULL)"
    )
    return conn


def _json_default(value):
    """numpy scalars sneak into results; store them as plain Python numbers."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot store {type(value).__name__} in the cache")


def _save(conn: sqlite3.Connection, key: str, payload: dict) -> None:
    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                 (key, json.dumps(payload, default=_json_default), time.time()))


def _load(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _age(conn: sqlite3.Connection, key: str) -> float | None:
    row = conn.execute("SELECT fetched_at FROM cache WHERE key = ?", (key,)).fetchone()
    return time.time() - row[0] if row else None


def _serialize(frame: pd.DataFrame, fmt: str) -> dict:
    return {"dates": [d.strftime(fmt) for d in frame.index],
            **{col: [round(float(v), 2) for v in frame[col]] for col in OHLCV}}


def _to_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame({c: payload[c] for c in OHLCV}, index=pd.to_datetime(payload["dates"]))


# ---------- network (refresh only) ----------

def _fresh_session() -> curl_requests.Session:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    yf.config.network.proxy = proxy  # explicit beats hoping env vars get picked up
    return curl_requests.Session(impersonate="chrome")


def _retry(fn: Callable, *args):
    """Call fn(*args) up to ATTEMPTS times; None if every attempt raised."""
    for attempt in range(ATTEMPTS):
        try:
            return fn(*args)
        except Exception:  # Yahoo throws all sorts; one bad ticker must not sink the batch
            time.sleep(1 + attempt)
    return None


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[OHLCV].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    frame["Volume"] = frame["Volume"].fillna(0)
    if frame.index.tz is not None:  # keep exchange-local wall-clock time (IST), drop tz
        frame.index = frame.index.tz_localize(None)
    return frame


def _download(symbols: list[str], session: curl_requests.Session, **kwargs) -> dict[str, pd.DataFrame]:
    """One batched call for all symbols, then per-symbol retries for any that came back empty."""
    raw = yf.download(symbols, group_by="ticker", progress=False, threads=True,
                      auto_adjust=True, session=session, **kwargs)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol in raw.columns.get_level_values(0) and not (frame := _clean_frame(raw[symbol])).empty:
            frames[symbol] = frame
    for symbol in set(symbols) - set(frames):
        history = _retry(lambda s: yf.Ticker(s, session=session).history(auto_adjust=True, **kwargs), symbol)
        if history is not None and not (frame := _clean_frame(history)).empty:
            frames[symbol] = frame
    return frames


def _statements(ticker: yf.Ticker) -> dict:
    out = {}
    for section, (attr, rows) in STATEMENT_ROWS.items():
        frame = getattr(ticker, attr)
        if frame is None or frame.empty:
            continue
        out[section] = {
            "years": [int(c.year) for c in frame.columns],
            "rows": {r: [_clean(v) for v in frame.loc[r]] for r in rows if r in frame.index},
        }
    return out


def _clean(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fetch_fundamentals(symbol: str, session: curl_requests.Session) -> dict:
    ticker = yf.Ticker(symbol, session=session)
    info = ticker.info
    return {"info": {key: info.get(key) for key in INFO_FIELDS}, "statements": _statements(ticker)}


def refresh(progress: Progress = _no_progress) -> RefreshResult:
    """Daily prices for everything; fundamentals only where stale. Raises RefreshError on total failure."""
    started, session = time.time(), _fresh_session()
    symbols = list(STOCKS) + list(INDICES)
    progress(0, 1, "daily prices")
    prices = _download(symbols, session, period="2y")
    if not prices:
        raise RefreshError("Could not reach Yahoo Finance. Check VPN/proxy and try again. "
                           "Showing previously cached data.")
    with _connect() as conn:
        for symbol, frame in prices.items():
            _save(conn, f"ohlcv:{symbol}", _serialize(frame, DAILY_FMT))
        stale = [s for s in STOCKS
                 if (age := _age(conn, FUNDAMENTALS_KEY.format(s))) is None or age > FUNDAMENTALS_TTL]

    total = 1 + len(stale)
    progress(1, total, "daily prices")
    fundamentals: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_retry, _fetch_fundamentals, s, session): s for s in stale}
        for done, future in enumerate(as_completed(futures), start=2):
            if (payload := future.result()) is not None:
                fundamentals[futures[future]] = payload
            progress(done, total, futures[future])
    with _connect() as conn:
        for symbol, payload in fundamentals.items():
            _save(conn, FUNDAMENTALS_KEY.format(symbol), payload)

    failed = sorted(set(symbols) - set(prices)) + [f"{s} (fundamentals)" for s in stale if s not in fundamentals]
    return RefreshResult(len(prices), failed, time.time() - started, len(fundamentals))


def refresh_intraday(progress: Progress = _no_progress) -> RefreshResult:
    """5-minute bars for the last 5 sessions (stocks + Nifty) in one batched call."""
    started, session = time.time(), _fresh_session()
    symbols = list(STOCKS) + [BENCHMARK]
    progress(0, 1, "intraday bars")
    frames = _download(symbols, session, period="5d", interval="5m")
    if not frames:
        raise RefreshError("Could not fetch intraday data from Yahoo Finance. Check VPN/proxy.")
    with _connect() as conn:
        for symbol, frame in frames.items():
            _save(conn, f"intraday:{symbol}", _serialize(frame, INTRADAY_FMT))
    progress(1, 1, "intraday bars")
    return RefreshResult(len(frames), sorted(set(symbols) - set(frames)), time.time() - started)


# ---------- cached reads ----------

def _load_frame(key: str) -> pd.DataFrame | None:
    with _connect() as conn:
        payload = _load(conn, key)
    return _to_frame(payload) if payload else None


def load_ohlcv(symbol: str) -> pd.DataFrame | None:
    return _load_frame(f"ohlcv:{symbol}")


def load_intraday(symbol: str) -> pd.DataFrame | None:
    return _load_frame(f"intraday:{symbol}")


def load_stocks() -> list[StockData]:
    out = []
    with _connect() as conn:
        for symbol in STOCKS:
            if payload := _load(conn, f"ohlcv:{symbol}"):
                fundamentals = _load(conn, FUNDAMENTALS_KEY.format(symbol)) or {}
                out.append(StockData(symbol, _to_frame(payload), fundamentals))
    return out


def _latest(*patterns: str) -> float | None:
    clause = " OR ".join("key LIKE ?" for _ in patterns)
    with _connect() as conn:
        return conn.execute(f"SELECT MAX(fetched_at) FROM cache WHERE {clause}", patterns).fetchone()[0]


def last_refresh() -> float | None:
    """When daily prices/fundamentals last changed (intraday refreshes don't count)."""
    return _latest("ohlcv:%", "fundamentals:%")


def last_intraday_refresh() -> float | None:
    return _latest("intraday:%")


# ---------- long history (backtesting) ----------

HISTORY_TTL = 20 * 3600
HISTORY_PERIOD = "10y"        # first download; afterwards history only ever GROWS (see merge_history)
TOP_UP_PERIOD = "3mo"         # incremental download that overlaps the stored tail
TOP_UP_MAX_GAP_DAYS = 45      # stored series older than this is topped up with a full download instead
OVERLAP_MIN = 20              # sessions of overlap needed to trust a merge
RESTATE_TOL = 0.005           # >0.5% drift of the price ratio inside the overlap = Yahoo restated history
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def merge_history(old: pd.DataFrame | None, new: pd.DataFrame, partial: bool) -> pd.DataFrame | None:
    """Extend stored history with a fresh download WITHOUT losing the older days.

    Yahoo's adjusted prices get restated after dividends/splits, so the old rows are rescaled onto the
    new basis using their overlap. For a `partial` (short) download an inconsistent ratio means a restatement
    happened inside it: return None and let the caller fetch the full history instead.
    """
    if old is None or old.empty:
        return new
    overlap = old.index.intersection(new.index)
    if len(overlap) < OVERLAP_MIN:
        return None if partial else new
    ratio = new.loc[overlap, "Close"] / old.loc[overlap, "Close"]
    if partial and ratio.max() / ratio.min() - 1 > RESTATE_TOL:
        return None
    older = old[old.index < new.index[0]].copy()
    older[PRICE_COLUMNS] *= float(ratio.iloc[:5].median())  # the basis at the join
    return pd.concat([older, new]).sort_index()


def ensure_history(progress: Progress = _no_progress) -> int:
    """Keep the long daily history current and COMPLETE. Returns how many series were updated.

    First run: 10 years. After that: a 3-month top-up merged onto what is stored, so the archive keeps
    growing instead of sliding. Kept apart from the 2-year live series so it never slows the daily refresh.
    """
    symbols = list(STOCKS) + [BENCHMARK]
    with _connect() as conn:
        stale = [s for s in symbols if (age := _age(conn, f"history:{s}")) is None or age > HISTORY_TTL]
    if not stale:
        return 0
    progress(0, 1, f"updating price history for {len(stale)} symbols")
    stored = {s: load_history(s) for s in stale}
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=TOP_UP_MAX_GAP_DAYS)
    recent = [s for s in stale if stored[s] is not None and len(stored[s]) >= OVERLAP_MIN * 5
              and stored[s].index[-1] >= cutoff]
    session = _fresh_session()
    merged: dict[str, pd.DataFrame] = {}
    if recent:
        for symbol, frame in _download(recent, session, period=TOP_UP_PERIOD).items():
            if (result := merge_history(stored[symbol], frame, partial=True)) is not None:
                merged[symbol] = result
    redo = [s for s in stale if s not in merged]   # first run, long gap, restated by Yahoo, or failed
    if redo:
        for symbol, frame in _download(redo, session, period=HISTORY_PERIOD).items():
            merged[symbol] = merge_history(stored[symbol], frame, partial=False)
    if not merged:
        raise RefreshError("Could not download price history from Yahoo Finance. Check VPN/proxy.")
    with _connect() as conn:
        for symbol, frame in merged.items():
            _save(conn, f"history:{symbol}", _serialize(frame, DAILY_FMT))
    return len(merged)


def load_history(symbol: str) -> pd.DataFrame | None:
    return _load_frame(f"history:{symbol}")


# ---------- derived results (backtest, news, model reports) ----------

def save_result(key: str, payload: dict) -> None:
    with _connect() as conn:
        _save(conn, f"result:{key}", payload)


def load_result(key: str) -> dict | None:
    with _connect() as conn:
        return _load(conn, f"result:{key}")


def last_research() -> float | None:
    """When any backtest/news/model result last changed (part of the analysis memo key)."""
    return _latest("result:%")
