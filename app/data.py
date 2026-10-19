"""Fetch market data from Yahoo Finance and cache it in SQLite.

Reads never hit the network; only `refresh()` does. Each refresh builds a brand
new HTTP session: yfinance keeps ONE global session, and in a long-running
server that session can go stale (proxy drops the connection) and then fail
every request forever. A fresh session per refresh sidesteps that entirely.
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

from app.universe import INDICES, STOCKS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.db"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
ATTEMPTS = 2

INFO_FIELDS = (
    "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToBook",
    "enterpriseToEbitda", "dividendYield", "returnOnEquity", "recommendationMean",
    "recommendationKey", "numberOfAnalystOpinions", "targetMeanPrice", "heldPercentInsiders",
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

    @property
    def message(self) -> str:
        msg = f"Updated {self.ok} symbols in {self.seconds:.0f}s."
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


def _save(conn: sqlite3.Connection, key: str, payload: dict) -> None:
    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                 (key, json.dumps(payload), time.time()))


def _load(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


# ---------- network (refresh only) ----------

def _fresh_session() -> curl_requests.Session:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    yf.config.network.proxy = proxy  # explicit beats hoping env vars get picked up
    return curl_requests.Session(impersonate="chrome")


def _clean(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _fetch_symbol(symbol: str, session: curl_requests.Session) -> dict:
    """Everything we need for one symbol. Raises if prices are unavailable."""
    ticker = yf.Ticker(symbol, session=session)
    history = ticker.history(period="2y", auto_adjust=True)[OHLCV].dropna()
    if history.empty:
        raise RefreshError(f"no prices for {symbol}")
    payload = {"ohlcv": {
        "dates": [d.strftime("%Y-%m-%d") for d in history.index],
        **{col: [round(float(v), 2) for v in history[col]] for col in OHLCV},
    }}
    if symbol in STOCKS:
        info = ticker.info
        payload["fundamentals"] = {
            "info": {key: info.get(key) for key in INFO_FIELDS},
            "statements": _statements(ticker),
        }
    return payload


def _fetch_with_retry(symbol: str, session: curl_requests.Session) -> dict | None:
    for attempt in range(ATTEMPTS):
        try:
            return _fetch_symbol(symbol, session)
        except Exception:  # Yahoo throws all sorts; one bad ticker must not sink the batch
            time.sleep(1 + attempt)
    return None


def refresh(progress: Progress | None = None) -> RefreshResult:
    """Re-download prices + fundamentals for every symbol. Raises RefreshError on total failure."""
    started = time.time()
    session = _fresh_session()
    symbols = list(STOCKS) + list(INDICES)
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_with_retry, s, session): s for s in symbols}
        for done, future in enumerate(as_completed(futures), start=1):
            if (payload := future.result()) is not None:
                results[futures[future]] = payload
            if progress:
                progress(done, len(symbols), futures[future])

    if not results:
        raise RefreshError("Could not reach Yahoo Finance. Check VPN/proxy and try again. "
                           "Showing previously cached data.")
    with _connect() as conn:
        for symbol, payload in results.items():
            _save(conn, f"ohlcv:{symbol}", payload["ohlcv"])
            if "fundamentals" in payload:
                _save(conn, f"fundamentals:{symbol}", payload["fundamentals"])
    failed = sorted(set(symbols) - set(results))
    return RefreshResult(ok=len(results), failed=failed, seconds=time.time() - started)


# ---------- cached reads ----------

def _to_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame({c: payload[c] for c in OHLCV}, index=pd.to_datetime(payload["dates"]))


def load_ohlcv(symbol: str) -> pd.DataFrame | None:
    with _connect() as conn:
        payload = _load(conn, f"ohlcv:{symbol}")
    return _to_frame(payload) if payload else None


def load_stocks() -> list[StockData]:
    out = []
    with _connect() as conn:
        for symbol in STOCKS:
            if payload := _load(conn, f"ohlcv:{symbol}"):
                fundamentals = _load(conn, f"fundamentals:{symbol}") or {}
                out.append(StockData(symbol, _to_frame(payload), fundamentals))
    return out


def last_refresh() -> float | None:
    with _connect() as conn:
        return conn.execute("SELECT MAX(fetched_at) FROM cache").fetchone()[0]
