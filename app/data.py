"""Fetch market data from Yahoo Finance and cache it in SQLite.

Reads never hit the network; only `refresh()` does. That keeps the UI snappy
and means a flaky VPN can't take the whole app down.
"""

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf

from app.universe import INDICES, STOCKS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.db"

FUNDAMENTAL_FIELDS = (
    "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToBook",
    "returnOnEquity", "profitMargins", "debtToEquity", "revenueGrowth",
    "earningsGrowth", "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
)


@dataclass
class StockData:
    symbol: str
    close: pd.Series
    fundamentals: dict = field(default_factory=dict)


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
    conn.execute(
        "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
        (key, json.dumps(payload), time.time()),
    )


def _load(conn: sqlite3.Connection, key: str) -> dict | None:
    row = conn.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


# ---------- network (refresh only) ----------

def _download_prices(symbols: list[str], period: str = "2y") -> dict[str, pd.Series]:
    raw = yf.download(
        symbols, period=period, auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )
    series: dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            close = raw[symbol]["Close"].dropna()
        except KeyError:
            continue
        if not close.empty:
            series[symbol] = close
    return series


def _first_row(frame: pd.DataFrame, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        if label in frame.index:
            values = frame.loc[label].dropna()
            if not values.empty:
                return float(values.iloc[0])  # statements are newest-first
    return None


def _computed_roe(ticker: yf.Ticker) -> float | None:
    """Yahoo leaves returnOnEquity blank for many NSE tickers; derive it from statements."""
    income = _first_row(ticker.financials, ("Net Income Common Stockholders", "Net Income"))
    equity = _first_row(ticker.balance_sheet, ("Common Stock Equity", "Stockholders Equity"))
    return income / equity if income is not None and equity and equity > 0 else None


def _download_info(symbol: str) -> tuple[str, dict | None]:
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.info
        info = {key: raw.get(key) for key in FUNDAMENTAL_FIELDS}
        if info["returnOnEquity"] is None:
            info["returnOnEquity"] = _computed_roe(ticker)
    except Exception:  # Yahoo throws all sorts; one bad ticker must not sink the batch
        return symbol, None
    return symbol, info


def refresh() -> dict[str, int]:
    """Re-download prices (stocks + indices) and fundamentals. Returns counts."""
    symbols = list(STOCKS) + list(INDICES)
    prices = _download_prices(symbols)
    with ThreadPoolExecutor(max_workers=8) as pool:
        infos = dict(pool.map(_download_info, STOCKS))

    with _connect() as conn:
        for symbol, close in prices.items():
            _save(conn, f"prices:{symbol}", {
                "dates": [d.strftime("%Y-%m-%d") for d in close.index],
                "close": [round(float(v), 2) for v in close],
            })
        good_infos = {s: i for s, i in infos.items() if i and i.get("sector")}
        for symbol, info in good_infos.items():
            _save(conn, f"fund:{symbol}", info)
    return {"prices": len(prices), "fundamentals": len(good_infos)}


# ---------- cached reads ----------

def load_close(symbol: str) -> pd.Series | None:
    with _connect() as conn:
        payload = _load(conn, f"prices:{symbol}")
    if not payload:
        return None
    return pd.Series(payload["close"], index=pd.to_datetime(payload["dates"]), name=symbol)


def load_stocks() -> list[StockData]:
    out = []
    with _connect() as conn:
        for symbol in STOCKS:
            payload = _load(conn, f"prices:{symbol}")
            if not payload:
                continue
            close = pd.Series(payload["close"], index=pd.to_datetime(payload["dates"]))
            out.append(StockData(symbol, close, _load(conn, f"fund:{symbol}") or {}))
    return out


def last_refresh() -> float | None:
    with _connect() as conn:
        row = conn.execute("SELECT MAX(fetched_at) FROM cache").fetchone()
    return row[0]
