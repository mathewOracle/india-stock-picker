"""Append-only archive of 5-minute bars: Yahoo only serves ~60 sessions, so we keep every bar we ever fetch.

Files live outside the SQLite cache (one gz CSV per symbol) so a year of bars does not bloat the database.
New bars are merged by timestamp (newest download wins), so refreshing repeatedly never duplicates or loses data.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app import data
from app.data import INTRADAY_FMT, OHLCV, RefreshResult
from app.universe import BENCHMARK, STOCKS

ARCHIVE_DIR = data.DB_PATH.parent / "intraday"
BACKFILL_PERIOD = "60d"      # Yahoo's maximum for 5-minute bars
MAX_GAP_SESSIONS = 4         # the routine 5-day refresh can bridge this many missed sessions; more needs a backfill


def path(symbol: str) -> Path:
    return ARCHIVE_DIR / f"{symbol.replace('^', '_')}.csv.gz"


def merge_bars(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new.sort_index()
    both = pd.concat([old, new])
    return both[~both.index.duplicated(keep="last")].sort_index()   # keep="last": the newest download wins


def load(symbol: str) -> pd.DataFrame | None:
    p = path(symbol)
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)[OHLCV]


def append(symbol: str, frame: pd.DataFrame) -> int:
    """Merge `frame` into the symbol's archive atomically. Returns how many bars are new."""
    old = load(symbol)
    merged = merge_bars(old, frame[OHLCV])
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path(symbol).with_suffix(".tmp.gz")
    merged.to_csv(tmp, date_format=INTRADAY_FMT)
    os.replace(tmp, path(symbol))
    return len(merged) - (0 if old is None else len(old))


def sessions_covered() -> tuple[int, str | None, str | None]:
    """(number of sessions, first, last) in the benchmark's archive: the headline coverage figure."""
    frame = load(BENCHMARK)
    if frame is None or frame.empty:
        return 0, None, None
    days = sorted({d.date() for d in frame.index})
    return len(days), days[0].isoformat(), days[-1].isoformat()


def needs_backfill() -> bool:
    frame = load(BENCHMARK)
    if frame is None or frame.empty:
        return True
    gap = len(pd.bdate_range(frame.index[-1].normalize(), pd.Timestamp.now().normalize())) - 1
    return gap > MAX_GAP_SESSIONS


def backfill(progress=data._no_progress) -> int:
    """Download the maximum 5-minute window and merge it into the archive."""
    symbols = list(STOCKS) + [BENCHMARK]
    progress(0, 1, "backfilling 5-minute history")
    frames = data._download(symbols, data._fresh_session(), period=BACKFILL_PERIOD, interval="5m")
    if not frames:
        raise data.RefreshError("Could not backfill 5-minute history from Yahoo Finance. Check VPN/proxy.")
    added = sum(append(symbol, frame) for symbol, frame in frames.items())
    progress(1, 1, "backfilling 5-minute history")
    return added


@dataclass
class ArchiveResult:
    refreshed: RefreshResult
    new_bars: int
    sessions: int
    first: str | None
    last: str | None

    @property
    def message(self) -> str:
        return (f"{self.refreshed.message} Archive: {self.sessions} sessions ({self.first} to {self.last}), "
                f"{self.new_bars} new bars.")


def refresh(progress=data._no_progress) -> ArchiveResult:
    """The 'intraday' job: refresh the live 5-day view, then fold those bars into the permanent archive."""
    symbols = list(STOCKS) + [BENCHMARK]
    added = backfill(progress) if needs_backfill() else 0
    refreshed = data.refresh_intraday(progress)
    added += sum(append(s, frame) for s in symbols if (frame := data.load_intraday(s)) is not None)
    count, first, last = sessions_covered()
    return ArchiveResult(refreshed, added, count, first, last)
