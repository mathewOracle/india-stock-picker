"""Does news sentiment (or a news EVENT) correlate with returns? -- tested without fooling ourselves.

Per weekly decision date t and stock:
  * features use ONLY headlines published before t (00:00 UTC, i.e. strictly earlier days)
  * targets = return from the NEXT open to the close h sessions later (what an acting trader could earn),
    minus the same-date median stock (removes market direction)
  * placebo = the PAST 20-session return: news usually reports a move that already happened; if sentiment
    correlates with the past but not the future, it describes price rather than predicts it
Stats: per-date Spearman rank IC across stocks; t-stat on every 4th date (non-overlapping 20-session horizons);
Bonferroni-corrected threshold because many tests are run.
"""

import math
import time

import numpy as np
import pandas as pd

from app import data
from app.research import ledger, news_history
from app.research.news_history import EVENTS
from app.universe import STOCKS

MIN_N7, MIN_N30 = 3, 5          # headlines needed in the window for a stock to count that week
MIN_CROSS = 15                  # stocks needed on a date for a cross-sectional statistic
THIN = 4                        # every 4th weekly date: 20-session horizons stop overlapping
HORIZONS = (5, 10, 20)
PAST = 20
ALPHA = 0.05


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(values)])


def symbol_panel(headlines: list[dict], history: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """One stock's features (from news strictly before each date) and targets (from prices)."""
    h = sorted(headlines, key=lambda x: x["p"])
    pub = np.array([x["p"] for x in h])
    score = np.array([x.get("sc", np.nan) for x in h], dtype=float)
    tags = [set(news_history.classify_events(x["t"])) for x in h]
    t_ts = np.array([pd.Timestamp(d).tz_localize("UTC").timestamp() for d in dates])
    hi, lo7, lo30 = (np.searchsorted(pub, t_ts - back * 86400, "left") for back in (0, 7, 30))
    cs = _prefix(np.nan_to_num(score))
    out = pd.DataFrame({"date": dates, "n7": hi - lo7, "n30": hi - lo30})
    out["sent_7"] = np.where(out["n7"] >= MIN_N7, (cs[hi] - cs[lo7]) / out["n7"].clip(lower=1), np.nan)
    out["sent_30"] = np.where(out["n30"] >= MIN_N30, (cs[hi] - cs[lo30]) / out["n30"].clip(lower=1), np.nan)
    for event in EVENTS:
        flag = np.array([event in t for t in tags], dtype=float)
        count = _prefix(flag)[hi] - _prefix(flag)[lo7]
        out[f"ev_{event}"] = (count > 0).astype(float)
        out[f"evs_{event}"] = np.where(count > 0, (_prefix(flag * np.nan_to_num(score))[hi]
                                                    - _prefix(flag * np.nan_to_num(score))[lo7]) / np.maximum(count, 1), np.nan)
    close, opn = history["Close"].to_numpy(), history["Open"].to_numpy()
    pos = history.index.get_indexer(dates)
    ok = pos >= PAST
    out["past20"] = np.where(ok, close[np.where(ok, pos, 0)] / close[np.where(ok, pos - PAST, 0)] - 1, np.nan)
    for hz in HORIZONS:
        valid = (pos >= 0) & (pos + hz < len(close))
        p = np.where(valid, pos, 0)
        out[f"fwd{hz}"] = np.where(valid, close[np.minimum(p + hz, len(close) - 1)] / opn[np.minimum(p + 1, len(close) - 1)] - 1, np.nan)
    return out


def build_panel(archive: dict[str, list[dict]], histories: dict[str, pd.DataFrame], dates: list[pd.Timestamp]) -> pd.DataFrame:
    frames = [symbol_panel(archive.get(s, []), histories[s], dates).assign(symbol=s) for s in histories]
    panel = pd.concat(frames, ignore_index=True)
    for col in [f"fwd{h}" for h in HORIZONS] + ["past20"]:   # cross-sectional excess: market direction removed
        panel[f"x_{col}"] = panel[col] - panel.groupby("date")[col].transform("median")
    return panel


def _per_date(panel: pd.DataFrame, feature: str, target: str, how: str) -> pd.Series:
    """One statistic per date: rank IC | top-vs-bottom quintile spread | flagged-minus-unflagged mean."""
    def stat(g: pd.DataFrame) -> float:
        g = g.dropna(subset=[feature, target]) if how != "flag" else g.dropna(subset=[target])
        if len(g) < MIN_CROSS:
            return np.nan
        if how == "ic":
            return g[feature].rank().corr(g[target].rank())
        if how == "spread":
            q = g[feature].rank(pct=True)
            return g.loc[q > 0.8, target].mean() - g.loc[q <= 0.2, target].mean()
        on, off = g.loc[g[feature] > 0, target], g.loc[g[feature] == 0, target]
        return on.mean() - off.mean() if len(on) >= 3 and len(off) >= 3 else np.nan
    return panel.groupby("date")[[feature, target]].apply(stat)


def summarize_series(series: pd.Series) -> dict:
    """Mean over all dates + t-stat over non-overlapping dates (honest degrees of freedom)."""
    s = series.dropna()
    thin = s.iloc[::THIN]
    n = len(thin)
    t = float(thin.mean() / (thin.std(ddof=1) / math.sqrt(n))) if n > 2 and thin.std(ddof=1) > 0 else float("nan")
    return {"mean": float(s.mean()) if len(s) else float("nan"), "dates": int(len(s)), "indep_dates": int(n), "t": t}


def coverage(panel: pd.DataFrame) -> list[dict]:
    rows = []
    for year, g in panel.groupby(panel["date"].dt.year):
        rows.append({"year": int(year), "share_with_sentiment": float(g["sent_7"].notna().mean()),
                     "median_headlines_7d": float(g["n7"].median())})
    return rows


def run_study(panel: pd.DataFrame) -> dict:
    tests = []
    signed = [("sent_7", "ic"), ("sent_7", "spread"), ("sent_30", "ic")]
    events = [(f"ev_{e}", "flag") for e in EVENTS] + [(f"evs_{e}", "ic") for e in EVENTS]
    for feature, how in signed + events:
        row = {"feature": feature, "how": how}
        for target in [f"x_fwd{h}" for h in HORIZONS] + ["x_past20"]:
            row[target] = summarize_series(_per_date(panel, feature, target, how))
        tests.append(row)
    primary = len(signed + events) * len(HORIZONS)                  # past-return placebo is not an alpha test
    from scipy.stats import norm
    bonf_t = float(norm.isf(ALPHA / 2 / primary))
    for row in tests:
        row["significant_future"] = [t for t in (f"x_fwd{h}" for h in HORIZONS)
                                     if abs(row[t]["t"]) >= bonf_t] if not math.isnan(row["x_fwd20"]["t"]) else []
    return {"tests": tests, "bonferroni_t": bonf_t, "n_primary_tests": primary, "coverage": coverage(panel)}


def run(progress=lambda d, t, label: None) -> dict:
    started = time.time()
    archive = news_history.load_all()
    if not archive:
        raise RuntimeError("No historical news archive yet: run `python -m app.research news_history` first.")
    stored, _ = ledger.load()
    if stored is None:
        raise RuntimeError("No backtest ledger yet.")
    histories = {s: h for s in STOCKS if (h := data.load_history(s)) is not None}
    first = pd.Timestamp(news_history.ARCHIVE_START) + pd.Timedelta(days=30)
    dates = sorted(d for d in stored["date"].unique() if d >= first)
    progress(0, 1, "building sentiment panel")
    panel = build_panel(archive, histories, dates)
    result = run_study(panel)
    models = {h.get("m") for hs in archive.values() for h in hs}
    result.update(period=[dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")], stocks=len(archive),
                  headlines=sum(len(v) for v in archive.values()), models=sorted(m for m in models if m),
                  seconds=round(time.time() - started, 1), generated_at=time.time())
    data.save_result("sentiment_study", result)
    progress(1, 1, "building sentiment panel")
    return result
