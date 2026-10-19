"""Research job: 10-year history -> parallel backtest replay -> evidence -> (later) ML training.

Runs as a background job (see app.jobs). Every step either completes or raises with a clear
message; previous results stay cached until a full run succeeds.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app import data
from app.research import ledger, news, news_history, sentiment, sentiment_study
from app.research.backtest import eval_dates, replay_stock
from app.research.evidence import summarize
from app.research.news import normalize
from app.research.ml import MODEL_PATH
from app.universe import BENCHMARK, STOCKS

MIN_HISTORY = 300  # fewer bars than this and a stock can't produce a single evaluation
WORKERS = max(1, min(10, (os.cpu_count() or 2) - 2))


class ResearchError(RuntimeError):
    pass


def _replay(tasks: dict, histories: dict, bench: pd.Series, progress) -> tuple[list[dict], dict, list[str]]:
    """Run the per-stock replays in parallel. Returns (rows, failures, symbols that succeeded)."""
    rows: list[dict] = []
    failures: dict[str, str] = {}
    ok: list[str] = []
    if not tasks:
        return rows, failures, ok
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(replay_stock, s, histories[s], bench, dates): s for s, dates in tasks.items()}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                rows += future.result()
                ok.append(symbol)
            except Exception as exc:  # one broken series must not sink the run -- but it is reported
                failures[symbol] = f"{type(exc).__name__}: {exc}"
            progress(done, len(futures), f"backtest {symbol}")
    return rows, failures, ok


def run_backtest(progress: data.Progress = data._no_progress) -> dict:
    """Extend the permanent ledger with new dates (and settle recent open trades); never redo settled rows."""
    started = time.time()
    data.ensure_history(progress)
    bench_frame = data.load_history(BENCHMARK)
    if bench_frame is None:
        raise ResearchError("No Nifty history available; cannot align evaluation dates.")
    bench = bench_frame["Close"]
    histories = {s: h for s in STOCKS if (h := data.load_history(s)) is not None and len(h) >= MIN_HISTORY}
    skipped = sorted(set(STOCKS) - set(histories))
    if len(histories) < 10:
        raise ResearchError(f"Only {len(histories)} stocks have enough history to backtest.")

    stored, meta = ledger.load()
    plan = ledger.plan_update(stored, meta, histories, bench_frame.index, eval_dates(bench_frame.index))
    rows, failures, ok = _replay(plan.tasks, histories, bench, progress)
    if plan.tasks and not ok:
        raise ResearchError(f"Backtest failed for every stock, e.g. {next(iter(failures.values()), '?')}")
    fresh = ledger.rows_to_frame(rows)
    # a failed symbol keeps its old rows: only successfully recomputed (symbol, date) pairs are replaced
    full = ledger.merge_rows(stored, fresh, {s: plan.tasks[s] for s in ok}) if plan.tasks else stored

    summary = summarize(_group(ledger.frame_to_rows(full)))
    summary.update(skipped=skipped, failures=failures, seconds=round(time.time() - started, 1),
                   generated_at=time.time(),
                   update={"new_dates": len(plan.new_dates), "computed_rows": len(fresh), "restated": plan.restated,
                           "reused_rows": plan.reused, "through": full["date"].max().strftime("%Y-%m-%d")})
    # result is saved BEFORE the ledger swap, and the ledger swap is atomic, so a crash can leave the
    # ledger one run behind the summary but never half-written
    data.save_result("backtest", summary)
    if plan.tasks:
        ledger.save(full, ledger.new_meta(full, histories))
    return summary


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["symbol"], []).append(row)
    return out


def load_dataset() -> pd.DataFrame | None:
    return ledger.load()[0]


def run_model(progress: data.Progress = data._no_progress) -> dict:
    """Train on the saved backtest dataset (so it always matches the stored backtest)."""
    from app.research import meta, ml
    raw = load_dataset()
    if raw is None:
        raise ResearchError("No backtest ledger yet: run the backtest first.")
    progress(0, 1, "training models (walk-forward)")
    try:
        out = meta.train(raw)
    except ml.ModelError as exc:
        raise ResearchError(str(exc)) from exc
    tmp = MODEL_PATH.with_suffix(".tmp")
    ml.save_model(out["model"], tmp)
    data.save_result("model", out["report"])
    os.replace(tmp, MODEL_PATH)
    progress(1, 1, "training models (walk-forward)")
    return out["report"]


# ---------- everything, as one background job ----------

@dataclass
class ResearchResult:
    parts: list[str]
    errors: list[str]

    @property
    def message(self) -> str:
        text = " ".join(self.parts)
        return text + (" Problems: " + " | ".join(self.errors) if self.errors else "")


def run_all(progress: data.Progress = data._no_progress) -> ResearchResult:
    """Backtest -> news -> model. Steps fail independently; old results survive a failed step."""
    parts, errors = [], []
    backtest = news_result = None
    try:
        backtest = run_backtest(progress)
        buy = backtest["universe"]["buy"]
        parts.append(f"Backtest: {buy['n']} trades, t={buy['t_stat']:.2f}, rules "
                     f"{'VALIDATED' if backtest['passed'] else 'not validated'}.")
    except Exception as exc:
        errors.append(f"backtest failed ({exc})")
    try:
        news_result = run_news(progress)
        parts.append(f"News: {len(news_result['stocks'])} stocks scored ({news_result['model']}).")
    except Exception as exc:
        errors.append(f"news failed ({exc})")
    try:   # incremental: closed months are fetched once, so later runs only add the newest weeks
        harvest = news_history.run(progress)
        study = sentiment_study.run(progress)
        parts.append(f"News history: {harvest['windows_fetched']} new windows, study over {study['headlines']} headlines.")
    except Exception as exc:
        errors.append(f"news history/study failed ({exc})")
    if backtest is not None:  # the model trains on the dataset the backtest just wrote
        try:
            report = run_model(progress)
            parts.append(f"Model: {report['best']} {'ADOPTED' if report['adopted'] else 'not adopted'}.")
        except Exception as exc:
            errors.append(f"model failed ({exc})")
    if not parts:
        raise ResearchError("; ".join(errors))
    return ResearchResult(parts, errors)


# ---------- news sentiment ----------

NEWS_WORKERS = 4
NEWS_ATTEMPTS = 3


def _fetch_news(symbol: str, proxy: str | None):
    last: Exception | None = None
    for attempt in range(NEWS_ATTEMPTS):
        try:
            return news.fetch(symbol, proxy)
        except Exception as exc:  # network blips, 429s, bad XML: retry, then report
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{type(last).__name__}: {last}")


def run_news(progress: data.Progress = data._no_progress, classifier=None) -> dict:
    started = time.time()
    classifier = classifier or sentiment.load_classifier()
    previous = data.load_result("news") or {}
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    headlines: dict[str, list] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=NEWS_WORKERS) as pool:
        futures = {pool.submit(_fetch_news, s, proxy): s for s in STOCKS}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                headlines[symbol] = future.result()
            except Exception as exc:
                failures[symbol] = str(exc)
            progress(done, len(futures), f"news {symbol}")
    if not headlines:
        raise ResearchError(f"Could not fetch news for any stock ({next(iter(failures.values()), '?')}). "
                            "Previous sentiment (if any) is kept.")

    # reuse earlier scores only if the SAME model produced them
    same_model = previous.get("model") == classifier.name
    stocks: dict[str, dict] = {}
    for symbol, items in headlines.items():
        cache = {normalize(h["title"]): h["score"] for h in previous.get("stocks", {}).get(symbol, {}).get("headlines", [])} \
            if same_model else {}
        stocks[symbol] = sentiment.aggregate(sentiment.score_headlines(items, classifier, cache), classifier.name)
    result = {"model": classifier.name, "stocks": stocks, "failures": failures,
              "seconds": round(time.time() - started, 1), "generated_at": time.time()}
    data.save_result("news", result)
    return result
