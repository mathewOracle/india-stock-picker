"""Command line: `python -m app.research [backtest|news|train|all]` -- same code the web job runs."""

import sys

from app.research import news_history, pipeline, sentiment_study
from app.research.report import print_backtest


def _progress(done: int, total: int, label: str) -> None:
    if done == 0 or done == total or done % 15 == 0:
        print(f"  [{done}/{total}] {label}", flush=True)


def main(step: str) -> None:
    if step in ("backtest", "all"):
        print_backtest(pipeline.run_backtest(_progress))
    if step == "news":
        print(f"news scored: {len(pipeline.run_news(_progress)['stocks'])} stocks")
    if step == "news_history":
        result = news_history.run(_progress)
        print({k: v for k, v in result.items() if k != "errors"}, f"| {len(result['errors'])} window errors")
    if step == "study":
        study = sentiment_study.run(_progress)
        print(f"study over {study['headlines']} headlines ({', '.join(study['models'])}); Bonferroni |t| >= {study['bonferroni_t']:.2f}")
    if step == "model":
        print(pipeline.run_model(_progress)["why"])
    if step == "all":
        print(pipeline.run_all(_progress).message)


if __name__ == "__main__":  # the guard matters: worker processes re-import this module
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
