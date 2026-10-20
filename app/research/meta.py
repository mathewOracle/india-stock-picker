"""Meta-labelling model: given the price rule fired, will THIS trade win? (same simulation + costs as the backtest)

Protocol, fixed before looking at any result:
  target     = simulated trade net return > 0 (the backtest's own stop/target/time-exit trade)
  pruning    = per training fold: keep features whose rank-correlation with the target has the SAME sign in all
               3 time slices of the training data, then the top K. No peeking at the test data.
  selection  = choose between candidate models on the SELECTION period only (test starts before HOLDOUT_START)
  validation = (a) final untouched hold-out period, (b) hold-out period again on UNSEEN stocks
               (train on 5/6 of the stocks, test on the other 1/6, rotating)
  adoption   = filtered signal trades (rule fired AND P(win) >= cut) must show n >= MIN_TRADES, edge vs control
               t >= MIN_T, a better average than the unfiltered rule, in >= FOLDS_POSITIVE of folds -- on BOTH (a) and (b)
  cut        = median model score on the training signals (never tuned on test data)
"""

import time

import numpy as np
import pandas as pd

from app.research import ledger
from app.research.backtest import non_overlapping
from app.research.evidence import edge_test, trade_stats
from app.research.features import FEATURES
from app.research.ml import MIN_TRAIN_ROWS, ModelError, walk_forward_folds

TOP_K = 8
SLICES = 3
HOLDOUT_START = pd.Timestamp("2024-11-01")
SYMBOL_GROUPS = 6
MIN_TRADES, MIN_T, FOLDS_POSITIVE, MIN_FOLD_TRADES = 100, 1.65, 0.6, 10
MIN_COVERAGE = 0.8


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a simulated trade outcome, with the win label. Features untouched."""
    missing = [f for f in FEATURES if f not in df.columns]
    if missing or "o_net_return" not in df.columns:
        raise ModelError("Ledger lacks outcome/feature columns: rebuild it with `python -m app.research backtest`.")
    out = df[df["o_entry_date"].notna() & df["final"].astype(bool)].copy()
    out["label"] = (out["o_net_return"] > 0).astype(int)
    out["buy"] = out["buy"].astype(bool)
    return out.sort_values("date").reset_index(drop=True)


def select_features(train: pd.DataFrame, k: int = TOP_K) -> list[str]:
    """Stable-sign pruning + top-k by strength. Uses ONLY the rows it is given (the training fold)."""
    ordered = train.sort_values("date")
    parts = np.array_split(np.arange(len(ordered)), SLICES)
    y = ordered["label"].reset_index(drop=True)
    scored = {}
    for feature in FEATURES:
        x = ordered[feature].reset_index(drop=True)
        if x.notna().mean() < MIN_COVERAGE or x.nunique() < 5:
            continue
        ranks = x.rank()
        cors = [ranks.iloc[p].corr(y.iloc[p]) for p in parts]
        if any(np.isnan(c) for c in cors) or len({np.sign(c) for c in cors}) != 1:
            continue                                        # relationship flips over time: that is noise
        scored[feature] = float(np.mean(np.abs(cors)))
    return sorted(scored, key=scored.get, reverse=True)[:k]


def candidates() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logistic": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(C=0.1, max_iter=1000)),
        "shallow_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=100, min_samples_leaf=300, l2_regularization=10.0, random_state=0),
    }


def fit_predict(factory, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, float, list[str]]:
    """Prune on `train`, fit, return (test scores, cut from training signals, columns used)."""
    columns = select_features(train)
    if not columns:
        raise ModelError("No stable features survived pruning.")
    model = factory().fit(train[columns], train["label"])
    train_signals = train[train["buy"]]
    cut = float(np.median(model.predict_proba(train_signals[columns])[:, 1])) if len(train_signals) else 0.5
    return model.predict_proba(test[columns])[:, 1], cut, columns


def folds_for(df: pd.DataFrame) -> list[tuple]:
    return [f for f in walk_forward_folds(df["date"])
            if len(df[df["date"] < f[0]]) >= MIN_TRAIN_ROWS]


def oos_scores(df: pd.DataFrame, factory, folds: list[tuple], symbol_holdout: bool = False,
               fit=None) -> pd.DataFrame:
    """Out-of-sample scored rows. With symbol_holdout, each test row's model never saw that stock.

    `fit` defaults to fit_predict; research probes pass their own (same signature) to reuse this exact protocol.
    """
    fit = fit or fit_predict
    symbols = sorted(df["symbol"].unique())
    group = {s: i % SYMBOL_GROUPS for i, s in enumerate(symbols)}
    pieces = []
    for fold_id, (train_end, test_start, test_end) in enumerate(folds):
        window = df[(df["date"] >= test_start) & (df["date"] < test_end)]
        if window.empty:
            continue
        history = df[df["date"] < train_end]
        splits = ([(history[history["symbol"].map(group) != g], window[window["symbol"].map(group) == g])
                   for g in range(SYMBOL_GROUPS)] if symbol_holdout else [(history, window)])
        for train, test in splits:
            if len(train) < MIN_TRAIN_ROWS or test.empty or train["label"].nunique() < 2:
                continue
            p, cut, _ = fit(factory, train, test)
            pieces.append(test.assign(p=p, cut=cut, fold=fold_id))
    return pd.concat(pieces) if pieces else pd.DataFrame()


def trades(rows: pd.DataFrame) -> list[dict]:
    """One stock's trades must not overlap in time: reuse the backtest's own rule."""
    out = []
    for symbol, group in rows.groupby("symbol"):
        out += non_overlapping(ledger.frame_to_rows(group.assign(buy=True)), "buy")
    return out


def score_filter(oos: pd.DataFrame) -> dict:
    """Unfiltered rule vs rule + model filter, against the same control group."""
    if oos.empty:
        return {"n": 0}
    control = float(oos["o_net_return"].mean())
    signals = oos[oos["buy"]]
    kept = signals[signals["p"] >= signals["cut"]]
    out = {"control": control}
    for name, rows in (("rule", signals), ("filtered", kept)):
        t = trades(rows)
        edge, t_stat = edge_test(t, control)
        out[name] = {**trade_stats(t), "edge": edge, "t_stat": t_stat}
    fold_edges = []
    for _, rows in kept.groupby("fold"):
        t = trades(rows)
        if len(t) >= MIN_FOLD_TRADES:
            fold_edges.append(np.mean([x["net_return"] for x in t]) - control)
    out["folds_positive"] = float(np.mean([e > 0 for e in fold_edges])) if fold_edges else float("nan")
    out["folds_rated"] = len(fold_edges)
    return out


def verdict(result: dict) -> tuple[bool, list[str]]:
    """The pre-registered adoption rules, applied to one scored block. Returns (passed, failed checks)."""
    if not result.get("n", 1) or "filtered" not in result:
        return False, ["no out-of-sample trades"]
    f, r = result["filtered"], result["rule"]
    checks = {
        f"filtered trades >= {MIN_TRADES} (got {f.get('n', 0)})": f.get("n", 0) >= MIN_TRADES,
        f"edge t-stat >= {MIN_T} (got {f.get('t_stat', float('nan')):.2f})": bool(f.get("t_stat", 0) >= MIN_T),
        f"beats the unfiltered rule ({f.get('avg_return', 0):+.2%} vs {r.get('avg_return', 0):+.2%})":
            f.get("avg_return", -1) > r.get("avg_return", 0),
        f"positive in >= {FOLDS_POSITIVE:.0%} of folds (got {result['folds_positive']:.0%})":
            bool(result["folds_positive"] >= FOLDS_POSITIVE),
    }
    return all(checks.values()), [c for c, ok in checks.items() if not ok]


def train(ledger_df: pd.DataFrame) -> dict:
    """Select on the selection period, validate on hold-out + unseen stocks, refit on everything."""
    df = prepare(ledger_df)
    folds = folds_for(df)
    selection = [f for f in folds if f[1] < HOLDOUT_START]
    holdout = [f for f in folds if f[1] >= HOLDOUT_START]
    if len(selection) < 3 or not holdout:
        raise ModelError("Not enough history for a selection period plus a hold-out period.")
    if any(f[2] > holdout[0][1] for f in selection):   # the guarantee that makes the hold-out honest
        raise ModelError("Selection windows overlap the hold-out period.")
    factories = candidates()
    selection_oos = {name: oos_scores(df, factory, selection) for name, factory in factories.items()}
    chosen_by = {name: score_filter(oos) for name, oos in selection_oos.items()}
    best = max(chosen_by, key=lambda n: np.nan_to_num(chosen_by[n].get("filtered", {}).get("t_stat", -9), nan=-9))
    holdout_oos = oos_scores(df, factories[best], holdout)
    holdout_time = score_filter(holdout_oos)
    holdout_unseen = score_filter(oos_scores(df, factories[best], holdout, symbol_holdout=True))
    ok_time, why_time = verdict(holdout_time)
    ok_unseen, why_unseen = verdict(holdout_unseen)
    adopted = ok_time and ok_unseen
    why = ("Adopted: passed on the untouched hold-out and on unseen stocks." if adopted else
           "Not adopted. Hold-out: " + ("passed" if ok_time else "; ".join(why_time)) +
           ". Unseen stocks: " + ("passed" if ok_unseen else "; ".join(why_unseen)) + ".")

    columns = select_features(df)
    final = factories[best]().fit(df[columns], df["label"])
    signals = df[df["buy"]]
    cut = float(np.median(final.predict_proba(signals[columns])[:, 1]))
    edges = calibration_edges(selection_oos[best])
    report = {"target": "trade_win", "best": best, "adopted": adopted, "why": why, "cut": cut, "columns": columns,
              "selection": {n: _slim(v) for n, v in chosen_by.items()}, "holdout_time": _slim(holdout_time),
              "holdout_unseen": _slim(holdout_unseen), "holdout_start": HOLDOUT_START.strftime("%Y-%m-%d"),
              "calibration": {"selection": calibration(selection_oos[best], edges),
                              "holdout": calibration(holdout_oos, edges)},
              "trained_rows": int(len(df)), "generated_at": time.time(),
              "period": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
              "label": "the rule's simulated trade ends with a net profit (after costs)"}
    return {"model": {"model": final, "columns": columns, "cut": cut}, "report": report}


def _slim(block: dict) -> dict:
    """Drop bulky per-exit counters so the stored report stays small."""
    return {k: ({kk: vv for kk, vv in v.items() if kk != "exits"} if isinstance(v, dict) else v) for k, v in block.items()}


# ---------- calibration: does a score of X% really mean X%? ----------

CALIBRATION_BINS = 5


def calibration_edges(oos: pd.DataFrame) -> list[float]:
    """Score quintiles of the SELECTION period's out-of-sample predictions, reused unchanged for the hold-out."""
    if oos.empty:
        return [0.0, 1.0]
    inner = np.unique(np.quantile(oos["p"], np.linspace(0, 1, CALIBRATION_BINS + 1)[1:-1]))
    return [0.0, *map(float, inner), 1.0]


def calibration(oos: pd.DataFrame, edges: list[float]) -> dict:
    """Out-of-sample reliability: per score bin, how often the trade really won. Reporting only, never tuning."""
    if oos.empty:
        return {"n": 0, "bins": []}
    from sklearn.metrics import roc_auc_score
    which = np.clip(np.searchsorted(edges, oos["p"], side="right") - 1, 0, len(edges) - 2)
    bins = []
    for i in range(len(edges) - 1):
        rows = oos[which == i]
        if len(rows):
            bins.append({"lo": edges[i], "hi": edges[i + 1], "n": int(len(rows)), "mean_p": float(rows["p"].mean()),
                         "win_rate": float(rows["label"].mean()), "avg_return": float(rows["o_net_return"].mean())})
    auc = float(roc_auc_score(oos["label"], oos["p"])) if oos["label"].nunique() == 2 else float("nan")
    return {"n": int(len(oos)), "win_rate": float(oos["label"].mean()), "auc": auc,
            "period": [oos["date"].min().strftime("%Y-%m-%d"), oos["date"].max().strftime("%Y-%m-%d")], "bins": bins}
