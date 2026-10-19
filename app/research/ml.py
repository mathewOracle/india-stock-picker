"""ML layer: choose the best base model by WALK-FORWARD out-of-sample results, then refit on all data.

Rules fixed before looking at results (no tuning on the test folds):
  label    = stock beats the same-date median stock over the next HORIZON sessions
  folds    = expanding train window; 6-month test blocks; EMBARGO days purged before each block
  adoption = best model's mean AUC >= MIN_AUC, beats the rules-only baseline by >= MIN_LIFT,
             > MIN_FOLDS_ABOVE of folds above 0.5, and its top-quintile picks have positive excess return
"""

import time

import numpy as np
import pandas as pd

from app import data
from app.research.features import FEATURES

MODEL_PATH = data.DB_PATH.parent / "research" / "model.joblib"

EMBARGO_DAYS = 40            # > 20 sessions of label overlap
MIN_TRAIN_YEARS = 3
TEST_MONTHS = 6
MIN_AUC, MIN_LIFT, MIN_FOLDS_ABOVE = 0.525, 0.01, 0.6
TOP_QUANTILE = 0.2
MIN_TRAIN_ROWS = 3000
RULES_ONLY = ["score_price"]


class ModelError(RuntimeError):
    pass


def candidates() -> dict:
    """Fresh, small, heavily-regularised models (finance data is mostly noise; big models just memorise it)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logistic": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(C=0.05, max_iter=1000)),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=200, l2_regularization=5.0, random_state=0),
        "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=200, min_samples_leaf=100, max_features="sqrt", n_jobs=-1, random_state=0)),
    }


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add label + excess return; drop rows whose future isn't known yet. Never touches features."""
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ModelError(f"Dataset is missing feature columns {missing[:3]}...: re-run the backtest.")
    df = df[np.isfinite(df["fwd"])].copy()
    df["excess"] = df["fwd"] - df.groupby("date")["fwd"].transform("median")
    df["label"] = (df["excess"] > 0).astype(int)
    return df.sort_values("date").reset_index(drop=True)


def walk_forward_folds(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """(train_end, test_start, test_end): train uses dates < train_end = test_start - embargo."""
    start, end = dates.min() + pd.DateOffset(years=MIN_TRAIN_YEARS), dates.max()
    folds, test_start = [], start
    while test_start < end:
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
        folds.append((test_start - pd.Timedelta(days=EMBARGO_DAYS), test_start, test_end))
        test_start = test_end
    return folds


def _probabilities(factory, columns: list[str], train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = factory()
    model.fit(train[columns], train["label"])
    return model.predict_proba(test[columns])[:, 1]


def _score_fold(test: pd.DataFrame, proba: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    scored = test.assign(p=proba)
    cutoff = scored.groupby("date")["p"].transform(lambda p: p.quantile(1 - TOP_QUANTILE))
    top = scored[scored["p"] >= cutoff]
    return {"auc": float(roc_auc_score(test["label"], proba)), "n": int(len(test)),
            "top_excess": float(top["excess"].mean()) if len(top) else float("nan")}


def evaluate(df: pd.DataFrame) -> dict:
    """Walk-forward results for every candidate + the two baselines. Pure function of the data."""
    models = candidates()
    results = {name: [] for name in [*models, "rules_only"]}
    fold_labels = []
    for train_end, test_start, test_end in walk_forward_folds(df["date"]):
        train, test = df[df["date"] < train_end], df[(df["date"] >= test_start) & (df["date"] < test_end)]
        if len(train) < MIN_TRAIN_ROWS or len(test) < 500 or test["label"].nunique() < 2:
            continue
        fold_labels.append(f"{test_start:%Y-%m} to {test_end:%Y-%m}")
        for name, factory in models.items():
            results[name].append(_score_fold(test, _probabilities(factory, list(FEATURES), train, test)))
        results["rules_only"].append(
            _score_fold(test, _probabilities(models["logistic"], RULES_ONLY, train, test)))
    if not fold_labels:
        raise ModelError("Not enough history for even one walk-forward fold.")
    summary = {name: {"folds": folds, "auc": float(np.mean([f["auc"] for f in folds])),
                      "top_excess": float(np.nanmean([f["top_excess"] for f in folds])),
                      "folds_above_chance": float(np.mean([f["auc"] > 0.5 for f in folds]))}
               for name, folds in results.items()}
    return {"fold_labels": fold_labels, "models": summary}


def adopt(summary: dict) -> tuple[str, bool, str]:
    """Pick the best candidate by mean OOS AUC, then decide whether it earns a say in verdicts."""
    best = max((n for n in summary if n != "rules_only"), key=lambda n: summary[n]["auc"])
    b, rules = summary[best], summary["rules_only"]
    checks = {
        f"AUC {b['auc']:.3f} >= {MIN_AUC}": b["auc"] >= MIN_AUC,
        f"beats rules-only ({rules['auc']:.3f}) by >= {MIN_LIFT}": b["auc"] - rules["auc"] >= MIN_LIFT,
        f"{b['folds_above_chance']:.0%} of folds above chance (need > {MIN_FOLDS_ABOVE:.0%})":
            b["folds_above_chance"] > MIN_FOLDS_ABOVE,
        f"top-quintile excess {b['top_excess']:+.2%} > 0": b["top_excess"] > 0,
    }
    failed = [c for c, ok in checks.items() if not ok]
    return best, not failed, ("All adoption checks passed." if not failed else "Not adopted: " + "; ".join(failed))


def train(df: pd.DataFrame) -> dict:
    """Evaluate, choose, refit the winner on ALL labelled data, return {model, report}."""
    df = prepare(df)
    summary = evaluate(df)
    best, adopted, why = adopt(summary["models"])
    final = candidates()[best]()
    final.fit(df[list(FEATURES)], df["label"])
    report = {**summary, "best": best, "adopted": adopted, "why": why, "trained_rows": int(len(df)),
              "period": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
              "label": f"beats same-date median stock over next 20 sessions", "generated_at": time.time(),
              "features": list(FEATURES), "importance": _importance(final, df)}
    return {"model": final, "report": report}


# ---------- persistence + live scoring ----------

PROBA_GATE = 0.55   # when the model is adopted, a buy also needs P(outperform) at least this high
_loaded: tuple[float, object] | None = None


def save_model(model, path) -> None:
    """Local file we wrote ourselves (joblib = pickle: never load model files from elsewhere)."""
    import joblib
    joblib.dump({"model": model, "features": list(FEATURES)}, path)


def load_model(path):
    """Cached by file mtime. None if absent; ModelError if it was trained on different features."""
    global _loaded
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _loaded is None or _loaded[0] != mtime:
        import joblib
        blob = joblib.load(path)
        if blob["features"] != list(FEATURES):
            raise ModelError("Saved model was trained on a different feature set: retrain it.")
        _loaded = (mtime, blob["model"])
    return _loaded[1]


def predict(model, feature_row: dict) -> float:
    frame = pd.DataFrame([feature_row])[list(FEATURES)]
    return float(model.predict_proba(frame)[0, 1])


def active_model():
    """(model or None, note). A model is returned ONLY if the saved report says it was adopted."""
    report = data.load_result("model")
    if not report:
        return None, "No ML model trained yet."
    if not report["adopted"]:
        return None, f"ML model not used: {report['why']}"
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:  # corrupt/mismatched file must never take the dashboard down
        return None, f"ML model unavailable: {exc}"
    return model, "ML model active."


def _importance(model, df: pd.DataFrame, top: int = 10) -> list[dict]:
    """Permutation importance on the most recent 20% of rows (in-sample for the final fit: indicative only)."""
    from sklearn.inspection import permutation_importance
    recent = df.tail(max(1000, len(df) // 5))
    result = permutation_importance(model, recent[list(FEATURES)], recent["label"], scoring="roc_auc",
                                    n_repeats=3, random_state=0, n_jobs=-1)
    order = np.argsort(-result.importances_mean)[:top]
    return [{"feature": FEATURES[i], "importance": float(result.importances_mean[i])} for i in order]
