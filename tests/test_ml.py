import numpy as np
import pandas as pd
import pytest

from app.research import ml
from app.research.features import FEATURES


def synthetic(signal: float, seed=0, stocks=30, weeks=420, column="score_price") -> pd.DataFrame:
    """Noise features; `signal` controls how much `column` predicts the forward excess return."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-04", periods=weeks, freq="7D")
    n = stocks * weeks
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=list(FEATURES))
    df["date"] = np.repeat(dates, stocks)
    df["symbol"] = np.tile([f"S{i}" for i in range(stocks)], weeks)
    df["fwd"] = 0.02 * (signal * df[column] + rng.normal(size=n)) + 0.004
    return df


def test_embargo_keeps_training_away_from_the_test_window():
    df = ml.prepare(synthetic(0.0))
    folds = ml.walk_forward_folds(df["date"])
    assert len(folds) >= 6
    for train_end, test_start, test_end in folds:
        assert (test_start - train_end).days == ml.EMBARGO_DAYS
        assert train_end > df["date"].min() + pd.DateOffset(years=ml.MIN_TRAIN_YEARS) - pd.Timedelta(days=ml.EMBARGO_DAYS + 1)
        assert test_end > test_start


def test_label_is_cross_sectional_so_market_direction_cannot_leak():
    df = synthetic(0.0)
    df["fwd"] += df["date"].dt.year.map(lambda y: 0.1 if y % 2 else -0.1)   # huge bull/bear year effect
    prepared = ml.prepare(df)
    assert abs(prepared["label"].mean() - 0.5) < 0.02                       # still balanced
    assert abs(prepared.groupby("date")["label"].mean().sub(0.5).abs().max()) <= 0.1


def test_pure_noise_is_not_adopted():
    report = ml.train(synthetic(0.0))["report"]
    best = report["models"][report["best"]]
    assert best["auc"] < 0.54                       # no phantom skill
    assert report["adopted"] is False and report["why"].startswith("Not adopted")


def test_signal_the_rules_do_not_use_is_found_and_adopted():
    out = ml.train(synthetic(1.0, column="ret_3m"))
    report = out["report"]
    assert report["adopted"] is True, report["why"]
    assert report["models"][report["best"]]["auc"] > 0.6
    assert report["models"]["rules_only"]["auc"] < 0.54            # the rules alone are blind to it
    assert report["importance"][0]["feature"] == "ret_3m"
    proba = out["model"].predict_proba(synthetic(1.0, seed=5).head(200)[list(FEATURES)])[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_model_adds_nothing_when_the_existing_rules_already_capture_the_signal():
    report = ml.train(synthetic(1.0, column="score_price"))["report"]
    assert report["models"]["rules_only"]["auc"] > 0.7             # the rule score is genuinely predictive here
    assert report["adopted"] is False and "beats rules-only" in report["why"]


def test_missing_feature_columns_fail_loudly():
    with pytest.raises(ml.ModelError, match="missing feature columns"):
        ml.prepare(synthetic(0.0).drop(columns=["rsi"]))


def test_too_little_history_fails_loudly():
    with pytest.raises(ml.ModelError):
        ml.train(synthetic(0.0, stocks=30, weeks=100))
