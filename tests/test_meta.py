import numpy as np
import pandas as pd
import pytest

from app.research import meta
from app.research.features import FEATURES
from app.research.ml import ModelError


def synthetic_ledger(kind: str, seed=0, stocks=36, weeks=520) -> pd.DataFrame:
    """Ledger-shaped rows. `kind`: noise | stable (rsi predicts wins everywhere) | flipping (sign flips by year)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-04", periods=weeks, freq="7D")
    n = stocks * weeks
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=list(FEATURES))
    df["date"] = np.repeat(dates, stocks)
    df["symbol"] = np.tile([f"S{i}" for i in range(stocks)], weeks)
    year = df["date"].dt.year.to_numpy()
    if kind == "stable":
        drive = 1.0 * df["rsi"]
    elif kind == "flipping":
        drive = np.where(year % 2 == 0, 1.0, -1.0) * df["rsi"]
    else:
        drive = 0.0 * df["rsi"]
    # a rule fires a quarter of the time; its trade return depends on `drive` (+ noise)
    df["buy"] = rng.random(n) < 0.25
    df["strong"] = False
    df["net_return"] = 0.002 + 0.02 * drive + rng.normal(0, 0.03, n)
    df["fwd"] = df["net_return"]
    df["final"] = True
    day = pd.to_timedelta(1, "D")
    df["o_entry_date"] = (df["date"] + day).dt.strftime("%Y-%m-%d")
    df["o_exit_date"] = (df["date"] + pd.to_timedelta(15, "D")).dt.strftime("%Y-%m-%d")
    df["o_entry"], df["o_exit"], df["o_reason"], df["o_bars"] = 100.0, 100.0, "time", 11
    df["o_net_return"], df["o_r_multiple"] = df["net_return"], df["net_return"] * 20
    return df.drop(columns="net_return")


def test_pruning_drops_a_feature_whose_relationship_flips_over_time():
    df = meta.prepare(synthetic_ledger("flipping"))
    assert "rsi" not in meta.select_features(df)


def test_pruning_keeps_a_stable_feature():
    df = meta.prepare(synthetic_ledger("stable"))
    assert meta.select_features(df)[0] == "rsi"


def test_pruning_never_sees_data_outside_what_it_is_given():
    df = meta.prepare(synthetic_ledger("stable"))
    first_half = df[df["date"] < df["date"].median()]
    assert meta.select_features(first_half) == meta.select_features(first_half.sample(frac=1, random_state=1))


def test_pure_noise_is_not_adopted():
    report = meta.train(synthetic_ledger("noise"))["report"]
    assert report["adopted"] is False and report["why"].startswith("Not adopted")


def test_stable_signal_is_adopted_on_holdout_and_on_unseen_stocks():
    out = meta.train(synthetic_ledger("stable"))
    report = out["report"]
    assert report["adopted"] is True, report["why"]
    assert report["holdout_time"]["filtered"]["t_stat"] > meta.MIN_T
    assert report["holdout_unseen"]["filtered"]["t_stat"] > meta.MIN_T      # generalises to stocks it never trained on
    assert report["columns"][0] == "rsi"
    bundle = out["model"]
    assert 0 < bundle["cut"] < 1 and "rsi" in bundle["columns"]


def test_flipping_signal_is_not_adopted():
    assert meta.train(synthetic_ledger("flipping"))["report"]["adopted"] is False


def test_hard_failures_are_loud():
    with pytest.raises(ModelError):
        meta.train(synthetic_ledger("noise", weeks=150))                     # too little history
    with pytest.raises(ModelError, match="lacks outcome"):
        meta.prepare(synthetic_ledger("noise").drop(columns=["o_net_return"]))


def test_holdout_is_never_used_to_choose_the_model():
    df = meta.prepare(synthetic_ledger("noise"))
    folds = meta.folds_for(df)
    selection = [f for f in folds if f[1] < meta.HOLDOUT_START]
    holdout = [f for f in folds if f[1] >= meta.HOLDOUT_START]
    scored = meta.oos_scores(df, meta.candidates()["logistic"], selection)
    # blocks are contiguous 6-month windows: selection may end exactly where the first hold-out window begins, never later
    assert scored["date"].max() < holdout[0][1]
    assert all(f[2] <= holdout[0][1] for f in selection)


def test_unseen_stock_scoring_never_trains_on_that_stock():
    df = meta.prepare(synthetic_ledger("stable"))
    holdout = [f for f in meta.folds_for(df) if f[1] >= meta.HOLDOUT_START][:1]
    seen = {}
    real = meta.fit_predict

    def spy(factory, train, test):
        seen[tuple(sorted(test["symbol"].unique()))] = set(train["symbol"].unique())
        return real(factory, train, test)

    meta.fit_predict = spy
    try:
        meta.oos_scores(df, meta.candidates()["logistic"], holdout, symbol_holdout=True)
    finally:
        meta.fit_predict = real
    assert seen and all(not (set(test_symbols) & train_symbols) for test_symbols, train_symbols in seen.items())


def test_oos_scores_accepts_a_custom_fit_and_never_trains_on_the_test_window():
    """Research probes (e.g. scripts/deep_sequence_probe.py) plug their own model into this exact protocol."""
    df = meta.prepare(synthetic_ledger("noise", stocks=12, weeks=400))
    folds = meta.folds_for(df)[:2]
    calls = []

    def fit(factory, train, test):
        calls.append(factory)
        assert train["date"].max() < test["date"].min()          # embargoed past only
        return np.full(len(test), 0.7), 0.5, []

    scored = meta.oos_scores(df, "probe", folds, fit=fit)
    assert calls == ["probe"] * len(folds)
    assert (scored["p"] == 0.7).all() and (scored["cut"] == 0.5).all()
