import numpy as np
import pandas as pd
import pytest

from app import indicators as ind
from app.signals import score
from app.technical import analyze_technical, extension_atr


def test_rsi_extremes():
    rising = pd.Series(np.arange(1, 60, dtype=float))
    falling = rising[::-1].reset_index(drop=True)
    assert ind.rsi(rising).iloc[-1] == pytest.approx(100)
    assert ind.rsi(falling).iloc[-1] == pytest.approx(0)


def test_macd_positive_in_uptrend(uptrend):
    assert ind.macd(uptrend["Close"])["macd"].iloc[-1] > 0


def test_bollinger_bands_bracket_the_mean(uptrend):
    bb = ind.bollinger(uptrend["Close"]).dropna()
    assert (bb["bb_upper"] >= bb["bb_mid"]).all() and (bb["bb_lower"] <= bb["bb_mid"]).all()


def test_atr_is_positive(uptrend):
    assert (ind.atr(uptrend).dropna() > 0).all()


def test_supertrend_direction_follows_trend(uptrend, downtrend):
    assert ind.supertrend(uptrend)["st_dir"].iloc[-1] == 1
    assert ind.supertrend(downtrend)["st_dir"].iloc[-1] == -1


def test_adx_directional_bias(uptrend):
    frame = ind.adx(uptrend)
    assert frame["plus_di"].iloc[-1] > frame["minus_di"].iloc[-1]


def test_compute_indicators_has_every_column(uptrend):
    frame = ind.compute_indicators(uptrend)
    for col in ("ema20", "sma50", "sma200", "rsi", "atr", "obv", "macd_hist", "pct_b", "adx", "stoch_k", "st_dir"):
        assert col in frame


def test_technical_score_separates_trends(uptrend, downtrend):
    bull = score(analyze_technical(ind.compute_indicators(uptrend), None))
    bear = score(analyze_technical(ind.compute_indicators(downtrend), None))
    assert bull > 60 > 40 > bear


def test_relative_strength_needs_benchmark(uptrend, downtrend):
    signals = analyze_technical(ind.compute_indicators(uptrend), downtrend["Close"])
    rs = next(s for s in signals if s.name.startswith("Relative strength"))
    assert rs.points == 1.0


def test_spike_is_flagged_as_extended(uptrend):
    """Regression: a one-day 12% pop should read as 'chasing', not as pure strength."""
    spiked = uptrend.copy()
    spiked.iloc[-1, spiked.columns.get_loc("Close")] *= 1.12
    spiked.iloc[-1, spiked.columns.get_loc("High")] = spiked["Close"].iloc[-1] * 1.005
    frame = ind.compute_indicators(spiked)
    assert extension_atr(frame) > 3
    signal = next(s for s in analyze_technical(frame, None) if s.name == "Extension vs 20-EMA")
    assert signal.tone == "bearish"
