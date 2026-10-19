import numpy as np

from app.indicators import compute_indicators
from app.patterns import (Level, _candles, _structure, _support_resistance, analyze_chart, cluster_levels,
                          pivots, significant_levels, with_extremes)
from app.signals import score
from tests.conftest import make_ohlcv


def test_single_touch_52w_high_still_counts_as_resistance():
    """Regression: a one-off spike high was invisible to the S/R check."""
    closes = list(np.linspace(100, 100, 30)) + [130] + list(np.linspace(110, 129, 30))  # ends 0.5 ATR under the spike
    df = make_ohlcv(closes)
    levels = with_extremes([], df)
    assert any(lv.label == "52w high" for lv in levels)
    signal = _support_resistance(df, atr_now=3.0, levels=levels)
    assert signal.points < 0 and "52w high" in signal.reading


def test_pivots_find_the_peak_and_trough():
    closes = np.concatenate([np.linspace(100, 120, 20), np.linspace(120, 90, 20), np.linspace(90, 110, 20)])
    highs, lows = pivots(make_ohlcv(closes), k=5)
    assert any(abs(p - 120 * 1.005) < 1 for _, p in highs)
    assert any(abs(p - 90 * 0.995) < 1 for _, p in lows)


def test_cluster_levels_groups_nearby_prices():
    levels = cluster_levels([100, 100.5, 101, 120, 120.8])
    assert [lv.touches for lv in levels] == [3, 2]


def test_significant_levels_ignore_noise_near_price():
    levels = [Level(99.8, 3), Level(95, 2), Level(100.3, 2), Level(110, 4)]
    support, resistances = significant_levels(levels, price=100, atr=2)
    assert support == 95          # 99.8 is within 0.5 ATR -> noise
    assert resistances == [110]   # 100.3 is within 1 ATR -> noise


def test_structure_uptrend_and_breakdown():
    up = _structure([(10, 100), (30, 110)], [(20, 90), (40, 95)], close=108)
    assert up.points == 1.0
    breaking = _structure([(10, 100), (30, 110)], [(20, 90), (40, 95)], close=94)
    assert breaking.points < 0 and "below last swing low" in breaking.reading


def test_bullish_engulfing_after_decline():
    closes = list(np.linspace(110, 100, 12)) + [100]
    df = make_ohlcv(closes)
    df.iloc[-2, df.columns.get_loc("Open")], df.iloc[-2, df.columns.get_loc("Close")] = 101, 99    # red candle
    df.iloc[-1, df.columns.get_loc("Open")], df.iloc[-1, df.columns.get_loc("Close")] = 98.5, 102  # engulfs it
    df["High"], df["Low"] = df[["Open", "Close"]].max(axis=1) + 0.2, df[["Open", "Close"]].min(axis=1) - 0.2
    signal = _candles(df)
    assert signal is not None and "Bullish engulfing" in signal.reading


def test_chart_score_separates_trends(uptrend, downtrend):
    bull = score(analyze_chart(compute_indicators(uptrend)).signals)
    bear = score(analyze_chart(compute_indicators(downtrend)).signals)
    assert bull > bear
