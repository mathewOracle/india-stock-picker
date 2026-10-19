import numpy as np
import pandas as pd
import pytest


def make_ohlcv(closes, volume=1_000_000.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "Open": opens,
        "High": np.maximum(opens, closes) * 1.005,
        "Low": np.minimum(opens, closes) * 0.995,
        "Close": closes,
        "Volume": np.full(len(closes), volume),
    }, index=pd.bdate_range("2024-01-01", periods=len(closes)))


@pytest.fixture
def uptrend() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return make_ohlcv(100 * np.cumprod(1 + 0.003 + rng.normal(0, 0.006, 400)))


@pytest.fixture
def downtrend() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return make_ohlcv(100 * np.cumprod(1 - 0.003 + rng.normal(0, 0.006, 400)))
