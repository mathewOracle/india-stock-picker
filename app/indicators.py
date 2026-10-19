"""Pure technical-indicator math on OHLCV DataFrames. No opinions here, just numbers."""

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), n)
    loss = _wilder(-delta.clip(upper=0), n)
    return 100 - 100 / (1 + gain / loss)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n).std()
    upper, lower = mid + k * std, mid - k * std
    return pd.DataFrame({
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "pct_b": (close - lower) / (upper - lower),
        "bandwidth": (upper - lower) / mid,
    })


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift()
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up, down = df["High"].diff(), -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = _wilder(true_range(df), n)
    plus_di = 100 * _wilder(plus_dm, n) / tr
    minus_di = 100 * _wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": _wilder(dx, n)})


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
    low, high = df["Low"].rolling(k).min(), df["High"].rolling(k).max()
    pct_k = 100 * (df["Close"] - low) / (high - low)
    return pd.DataFrame({"stoch_k": pct_k, "stoch_d": pct_k.rolling(d).mean()})


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """The Indian trader's favourite trailing-stop indicator. direction: +1 up, -1 down."""
    hl2 = ((df["High"] + df["Low"]) / 2).to_numpy()
    band = mult * atr(df, n).to_numpy()
    close = df["Close"].to_numpy()
    upper, lower = hl2 + band, hl2 - band
    final_up, final_lo = upper.copy(), lower.copy()
    direction = np.ones(len(df))
    for i in range(1, len(df)):
        final_up[i] = upper[i] if upper[i] < final_up[i - 1] or close[i - 1] > final_up[i - 1] else final_up[i - 1]
        final_lo[i] = lower[i] if lower[i] > final_lo[i - 1] or close[i - 1] < final_lo[i - 1] else final_lo[i - 1]
        if direction[i - 1] < 0 and close[i] > final_up[i]:
            direction[i] = 1
        elif direction[i - 1] > 0 and close[i] < final_lo[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    line = np.where(direction > 0, final_lo, final_up)
    return pd.DataFrame({"st_line": line, "st_dir": direction}, index=df.index)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV in, OHLCV + every indicator column out. Single source of truth for charts too."""
    close = df["Close"]
    out = df.copy()
    out["ema20"], out["sma50"], out["sma200"] = ema(close, 20), sma(close, 50), sma(close, 200)
    out["rsi"] = rsi(close)
    out["atr"] = atr(df)
    out["obv"] = obv(df)
    out["vol_avg50"] = sma(df["Volume"], 50)
    return pd.concat([out, macd(close), bollinger(close), adx(df), stochastic(df), supertrend(df)], axis=1)
