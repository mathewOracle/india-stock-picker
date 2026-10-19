import numpy as np
import pandas as pd
import pytest

from app import data


def frame(start, n, base=100.0, step=0.5) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=n)
    close = base + step * np.arange(n)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000.0}, index=index)


def test_merge_keeps_older_days_that_rolled_out_of_the_new_download():
    old = frame("2020-01-01", 300)
    new = frame(old.index[250], 100, base=float(old["Close"].iloc[250]))   # overlaps the tail, extends it
    merged = data.merge_history(old, new, partial=True)
    assert merged.index[0] == old.index[0] and merged.index[-1] == new.index[-1]
    assert merged.index.is_unique and merged.index.is_monotonic_increasing
    assert len(merged) == 250 + 100


def test_merge_rescales_old_rows_onto_the_restated_basis():
    old = frame("2020-01-01", 300)
    new = frame(old.index[250], 100, base=float(old["Close"].iloc[250]))
    new[["Open", "High", "Low", "Close"]] *= 0.98       # a dividend restated everything by -2% (uniformly)
    merged = data.merge_history(old, new, partial=True)
    join = merged.index.get_loc(new.index[0])
    assert merged["Close"].iloc[join - 1] / merged["Close"].iloc[join] == pytest.approx(
        old["Close"].iloc[249] / old["Close"].iloc[250], rel=1e-9)   # no artificial jump at the seam
    assert merged["Close"].iloc[0] == pytest.approx(old["Close"].iloc[0] * 0.98)


def test_partial_download_spanning_a_split_is_rejected_for_a_full_refetch():
    old = frame("2020-01-01", 300)
    new = frame(old.index[250], 100, base=float(old["Close"].iloc[250]))
    new.iloc[30:, :4] *= 0.5                                  # 2:1 split inside the window: ratio no longer constant
    assert data.merge_history(old, new, partial=True) is None
    assert data.merge_history(old, new, partial=False) is not None   # a full download is merged, not refused


def test_too_little_overlap_cannot_be_trusted_in_a_partial_merge():
    old, new = frame("2020-01-01", 100), frame("2021-06-01", 100)
    assert data.merge_history(old, new, partial=True) is None
    assert data.merge_history(None, new, partial=True) is new
