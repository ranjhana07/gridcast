import numpy as np
import pandas as pd
import pytest

from gridcast.data.validation import ValidationError, fill_gaps, prepare_series, validate_series


def make_series(n: int = 24 * 14) -> pd.Series:
    index = pd.date_range("2024-03-01", periods=n, freq="1h")
    values = 100 + 20 * np.sin(2 * np.pi * index.hour.to_numpy() / 24)
    return pd.Series(values, index=index, name="load_kw")


def test_report_flags_gaps_and_duplicates():
    series = make_series()
    series.iloc[10:20] = np.nan
    series = pd.concat([series, series.iloc[:2]])
    report = validate_series(series, max_missing_share=0.01)
    assert report.duplicates == 2
    assert report.missing == 10
    assert report.longest_gap == 10
    assert not report.ok


def test_index_must_be_datetime():
    with pytest.raises(ValidationError):
        validate_series(pd.Series([1.0, 2.0]))


def test_prepare_fills_short_and_long_gaps():
    series = make_series()
    series.iloc[40:42] = np.nan
    series.iloc[100:130] = np.nan
    series.iloc[200] = 0.0
    prepared = prepare_series(series)
    assert not prepared.isna().any()
    assert len(prepared) == len(series)
    assert prepared.iloc[200] > 50


def test_fill_gaps_uses_daily_pattern():
    series = make_series()
    truth = series.copy()
    series.iloc[60:80] = np.nan
    filled = fill_gaps(series)
    assert np.allclose(filled.iloc[60:80], truth.iloc[60:80], atol=1e-6)
