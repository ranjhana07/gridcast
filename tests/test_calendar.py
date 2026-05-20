import pandas as pd

from gridcast.data.calendar import calendar_features


def test_weekend_and_holiday_flags():
    index = pd.date_range("2024-12-24", periods=24 * 4, freq="1h")
    features = calendar_features(index, "PT")
    christmas = features.loc["2024-12-25":"2024-12-25"]
    assert (christmas["is_holiday"] == 1).all()
    assert (features.loc["2024-12-24":"2024-12-24", "is_holiday"] == 0).all()
    saturday = features.loc["2024-12-28":"2024-12-28"]
    assert (saturday["is_weekend"] == 1).all()


def test_cyclic_features_bounded():
    index = pd.date_range("2024-01-01", periods=48, freq="1h")
    features = calendar_features(index, "PT")
    assert features["hour_sin"].between(-1, 1).all()
    assert features["hour"].iloc[0] == 0
