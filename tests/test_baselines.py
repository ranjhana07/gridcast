import numpy as np
import pandas as pd
import pytest

from gridcast.models.ets import ETSForecaster
from gridcast.models.naive import SeasonalNaive


def daily_wave(days: int = 40) -> pd.DataFrame:
    index = pd.date_range("2024-02-01", periods=24 * days, freq="1h", name="timestamp")
    rng = np.random.default_rng(1)
    load = 200 + 40 * np.sin(2 * np.pi * index.hour.to_numpy() / 24) + rng.normal(0, 1, len(index))
    return pd.DataFrame({"load_kw": load}, index=index)


def test_naive_rejects_period_shorter_than_horizon():
    frame = daily_wave()
    model = SeasonalNaive(horizon=24, period=12)
    with pytest.raises(ValueError):
        model.predict(frame, frame.index[[500]])


def test_naive_copies_previous_week():
    frame = daily_wave(30)
    origin = frame.index[[400]]
    forecast = SeasonalNaive(period=168).predict(frame, origin)
    expected = frame["load_kw"].iloc[400 + 1 - 168 : 400 + 25 - 168].to_numpy()
    assert np.allclose(forecast.point[0], expected)


def test_ets_tracks_daily_cycle():
    frame = daily_wave()
    origin = frame.index[[24 * 30 - 1]]
    forecast = ETSForecaster(horizon=24, window_days=14).predict(frame, origin)
    truth = frame["load_kw"].iloc[24 * 30 : 24 * 31].to_numpy()
    assert forecast.point.shape == (1, 24)
    assert np.mean(np.abs(forecast.point[0] - truth)) < 5
