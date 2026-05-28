import numpy as np
import pandas as pd
import pytest

from gridcast.features import (
    FeatureSpec,
    horizon_features,
    horizon_target,
    origin_features,
    solar_spec,
    training_matrix,
)

ORIGIN = 24 * 60


def corrupt_after(frame: pd.DataFrame, position: int, columns: list[str]) -> pd.DataFrame:
    damaged = frame.astype(float)
    damaged.iloc[position + 1 :, [damaged.columns.get_loc(c) for c in columns]] = 1e9
    return damaged


@pytest.mark.parametrize("step", [1, 6, 24])
def test_features_ignore_future_target_values(synthetic_frame, step):
    spec = FeatureSpec()
    clean = horizon_features(synthetic_frame, spec, step).iloc[ORIGIN]
    damaged = corrupt_after(synthetic_frame, ORIGIN, ["load_kw"])
    dirty = horizon_features(damaged, spec, step).iloc[ORIGIN]
    assert np.allclose(clean.to_numpy(dtype=float), dirty.to_numpy(dtype=float), equal_nan=True)


def test_origin_features_ignore_all_future_columns(synthetic_frame):
    spec = FeatureSpec()
    clean = origin_features(synthetic_frame, spec).iloc[ORIGIN]
    damaged = corrupt_after(synthetic_frame, ORIGIN, list(synthetic_frame.columns))
    dirty = origin_features(damaged, spec).iloc[ORIGIN]
    assert np.allclose(clean.to_numpy(dtype=float), dirty.to_numpy(dtype=float))


def test_target_is_the_value_at_origin_plus_step(synthetic_frame):
    spec = FeatureSpec()
    target = horizon_target(synthetic_frame, spec, 5)
    assert target.iloc[ORIGIN] == synthetic_frame["load_kw"].iloc[ORIGIN + 5]


def test_seasonal_lag_points_at_known_data(synthetic_frame):
    spec = FeatureSpec()
    for step in (1, 12, 24):
        features = horizon_features(synthetic_frame, spec, step)
        expected = synthetic_frame["load_kw"].iloc[ORIGIN + step - 24]
        assert features["seasonal_24"].iloc[ORIGIN] == expected
        assert ORIGIN + step - 24 <= ORIGIN


def test_short_seasonal_lag_is_rejected():
    with pytest.raises(ValueError):
        FeatureSpec(horizon=24, seasonal_lags=(12,))


def test_step_outside_horizon_is_rejected(synthetic_frame):
    with pytest.raises(ValueError):
        horizon_features(synthetic_frame, FeatureSpec(), 25)


def test_training_matrix_drops_unlabelled_rows(synthetic_frame):
    features, target = training_matrix(synthetic_frame, FeatureSpec(), 24)
    assert len(features) == len(target)
    assert not features.isna().any().any()
    assert features.index.max() <= synthetic_frame.index[-25]


def test_solar_spec_excludes_target_time_weather(synthetic_frame):
    frame = synthetic_frame.assign(sun_elevation=0.5)
    spec = solar_spec()
    names = horizon_features(frame, spec, 3).columns
    assert not any(name.startswith("future_shortwave") for name in names)
    assert "future_sun_elevation" in names
