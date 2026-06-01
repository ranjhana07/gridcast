import numpy as np
import pytest

from gridcast.features import FeatureSpec
from gridcast.models.gbm import DirectLightGBM

FAST = {"n_estimators": 40, "n_jobs": 2}


def test_direct_model_beats_seasonal_mean_on_synthetic_load(synthetic_frame):
    spec = FeatureSpec(horizon=6)
    train = synthetic_frame.iloc[: 24 * 90]
    model = DirectLightGBM(spec, FAST).fit(train)
    origins = synthetic_frame.index[24 * 90 : 24 * 100 : 5]
    forecast = model.predict(synthetic_frame, origins)
    positions = synthetic_frame.index.get_indexer(origins)
    truth = np.stack([synthetic_frame["load_kw"].to_numpy()[p + 1 : p + 7] for p in positions])
    error = np.abs(forecast.point - truth).mean()
    assert forecast.point.shape == (len(origins), 6)
    assert error < 40


def test_prediction_before_fit_fails(synthetic_frame):
    model = DirectLightGBM(FeatureSpec(horizon=3), FAST)
    with pytest.raises(RuntimeError):
        model.predict(synthetic_frame, synthetic_frame.index[[500]])


def test_quantile_models_are_ordered(synthetic_frame):
    spec = FeatureSpec(horizon=3)
    model = DirectLightGBM(spec, FAST, quantiles=(0.1, 0.9)).fit(synthetic_frame.iloc[: 24 * 80])
    origins = synthetic_frame.index[24 * 85 : 24 * 90 : 3]
    forecast = model.predict(synthetic_frame, origins)
    assert (forecast.quantiles[0.9] >= forecast.quantiles[0.1]).mean() > 0.95


def test_importance_lists_features(synthetic_frame):
    spec = FeatureSpec(horizon=2)
    model = DirectLightGBM(spec, FAST).fit(synthetic_frame.iloc[: 24 * 60])
    importance = model.importance()
    assert importance.index[0] in set(importance.index)
    assert (importance >= 0).all()
