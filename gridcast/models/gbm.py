from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from gridcast.features import FeatureSpec, horizon_features, origin_features, training_matrix
from gridcast.models.base import Forecast, origin_positions

DEFAULT_PARAMS = {
    "n_estimators": 220,
    "learning_rate": 0.07,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": 2,
    "max_bin": 127,
}


class DirectLightGBM:
    def __init__(
        self,
        spec: FeatureSpec | None = None,
        params: dict | None = None,
        quantiles: tuple[float, ...] = (),
        name: str = "lightgbm",
    ) -> None:
        self.spec = spec or FeatureSpec()
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.quantiles = quantiles
        self.name = name
        self.horizon = self.spec.horizon
        self.models: dict[int, lgb.LGBMRegressor] = {}
        self.quantile_models: dict[tuple[float, int], lgb.LGBMRegressor] = {}

    def fit(self, frame: pd.DataFrame) -> "DirectLightGBM":
        origin = origin_features(frame, self.spec)
        for step in range(1, self.horizon + 1):
            features, target = training_matrix(frame, self.spec, step, origin)
            model = lgb.LGBMRegressor(objective="regression", **self.params)
            model.fit(features, target)
            self.models[step] = model
            for level in self.quantiles:
                quantile_model = lgb.LGBMRegressor(objective="quantile", alpha=level, **self.params)
                quantile_model.fit(features, target)
                self.quantile_models[(level, step)] = quantile_model
        return self

    def predict(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> Forecast:
        if not self.models:
            raise RuntimeError("model has not been fitted")
        positions = origin_positions(frame, origins)
        origin = origin_features(frame, self.spec)
        point = np.empty((len(positions), self.horizon))
        quantiles = {level: np.empty_like(point) for level in self.quantiles}
        for step in range(1, self.horizon + 1):
            features = horizon_features(frame, self.spec, step, origin).iloc[positions]
            point[:, step - 1] = self.models[step].predict(features)
            for level in self.quantiles:
                quantiles[level][:, step - 1] = self.quantile_models[(level, step)].predict(features)
        return Forecast(point=point, quantiles=quantiles)

    def importance(self) -> pd.Series:
        gains = [
            pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
            for model in self.models.values()
        ]
        return pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False)
