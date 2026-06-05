from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from gridcast.conformal import ConformalizedQuantile, SplitConformal
from gridcast.models.base import Forecast, Forecaster, origin_positions


def realised(frame: pd.DataFrame, target: str, positions: np.ndarray, horizon: int) -> np.ndarray:
    values = frame[target].to_numpy()
    steps = np.arange(1, horizon + 1)
    return values[positions[:, None] + steps[None, :]]


class ConformalForecaster:
    def __init__(
        self,
        factory: Callable[[], Forecaster],
        alpha: float = 0.2,
        calibration_hours: int = 24 * 42,
        stride: int = 3,
        name: str | None = None,
    ) -> None:
        self.inner = factory()
        self.alpha = alpha
        self.calibration_hours = calibration_hours
        self.stride = stride
        self.name = name or self.inner.name
        self.horizon = self.inner.horizon
        self.target = getattr(self.inner, "target", "load_kw")
        self.lower_level = alpha / 2
        self.upper_level = 1 - alpha / 2
        self.split: SplitConformal | None = None
        self.cqr: ConformalizedQuantile | None = None

    def calibration_positions(self, n_rows: int, fit_end: int) -> np.ndarray:
        return np.arange(fit_end - 1, n_rows - self.horizon, self.stride)

    def fit(self, frame: pd.DataFrame) -> "ConformalForecaster":
        n_rows = len(frame)
        fit_end = n_rows - self.calibration_hours
        if fit_end <= self.horizon:
            raise ValueError("calibration window leaves no training data")
        self.inner.fit(frame.iloc[:fit_end])
        positions = self.calibration_positions(n_rows, fit_end)
        forecast = self.inner.predict(frame, frame.index[positions])
        actual = realised(frame, self.target, positions, self.horizon)
        if self.lower_level in forecast.quantiles and self.upper_level in forecast.quantiles:
            self.cqr = ConformalizedQuantile(self.alpha).fit(
                actual, forecast.quantiles[self.lower_level], forecast.quantiles[self.upper_level]
            )
        else:
            self.split = SplitConformal(self.alpha).fit(actual, forecast.point)
        return self

    def predict(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> Forecast:
        forecast = self.inner.predict(frame, origins)
        extras: dict[str, np.ndarray] = {}
        if self.cqr is not None:
            raw_low = forecast.quantiles[self.lower_level]
            raw_high = forecast.quantiles[self.upper_level]
            lower, upper = self.cqr.interval(raw_low, raw_high)
            extras["raw_q10"] = raw_low
            extras["raw_q90"] = raw_high
        elif self.split is not None:
            lower, upper = self.split.interval(forecast.point)
        else:
            raise RuntimeError("model has not been fitted")
        quantiles = {self.lower_level: lower, self.upper_level: upper}
        return Forecast(point=forecast.point, quantiles=quantiles, extras=extras)
