from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from gridcast.conformal import ConformalizedQuantile, SplitConformal, online_corrections
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
        online: bool = False,
        online_window: int = 400,
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
        self.online = online
        self.online_window = online_window
        self.calibration_scores: np.ndarray | None = None

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
            lower = forecast.quantiles[self.lower_level]
            upper = forecast.quantiles[self.upper_level]
            self.cqr = ConformalizedQuantile(self.alpha).fit(actual, lower, upper)
            self.calibration_scores = np.maximum(lower - actual, actual - upper)
        else:
            self.split = SplitConformal(self.alpha).fit(actual, forecast.point)
            self.calibration_scores = np.abs(actual - forecast.point)
        return self

    def online_interval(
        self,
        frame: pd.DataFrame,
        positions: np.ndarray,
        forecast: Forecast,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = frame[self.target].to_numpy()
        steps = np.arange(1, self.horizon + 1)
        reach = np.clip(positions[:, None] + steps[None, :], 0, len(values) - 1)
        actual = values[reach]
        if self.cqr is not None:
            raw_low = forecast.quantiles[self.lower_level]
            raw_high = forecast.quantiles[self.upper_level]
            scores = np.maximum(raw_low - actual, actual - raw_high)
            shift = online_corrections(self.calibration_scores, scores, positions, self.alpha, self.online_window)
            return raw_low - shift, raw_high + shift
        scores = np.abs(actual - forecast.point)
        radius = online_corrections(self.calibration_scores, scores, positions, self.alpha, self.online_window)
        return forecast.point - radius, forecast.point + radius

    def predict(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> Forecast:
        forecast = self.inner.predict(frame, origins)
        extras: dict[str, np.ndarray] = {}
        if self.online and self.calibration_scores is not None:
            lower, upper = self.online_interval(frame, origin_positions(frame, origins), forecast)
            if self.cqr is not None:
                extras["raw_q10"] = forecast.quantiles[self.lower_level]
                extras["raw_q90"] = forecast.quantiles[self.upper_level]
        elif self.cqr is not None:
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
