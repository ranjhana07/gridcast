from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from gridcast.models.base import Forecast, origin_positions


class ETSForecaster:
    def __init__(
        self,
        target: str = "load_kw",
        horizon: int = 24,
        period: int = 24,
        window_days: int = 21,
        damped: bool = True,
    ) -> None:
        self.target = target
        self.horizon = horizon
        self.period = period
        self.window = window_days * 24
        self.damped = damped
        self.name = "ets"

    def fit(self, frame: pd.DataFrame) -> "ETSForecaster":
        return self

    def forecast_from(self, history: np.ndarray) -> np.ndarray:
        model = ExponentialSmoothing(
            history,
            trend="add",
            damped_trend=self.damped,
            seasonal="add",
            seasonal_periods=self.period,
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            fitted = model.fit(optimized=True)
        return np.asarray(fitted.forecast(self.horizon))

    def predict(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> Forecast:
        values = frame[self.target].to_numpy()
        positions = origin_positions(frame, origins)
        rows = []
        for position in positions:
            history = values[max(0, position + 1 - self.window) : position + 1]
            rows.append(self.forecast_from(history))
        return Forecast(point=np.vstack(rows))
