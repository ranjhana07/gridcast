from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.models.base import Forecast, origin_positions


class SeasonalNaive:
    def __init__(self, target: str = "load_kw", horizon: int = 24, period: int = 168) -> None:
        self.target = target
        self.horizon = horizon
        self.period = period
        self.name = f"seasonal_naive_{period}"

    def fit(self, frame: pd.DataFrame) -> "SeasonalNaive":
        return self

    def predict(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> Forecast:
        values = frame[self.target].to_numpy()
        positions = origin_positions(frame, origins)
        steps = np.arange(1, self.horizon + 1)
        source = positions[:, None] + steps[None, :] - self.period
        if (source > positions[:, None]).any():
            raise ValueError("seasonal period is shorter than the horizon")
        return Forecast(point=values[np.clip(source, 0, None)])
