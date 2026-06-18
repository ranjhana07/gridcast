from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Tariff:
    off_peak: float = 0.10
    shoulder: float = 0.18
    peak: float = 0.32
    demand_rate: float = 14.0
    peak_hours: tuple[int, int] = (17, 21)
    shoulder_hours: tuple[int, int] = (8, 23)

    def energy_prices(self, index: pd.DatetimeIndex) -> np.ndarray:
        hour = index.hour.to_numpy()
        weekday = index.day_of_week.to_numpy() < 5
        prices = np.full(len(index), self.off_peak, dtype=float)
        shoulder = weekday & (hour >= self.shoulder_hours[0]) & (hour < self.shoulder_hours[1])
        peak = weekday & (hour >= self.peak_hours[0]) & (hour < self.peak_hours[1])
        prices[shoulder] = self.shoulder
        prices[peak] = self.peak
        return prices
