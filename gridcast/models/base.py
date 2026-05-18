from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class Forecast:
    point: np.ndarray
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return self.point.shape[1]

    def interval(self, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
        return self.quantiles[lower], self.quantiles[upper]


class Forecaster(Protocol):
    name: str
    horizon: int

    def fit(self, frame: pd.DataFrame) -> "Forecaster": ...

    def predict(self, frame: pd.DataFrame, origins: pd.DatetimeIndex) -> Forecast: ...


def origin_positions(frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
    positions = frame.index.get_indexer(origins)
    if (positions < 0).any():
        raise KeyError("some origins are missing from the frame")
    return positions
