from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float
    power_kw: float
    round_trip_efficiency: float = 0.90
    min_soc_share: float = 0.10
    initial_soc_share: float = 0.50
    wear_cost_per_kwh: float = 0.004

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0 or self.power_kw <= 0:
            raise ValueError("capacity and power must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("efficiency must lie in (0, 1]")

    @property
    def one_way_efficiency(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))

    @property
    def min_soc_kwh(self) -> float:
        return self.capacity_kwh * self.min_soc_share

    @property
    def initial_soc_kwh(self) -> float:
        return self.capacity_kwh * self.initial_soc_share

    @classmethod
    def sized_for(cls, load: np.ndarray, power_share: float = 0.08, hours: float = 4.0) -> "Battery":
        power = float(np.max(load)) * power_share
        return cls(capacity_kwh=power * hours, power_kw=power)
