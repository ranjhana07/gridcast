from __future__ import annotations

import numpy as np
import pandas as pd

STANDARD_IRRADIANCE = 1000.0
REFERENCE_TEMPERATURE = 25.0


def simulate_pv(
    weather: pd.DataFrame,
    capacity_kw: float = 500.0,
    performance_ratio: float = 0.82,
    temperature_coefficient: float = -0.004,
) -> pd.Series:
    irradiance = weather["shortwave_radiation"].clip(lower=0)
    cell_temperature = weather["temperature_2m"] + 0.03 * irradiance
    derate = 1 + temperature_coefficient * (cell_temperature - REFERENCE_TEMPERATURE)
    power = capacity_kw * performance_ratio * irradiance / STANDARD_IRRADIANCE * derate
    power = power.clip(lower=0, upper=capacity_kw)
    return power.rename("solar_kw")


def clear_sky_hours(index: pd.DatetimeIndex, latitude: float) -> np.ndarray:
    day = index.dayofyear.to_numpy()
    declination = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + day) / 365)
    hour_angle = np.deg2rad(15 * (index.hour.to_numpy() - 12))
    lat = np.deg2rad(latitude)
    elevation = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    return np.clip(elevation, 0, None)
