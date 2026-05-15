from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.data.calendar import calendar_features


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=24 * 120, freq="1h", name="timestamp")
    rng = np.random.default_rng(3)
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    temperature = 14 + 8 * np.sin(2 * np.pi * (index.dayofyear.to_numpy() - 100) / 365) + rng.normal(0, 1, len(index))
    load = 900 + 250 * np.sin(2 * np.pi * (hour - 8) / 24) - 120 * (dow >= 5) + 6 * temperature
    load = load + rng.normal(0, 18, len(index))
    radiation = np.clip(700 * np.sin(np.pi * (hour - 6) / 12), 0, None) * rng.uniform(0.6, 1.0, len(index))
    frame = pd.DataFrame(
        {
            "load_kw": load,
            "solar_kw": radiation * 0.4,
            "temperature_2m": temperature,
            "relative_humidity_2m": 60 + rng.normal(0, 5, len(index)),
            "cloud_cover": rng.uniform(0, 100, len(index)),
            "wind_speed_10m": rng.uniform(0, 12, len(index)),
            "shortwave_radiation": radiation,
        },
        index=index,
    )
    return frame.join(calendar_features(index, "PT"))
