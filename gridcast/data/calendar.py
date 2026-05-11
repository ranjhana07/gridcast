from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def holiday_calendar(country: str, years: range) -> holidays.HolidayBase:
    return holidays.country_holidays(country, years=list(years))


def calendar_features(index: pd.DatetimeIndex, country: str = "PT") -> pd.DataFrame:
    years = range(index.min().year - 1, index.max().year + 2)
    calendar = holiday_calendar(country, years)
    dates = index.normalize()
    holiday_dates = pd.DatetimeIndex(sorted(pd.Timestamp(day) for day in calendar.keys()))
    is_holiday = dates.isin(holiday_dates)
    day_before = (dates + pd.Timedelta(days=1)).isin(holiday_dates)
    day_after = (dates - pd.Timedelta(days=1)).isin(holiday_dates)

    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()
    frame = pd.DataFrame(
        {
            "hour": hour,
            "dayofweek": dow,
            "month": index.month.to_numpy(),
            "is_weekend": (dow >= 5).astype(np.int8),
            "is_holiday": is_holiday.astype(np.int8),
            "is_bridge": ((day_before | day_after) & (dow < 5) & ~is_holiday).astype(np.int8),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )
    return frame
