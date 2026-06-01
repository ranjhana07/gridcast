from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CALENDAR_COLUMNS = (
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "is_bridge",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
)
WEATHER_COLUMNS = (
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
)


@dataclass(frozen=True)
class FeatureSpec:
    target: str = "load_kw"
    horizon: int = 24
    lags: tuple[int, ...] = (0, 1, 2, 3, 6, 12, 23, 24, 47, 48, 72, 167, 168, 169)
    windows: tuple[int, ...] = (6, 24, 168)
    seasonal_lags: tuple[int, ...] = (24, 48, 168)
    future_columns: tuple[str, ...] = CALENDAR_COLUMNS + WEATHER_COLUMNS
    origin_columns: tuple[str, ...] = ("temperature_2m", "shortwave_radiation")
    extras: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if min(self.seasonal_lags) < self.horizon:
            raise ValueError("seasonal lags must be at least as long as the horizon")


def solar_spec(horizon: int = 24) -> FeatureSpec:
    return FeatureSpec(
        target="solar_kw",
        horizon=horizon,
        lags=(0, 1, 2, 3, 6, 12, 23, 24, 48, 168),
        windows=(6, 24),
        future_columns=CALENDAR_COLUMNS + ("sun_elevation",),
        origin_columns=("cloud_cover", "shortwave_radiation", "relative_humidity_2m"),
    )


def origin_features(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    series = frame[spec.target]
    columns: dict[str, pd.Series] = {}
    for lag in spec.lags:
        columns[f"lag_{lag}"] = series.shift(lag)
    for window in spec.windows:
        rolled = series.rolling(window, min_periods=window)
        columns[f"roll_mean_{window}"] = rolled.mean()
        columns[f"roll_std_{window}"] = rolled.std()
    columns["trend_24"] = series - series.shift(24)
    for name in spec.origin_columns:
        columns[f"origin_{name}"] = frame[name]
    return pd.DataFrame(columns, index=frame.index)


def horizon_features(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    step: int,
    origin: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not 1 <= step <= spec.horizon:
        raise ValueError("step outside the forecast horizon")
    base = origin if origin is not None else origin_features(frame, spec)
    series = frame[spec.target]
    parts = {f"seasonal_{lag}": series.shift(lag - step) for lag in spec.seasonal_lags}
    for name in spec.future_columns:
        parts[f"future_{name}"] = frame[name].shift(-step)
    return pd.concat([base, pd.DataFrame(parts, index=frame.index)], axis=1)


def horizon_target(frame: pd.DataFrame, spec: FeatureSpec, step: int) -> pd.Series:
    return frame[spec.target].shift(-step)


def training_matrix(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    step: int,
    origin: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    features = horizon_features(frame, spec, step, origin)
    target = horizon_target(frame, spec, step)
    valid = features.notna().all(axis=1) & target.notna()
    return features[valid], target[valid]


def feature_names(spec: FeatureSpec, step: int = 1) -> list[str]:
    probe = pd.DataFrame(
        0.0,
        index=pd.date_range("2024-01-01", periods=400, freq="1h"),
        columns=sorted({spec.target, *spec.future_columns, *spec.origin_columns}),
    )
    return list(horizon_features(probe, spec, step).columns)
