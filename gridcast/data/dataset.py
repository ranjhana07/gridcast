from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.config import Settings, get_settings
from gridcast.data import uci
from gridcast.data.calendar import calendar_features
from gridcast.data.solar import clear_sky_hours, simulate_pv
from gridcast.data.validation import prepare_series, require_columns
from gridcast.data.weather import WEATHER_VARIABLES, Fetcher, http_fetch, load_weather

TARGETS = ("load_kw", "solar_kw")


def build_frame(
    load: pd.Series,
    weather: pd.DataFrame,
    country: str,
    latitude: float,
    pv_capacity_kw: float = 500.0,
) -> pd.DataFrame:
    load = prepare_series(load)
    weather = weather.reindex(load.index).interpolate(limit_direction="both")
    frame = pd.DataFrame({"load_kw": load})
    frame["solar_kw"] = simulate_pv(weather, capacity_kw=pv_capacity_kw)
    frame = frame.join(weather)
    frame["sun_elevation"] = clear_sky_hours(frame.index, latitude)
    frame = frame.join(calendar_features(frame.index, country))
    return frame


def build_from_uci(
    raw_path: Path,
    settings: Settings | None = None,
    n_clients: int = 30,
    start: str = "2012-01-01",
    fetcher: Fetcher = http_fetch,
) -> pd.DataFrame:
    settings = settings or get_settings()
    load = uci.load_microgrid(raw_path, n_clients=n_clients, start=start)
    weather = load_weather(
        settings.open_meteo_url,
        settings.latitude,
        settings.longitude,
        load.index.min().strftime("%Y-%m-%d"),
        load.index.max().strftime("%Y-%m-%d"),
        settings.cache_dir,
        fetcher,
    )
    return build_frame(load, weather, settings.country, settings.latitude)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "timestamp"
    require_columns(frame, ("load_kw", "solar_kw", "temperature_2m", "hour"))
    return frame


def extend_frame(frame: pd.DataFrame, steps: int, country: str, latitude: float) -> pd.DataFrame:
    future_index = pd.date_range(frame.index[-1] + pd.Timedelta(hours=1), periods=steps, freq="1h", name="timestamp")
    future = pd.DataFrame(index=future_index, columns=frame.columns, dtype=float)
    for column in WEATHER_VARIABLES:
        source = frame[column].iloc[-24:].to_numpy()
        future[column] = np.resize(source, steps)
    future["sun_elevation"] = clear_sky_hours(future_index, latitude)
    calendar = calendar_features(future_index, country)
    for column in calendar.columns:
        future[column] = calendar[column]
    return pd.concat([frame, future])
