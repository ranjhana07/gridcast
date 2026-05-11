from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import requests

WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
)

Fetcher = Callable[[str, dict[str, str]], dict]


def http_fetch(url: str, params: dict[str, str]) -> dict:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"open-meteo request failed: {last_error}")


def cache_path(cache_dir: Path, latitude: float, longitude: float, start: str, end: str) -> Path:
    key = f"{latitude:.3f}|{longitude:.3f}|{start}|{end}|{','.join(WEATHER_VARIABLES)}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return cache_dir / f"weather_{start}_{end}_{digest}.csv"


def parse_response(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    frame = pd.DataFrame({name: hourly[name] for name in WEATHER_VARIABLES})
    frame.index = pd.DatetimeIndex(pd.to_datetime(hourly["time"]), name="timestamp")
    return frame


def fetch_window(
    url: str,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    cache_dir: Path,
    fetcher: Fetcher = http_fetch,
) -> pd.DataFrame:
    path = cache_path(cache_dir, latitude, longitude, start, end)
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    params = {
        "latitude": f"{latitude}",
        "longitude": f"{longitude}",
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "Europe/Lisbon",
    }
    frame = parse_response(fetcher(url, params))
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


def load_weather(
    url: str,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    cache_dir: Path,
    fetcher: Fetcher = http_fetch,
) -> pd.DataFrame:
    windows = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        window_start = max(pd.Timestamp(start), pd.Timestamp(year=year, month=1, day=1))
        window_end = min(pd.Timestamp(end), pd.Timestamp(year=year, month=12, day=31))
        windows.append(
            fetch_window(
                url,
                latitude,
                longitude,
                window_start.strftime("%Y-%m-%d"),
                window_end.strftime("%Y-%m-%d"),
                cache_dir,
                fetcher,
            )
        )
    frame = pd.concat(windows)
    return frame[~frame.index.duplicated(keep="first")].sort_index()
