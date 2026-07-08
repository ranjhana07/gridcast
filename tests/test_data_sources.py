import numpy as np
import pandas as pd

from gridcast.config import Settings
from gridcast.data import dataset, uci
from gridcast.data.pjm import load_pjm
from gridcast.data.solar import simulate_pv
from gridcast.data.weather import WEATHER_VARIABLES, fetch_window, load_weather


class CountingFetcher:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, url: str, params: dict) -> dict:
        self.calls += 1
        index = pd.date_range(params["start_date"], f"{params['end_date']} 23:00", freq="1h")
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in index]}
        for name in WEATHER_VARIABLES:
            hourly[name] = [10.0] * len(index)
        hourly["shortwave_radiation"] = [max(0.0, 800 * np.sin(np.pi * (t.hour - 6) / 12)) for t in index]
        return {"hourly": hourly}


def test_weather_is_cached_on_disk(tmp_path):
    fetcher = CountingFetcher()
    first = fetch_window("http://example", 38.7, -9.1, "2024-01-01", "2024-01-10", tmp_path, fetcher)
    second = fetch_window("http://example", 38.7, -9.1, "2024-01-01", "2024-01-10", tmp_path, fetcher)
    assert fetcher.calls == 1
    assert len(first) == len(second) == 240
    assert list(second.columns) == list(WEATHER_VARIABLES)


def test_weather_is_requested_per_year(tmp_path):
    fetcher = CountingFetcher()
    frame = load_weather("http://example", 38.7, -9.1, "2023-12-30", "2024-01-02", tmp_path, fetcher)
    assert fetcher.calls == 2
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 4 * 24


def test_pv_is_zero_at_night_and_capped():
    index = pd.date_range("2024-06-21", periods=24, freq="1h")
    weather = pd.DataFrame({"shortwave_radiation": [0.0] * 12 + [1200.0] * 12, "temperature_2m": 25.0}, index=index)
    power = simulate_pv(weather, capacity_kw=100.0)
    assert (power.iloc[:12] == 0).all()
    assert power.max() <= 100.0


def test_uci_loader_aggregates_selected_clients(tmp_path):
    index = pd.date_range("2012-01-01 00:15", periods=4 * 48, freq="15min")
    rows = ["\"\";\"MT_001\";\"MT_002\";\"MT_003\""]
    for stamp in index:
        rows.append(f"\"{stamp}\";1,5;2,5;0,0")
    path = tmp_path / "ld.txt"
    path.write_text("\n".join(rows))
    hourly = uci.to_hourly(uci.read_raw(path))
    assert uci.active_clients(hourly, "2012-01-01") == ["MT_001", "MT_002"]
    total = uci.load_microgrid(path, n_clients=2, start="2012-01-01")
    assert np.isclose(total.iloc[3], 4.0)


def test_pjm_loader_sorts_and_dedupes(tmp_path):
    path = tmp_path / "pjm.csv"
    path.write_text("Datetime,PJME_MW\n2020-01-01 02:00:00,3\n2020-01-01 01:00:00,2\n2020-01-01 01:00:00,9\n")
    series = load_pjm(path)
    assert list(series.values) == [2.0, 3.0]


def test_build_frame_has_all_modelling_columns(tmp_path):
    fetcher = CountingFetcher()
    index = pd.date_range("2024-01-01", periods=24 * 20, freq="1h", name="timestamp")
    load = pd.Series(100 + 20 * np.sin(2 * np.pi * index.hour.to_numpy() / 24), index=index, name="load_kw")
    weather = load_weather("http://example", 38.7, -9.1, "2024-01-01", "2024-01-20", tmp_path, fetcher)
    frame = dataset.build_frame(load, weather, "PT", 38.7)
    expected = {"load_kw", "solar_kw", "sun_elevation", "temperature_2m", "hour_sin", "is_holiday"}
    assert expected <= set(frame.columns)
    assert not frame.isna().any().any()


def test_extend_frame_adds_future_rows(tmp_path):
    fetcher = CountingFetcher()
    index = pd.date_range("2024-01-01", periods=24 * 20, freq="1h", name="timestamp")
    load = pd.Series(100.0, index=index, name="load_kw")
    weather = load_weather("http://example", 38.7, -9.1, "2024-01-01", "2024-01-20", tmp_path, fetcher)
    frame = dataset.build_frame(load, weather, "PT", 38.7)
    extended = dataset.extend_frame(frame, 24, "PT", 38.7)
    assert len(extended) == len(frame) + 24
    assert extended["load_kw"].iloc[-24:].isna().all()
    assert extended["temperature_2m"].iloc[-24:].notna().all()
    assert extended.index[-1] == frame.index[-1] + pd.Timedelta(hours=24)


def test_settings_paths_follow_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("GRIDCAST_DATA_DIR", str(tmp_path))
    settings = Settings()
    assert settings.cache_dir == tmp_path / "cache"
    assert settings.processed_dir == tmp_path / "processed"
