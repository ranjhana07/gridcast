import numpy as np
import pandas as pd

from gridcast.config import Settings
from gridcast.data.calendar import calendar_features
from gridcast.dispatch.battery import Battery
from gridcast.dispatch.tariff import Tariff
from gridcast.flows import dispatch_payload, forecast_payload
from gridcast.zoo import build_factories


def full_frame(days: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=24 * days, freq="1h", name="timestamp")
    rng = np.random.default_rng(4)
    hour = index.hour.to_numpy()
    frame = pd.DataFrame(
        {
            "load_kw": 500 + 100 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 5, len(index)),
            "solar_kw": 0.0,
            "temperature_2m": 15.0,
            "relative_humidity_2m": 60.0,
            "cloud_cover": 40.0,
            "wind_speed_10m": 3.0,
            "shortwave_radiation": 100.0,
            "sun_elevation": 0.3,
        },
        index=index,
    )
    return frame.join(calendar_features(index, "PT"))


def test_forecast_and_dispatch_payloads_at_the_data_end():
    frame = full_frame()
    model = build_factories(["naive_168"])["naive_168"]().fit(frame)
    settings = Settings()
    payload = forecast_payload(frame, model, "load_kw", frame.index[-1], settings)
    assert len(payload["point"]) == 24
    assert all(lo <= hi for lo, hi in zip(payload["lower"], payload["upper"]))
    assert payload["timestamps"][0] == (frame.index[-1] + pd.Timedelta(hours=1)).isoformat()
    battery = Battery.sized_for(frame["load_kw"].to_numpy())
    plan = dispatch_payload(payload, battery, Tariff())
    assert len(plan["charge"]) == 24
    assert plan["peak_with_kw"] <= plan["peak_without_kw"] + 1e-6
    assert plan["cost_with"] <= plan["cost_without"] + 1e-6
