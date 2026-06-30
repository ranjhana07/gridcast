from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.config import Settings, get_settings
from gridcast.data import dataset
from gridcast.dispatch.battery import Battery
from gridcast.dispatch.optimizer import plan_dispatch
from gridcast.dispatch.tariff import Tariff
from gridcast.registry import (
    load_production_model,
    promote_if_better,
    registered_name,
)
from gridcast.scheduler import Job
from gridcast.store import ArtifactStore
from gridcast.tracking import configure
from gridcast.training import register_result, train_candidate

HISTORY_HOURS = 24 * 7


def frame_path(settings: Settings) -> Path:
    return settings.processed_dir / "microgrid_hourly.csv"


def retrain_flow(
    settings: Settings | None = None,
    targets: tuple[str, ...] = ("load_kw",),
    model_name: str = "lightgbm",
    tolerance: float = 0.02,
) -> list[dict]:
    settings = settings or get_settings()
    frame = dataset.load_frame(frame_path(settings))
    store = ArtifactStore(settings.artifact_dir)
    reports = []
    for target in targets:
        result = train_candidate(frame, model_name, target, settings.horizon, settings.mlflow_uri)
        version = register_result(settings.mlflow_uri, result)
        client = configure(settings.mlflow_uri)
        decision = promote_if_better(client, result.registered_name, version, tolerance=tolerance)
        reports.append({"target": target, "model": model_name, "run_id": result.run_id, **asdict(decision)})
    store.write_json("retrain_latest.json", {"runs": reports, "data_end": str(frame.index[-1])})
    return reports


def forecast_payload(
    frame: pd.DataFrame,
    model,
    target: str,
    origin: pd.Timestamp,
    settings: Settings,
) -> dict:
    horizon = settings.horizon
    history = frame.loc[:origin]
    extended = dataset.extend_frame(history, horizon, settings.country, settings.latitude)
    forecast = model.predict(extended, pd.DatetimeIndex([origin]))
    lower, upper = forecast.interval(model.lower_level, model.upper_level)
    times = pd.date_range(origin + timedelta(hours=1), periods=horizon, freq="1h")
    recent = history[target].iloc[-HISTORY_HOURS:]
    return {
        "target": target,
        "model": model.name,
        "origin": origin.isoformat(),
        "interval": {"lower_level": model.lower_level, "upper_level": model.upper_level},
        "timestamps": [t.isoformat() for t in times],
        "point": forecast.point[0].round(2).tolist(),
        "lower": lower[0].round(2).tolist(),
        "upper": upper[0].round(2).tolist(),
        "history": {
            "timestamps": [t.isoformat() for t in recent.index],
            "values": recent.round(2).tolist(),
        },
    }


def dispatch_payload(
    forecast: dict,
    battery: Battery,
    tariff: Tariff,
    soc: float | None = None,
) -> dict:
    index = pd.DatetimeIndex(pd.to_datetime(forecast["timestamps"]))
    load = np.array(forecast["point"])
    prices = tariff.energy_prices(index)
    plan = plan_dispatch(load, prices, battery, tariff.demand_rate, initial_soc=soc)
    grid_with = load + plan.net
    return {
        "origin": forecast["origin"],
        "timestamps": forecast["timestamps"],
        "load": load.round(2).tolist(),
        "price": prices.round(4).tolist(),
        "charge": plan.charge.round(2).tolist(),
        "discharge": plan.discharge.round(2).tolist(),
        "soc": plan.soc.round(2).tolist(),
        "grid": grid_with.round(2).tolist(),
        "peak_without_kw": float(load.max()),
        "peak_with_kw": float(grid_with.max()),
        "cost_without": float((load * prices).sum()),
        "cost_with": float((grid_with * prices).sum()),
        "battery": {"capacity_kwh": battery.capacity_kwh, "power_kw": battery.power_kw},
    }


def forecast_flow(
    settings: Settings | None = None,
    targets: tuple[str, ...] = ("load_kw",),
    model_name: str = "lightgbm",
    as_of: pd.Timestamp | None = None,
) -> dict:
    settings = settings or get_settings()
    frame = dataset.load_frame(frame_path(settings))
    origin = as_of or frame.index[-1]
    client = configure(settings.mlflow_uri)
    store = ArtifactStore(settings.artifact_dir)
    payloads = {}
    for target in targets:
        model = load_production_model(client, registered_name(target, model_name))
        payload = forecast_payload(frame, model, target, origin, settings)
        store.write_json(store.forecast_name(target), payload)
        payloads[target] = payload
        if target == "load_kw":
            battery = Battery.sized_for(frame["load_kw"].to_numpy())
            store.write_json(store.dispatch_name(), dispatch_payload(payload, battery, Tariff()))
    return payloads


def default_jobs(settings: Settings | None = None) -> list[Job]:
    settings = settings or get_settings()
    return [
        Job("retrain", timedelta(days=1), lambda: retrain_flow(settings)),
        Job("forecast", timedelta(hours=1), lambda: forecast_flow(settings)),
    ]
