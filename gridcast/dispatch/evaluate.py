from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gridcast.dispatch.battery import Battery
from gridcast.dispatch.simulate import compare_strategies, forecasts_from_predictions
from gridcast.dispatch.tariff import Tariff


def evaluate_dispatch(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    models: list[str],
    reference: str,
    tariff: Tariff,
    battery: Battery,
    target: str = "load_kw",
) -> pd.DataFrame:
    forecasts = {
        model: forecasts_from_predictions(predictions, model)
        for model in models
        if model in set(predictions["model"])
    }
    forecasts = {name: value for name, value in forecasts.items() if value}
    if reference not in forecasts:
        raise KeyError(f"reference model {reference!r} has no usable forecasts")
    return compare_strategies(frame[target], forecasts, tariff, battery, reference)


def write_dispatch_report(table: pd.DataFrame, battery: Battery, tariff: Tariff, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.2f")
    setup = {
        "battery": {
            "capacity_kwh": battery.capacity_kwh,
            "power_kw": battery.power_kw,
            "round_trip_efficiency": battery.round_trip_efficiency,
        },
        "tariff": {
            "off_peak": tariff.off_peak,
            "shoulder": tariff.shoulder,
            "peak": tariff.peak,
            "demand_rate_per_kw_month": tariff.demand_rate,
        },
    }
    path.with_suffix(".json").write_text(json.dumps(setup, indent=2))
