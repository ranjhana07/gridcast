from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlflow
import numpy as np
import pandas as pd

from gridcast import metrics
from gridcast.models.calibrated import ConformalForecaster, realised
from gridcast.registry import register_run, registered_name
from gridcast.tracking import configure, log_model_artifact, tracked_run
from gridcast.zoo import build_factories


@dataclass
class TrainResult:
    run_id: str
    model_name: str
    registered_name: str
    metrics: dict[str, float]
    model: ConformalForecaster = field(repr=False)


def holdout_metrics(
    model: ConformalForecaster,
    frame: pd.DataFrame,
    train_end: int,
    stride: int = 6,
) -> dict[str, float]:
    horizon = model.horizon
    positions = np.arange(train_end - 1, len(frame) - horizon, stride)
    forecast = model.predict(frame, frame.index[positions])
    actual = realised(frame, model.target, positions, horizon)
    lower, upper = forecast.interval(model.lower_level, model.upper_level)
    return {
        "holdout_mae": float(metrics.mae(actual, forecast.point)),
        "holdout_rmse": float(metrics.rmse(actual, forecast.point)),
        "holdout_smape": float(metrics.smape(actual, forecast.point)),
        "holdout_coverage": metrics.interval_coverage(actual, lower, upper),
        "holdout_width": metrics.interval_width(lower, upper),
    }


def train_candidate(
    frame: pd.DataFrame,
    model_name: str,
    target: str,
    horizon: int,
    tracking_uri: str,
    holdout_hours: int = 24 * 14,
    experiment: str = "gridcast",
) -> TrainResult:
    configure(tracking_uri)
    factory = build_factories([model_name], target, horizon)[model_name]
    model = factory()
    train_end = len(frame) - holdout_hours
    started = time.perf_counter()
    model.fit(frame.iloc[:train_end])
    fit_seconds = time.perf_counter() - started
    scores = holdout_metrics(model, frame, train_end)
    with tracked_run(experiment, f"{target}-{model_name}", {"target": target, "model": model_name}) as run:
        mlflow.log_params(
            {
                "model": model_name,
                "target": target,
                "horizon": horizon,
                "train_rows": train_end,
                "holdout_hours": holdout_hours,
                "data_end": str(frame.index[-1]),
            }
        )
        mlflow.log_metrics({**scores, "fit_seconds": fit_seconds})
        log_model_artifact(model)
        run_id = run.info.run_id
    return TrainResult(run_id, model_name, registered_name(target, model_name), scores, model)


def register_result(tracking_uri: str, result: TrainResult) -> str:
    client = configure(tracking_uri)
    return register_run(client, result.registered_name, result.run_id)
