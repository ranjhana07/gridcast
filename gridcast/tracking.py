from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient


def configure(tracking_uri: str) -> MlflowClient:
    if tracking_uri.startswith("file:") or "://" not in tracking_uri:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri)


@contextmanager
def tracked_run(experiment: str, run_name: str, tags: dict[str, str] | None = None) -> Iterator[mlflow.ActiveRun]:
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_model_artifact(model: object, file_name: str = "model.joblib") -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / file_name
        joblib.dump(model, path)
        mlflow.log_artifact(str(path), "model")


def load_model_artifact(local_dir: str, file_name: str = "model.joblib") -> object:
    return joblib.load(Path(local_dir) / file_name)


def log_benchmark(tables: dict, meta: dict, experiment: str = "gridcast-benchmark", run_name: str = "backtest") -> str:
    with tracked_run(experiment, run_name) as run:
        config = meta["config"]
        mlflow.log_params(
            {
                "target": config["target"],
                "folds": config["n_folds"],
                "fold_hours": config["fold_hours"],
                "stride": config["stride"],
                "alpha": meta["alpha"],
                "rows": meta["rows"],
            }
        )
        for row in tables["metrics_overall"].to_dict(orient="records"):
            model = row.pop("model")
            mlflow.log_metrics({f"{model}.{name}": value for name, value in row.items()})
        for row in tables["coverage_overall"].to_dict(orient="records"):
            model = row.pop("model")
            mlflow.log_metrics({f"{model}.{name}": value for name, value in row.items()})
        for row in tables["timing"].to_dict(orient="records"):
            model = row.pop("model")
            mlflow.log_metrics({f"{model}.{name}": value for name, value in row.items()})
        return run.info.run_id
