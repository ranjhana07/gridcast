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
