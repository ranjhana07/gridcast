from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from gridcast.tracking import load_model_artifact

PRODUCTION_ALIAS = "production"
CANDIDATE_METRIC = "holdout_mae"


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    version: str
    reason: str
    candidate_metric: float
    production_metric: float | None


def registered_name(target: str, model_name: str) -> str:
    return f"gridcast-{target}-{model_name}".replace("_", "-")


def ensure_registered(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)


def register_run(client: MlflowClient, name: str, run_id: str) -> str:
    ensure_registered(client, name)
    run = client.get_run(run_id)
    version = client.create_model_version(
        name, source=f"{run.info.artifact_uri}/model", run_id=run_id
    )
    return str(version.version)


def production_version(client: MlflowClient, name: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, PRODUCTION_ALIAS).version)
    except MlflowException:
        return None


def metric_of(client: MlflowClient, name: str, version: str, metric: str = CANDIDATE_METRIC) -> float:
    run_id = client.get_model_version(name, version).run_id
    return float(client.get_run(run_id).data.metrics[metric])


def promote_if_better(
    client: MlflowClient,
    name: str,
    version: str,
    tolerance: float = 0.0,
    metric: str = CANDIDATE_METRIC,
) -> PromotionDecision:
    candidate = metric_of(client, name, version, metric)
    current = production_version(client, name)
    if current is None:
        client.set_registered_model_alias(name, PRODUCTION_ALIAS, version)
        return PromotionDecision(True, version, "no production model yet", candidate, None)
    incumbent = metric_of(client, name, current, metric)
    if candidate <= incumbent * (1 + tolerance):
        client.set_registered_model_alias(name, PRODUCTION_ALIAS, version)
        return PromotionDecision(True, version, "candidate matches or beats production", candidate, incumbent)
    return PromotionDecision(False, version, "candidate is worse than production", candidate, incumbent)


def load_production_model(client: MlflowClient, name: str) -> object:
    version = client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    local = mlflow.artifacts.download_artifacts(artifact_uri=version.source)
    return load_model_artifact(local)
