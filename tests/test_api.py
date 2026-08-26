import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridcast.api.app import create_app
from gridcast.config import Settings


def forecast_document(target: str = "load_kw") -> dict:
    times = pd.date_range("2024-06-02 01:00", periods=24, freq="1h")
    past = pd.date_range("2024-05-31 01:00", periods=48, freq="1h")
    return {
        "target": target,
        "model": "lightgbm",
        "origin": "2024-06-02T00:00:00",
        "interval": {"lower_level": 0.1, "upper_level": 0.9},
        "timestamps": [t.isoformat() for t in times],
        "point": [100.0 + i for i in range(24)],
        "lower": [90.0 + i for i in range(24)],
        "upper": [112.0 + i for i in range(24)],
        "history": {"timestamps": [t.isoformat() for t in past], "values": [95.0] * 48},
    }


def dispatch_document() -> dict:
    times = pd.date_range("2024-06-02 01:00", periods=24, freq="1h")
    return {
        "origin": "2024-06-02T00:00:00",
        "timestamps": [t.isoformat() for t in times],
        "load": [100.0] * 24,
        "price": [0.1] * 24,
        "charge": [0.0] * 24,
        "discharge": [0.0] * 24,
        "soc": [50.0] * 24,
        "grid": [100.0] * 24,
        "peak_without_kw": 100.0,
        "peak_with_kw": 100.0,
        "cost_without": 240.0,
        "cost_with": 240.0,
        "battery": {"capacity_kwh": 200.0, "power_kw": 50.0},
    }


def prediction_rows() -> pd.DataFrame:
    rows = []
    for model in ("lightgbm", "naive_168"):
        for origin in pd.date_range("2024-05-01", periods=3, freq="6h"):
            for horizon in range(1, 25):
                rows.append(
                    {
                        "model": model,
                        "fold": 0,
                        "origin": origin,
                        "horizon": horizon,
                        "actual": 100.0,
                        "point": 99.0,
                        "q10": 90.0,
                        "q90": 110.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def populated(tmp_path: Path) -> Settings:
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    artifacts.mkdir()
    results.mkdir()
    (artifacts / "forecast_load_kw.json").write_text(json.dumps(forecast_document()))
    (artifacts / "dispatch_latest.json").write_text(json.dumps(dispatch_document()))
    pd.DataFrame({"model": ["lightgbm"], "mae": [80.0], "rmse": [110.0], "mape": [3.0], "smape": [3.0]}).to_csv(
        results / "metrics_overall.csv", index=False
    )
    pd.DataFrame({"model": ["lightgbm", "lightgbm"], "horizon": [1, 2], "mae": [70.0, 75.0]}).to_csv(
        results / "metrics_by_horizon.csv", index=False
    )
    pd.DataFrame({"model": ["lightgbm"], "horizon": [1], "coverage": [0.8], "width": [20.0]}).to_csv(
        results / "coverage_by_horizon.csv", index=False
    )
    prediction_rows().to_csv(results / "predictions.csv.gz", index=False)
    pd.DataFrame({"strategy": ["no_battery", "lightgbm"], "total_cost": [1000.0, 950.0]}).to_csv(
        results / "dispatch_savings.csv", index=False
    )
    pd.DataFrame({"feature": ["seasonal_24", "lag_0", "roll_mean_6"], "gain": [8.0, 1.5, 0.5], "share": [0.8, 0.15, 0.05], "family": ["seasonal lags", "recent lags", "rolling statistics"]}).to_csv(
        results / "feature_importance.csv", index=False
    )
    pd.DataFrame({"family": ["seasonal lags", "recent lags"], "share": [0.8, 0.2]}).to_csv(
        results / "feature_importance_families.csv", index=False
    )
    return Settings(artifact_dir=artifacts, results_dir=results)


@pytest.fixture
def client(populated: Settings) -> TestClient:
    return TestClient(create_app(populated))


@pytest.fixture
def empty_client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(artifact_dir=tmp_path / "a", results_dir=tmp_path / "r")))


def test_health_reports_available_data(client, empty_client):
    assert client.get("/health").json()["forecast_available"] is True
    body = empty_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["forecast_available"] is False


def test_forecast_and_intervals(client):
    forecast = client.get("/api/forecast").json()
    assert len(forecast["point"]) == 24
    intervals = client.get("/api/intervals").json()
    assert intervals["nominal_coverage"] == 0.8
    assert intervals["width"][0] == 22.0


def test_unknown_target_is_rejected(client):
    assert client.get("/api/forecast", params={"target": "wind"}).status_code == 422


def test_missing_forecast_is_404(empty_client):
    assert empty_client.get("/api/forecast").status_code == 404
    assert empty_client.get("/api/dispatch").status_code == 404


def test_history_is_trimmed(client):
    body = client.get("/api/history", params={"hours": 10}).json()
    assert len(body["values"]) == 10


def test_backtest_tables(client):
    summary = client.get("/api/backtest/summary").json()
    assert summary["metrics"][0]["model"] == "lightgbm"
    horizon = client.get("/api/backtest/horizon", params={"model": "lightgbm"}).json()
    assert [row["horizon"] for row in horizon["errors"]] == [1, 2]
    assert horizon["coverage"][0]["coverage"] == 0.8


def test_backtest_series_and_origins(client):
    origins = client.get("/api/backtest/origins", params={"model": "lightgbm"}).json()
    assert len(origins["origins"]) == 3
    assert origins["models"] == ["lightgbm", "naive_168"]
    series = client.get("/api/backtest/series", params={"model": "lightgbm"}).json()
    assert series["horizon"][0] == 1 and len(series["actual"]) == 24
    assert series["lower"][0] == 90.0
    assert client.get("/api/backtest/series", params={"model": "ets"}).status_code == 404
    bad = client.get("/api/backtest/series", params={"model": "lightgbm", "origin": "2030-01-01"})
    assert bad.status_code == 404


def test_dispatch_routes(client):
    schedule = client.get("/api/dispatch").json()
    assert schedule["battery"]["power_kw"] == 50.0
    savings = client.get("/api/dispatch/savings").json()
    assert savings["strategies"][1]["total_cost"] == 950.0


def test_models_route_defaults_to_empty(empty_client):
    assert empty_client.get("/api/models").json() == {"runs": []}


def test_importance_route(client, empty_client):
    body = client.get("/api/importance", params={"top": 2}).json()
    assert body["families"][0]["family"] == "seasonal lags"
    assert [row["feature"] for row in body["features"]] == ["seasonal_24", "lag_0"]
    assert empty_client.get("/api/importance").status_code == 404
