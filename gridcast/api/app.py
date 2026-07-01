from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from gridcast import __version__
from gridcast.api import schemas
from gridcast.config import Settings, get_settings
from gridcast.store import ArtifactStore, ResultsStore

TARGETS = ("load_kw", "solar_kw")


def records(table: pd.DataFrame | None) -> list[dict]:
    if table is None:
        return []
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def create_app(settings: Settings | None = None, dashboard_dir: Path | None = None) -> FastAPI:
    settings = settings or get_settings()
    artifacts = ArtifactStore(settings.artifact_dir)
    results = ResultsStore(settings.results_dir)
    app = FastAPI(title="GridCast", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

    def require_forecast(target: str) -> dict:
        if target not in TARGETS:
            raise HTTPException(status_code=422, detail=f"target must be one of {TARGETS}")
        payload = artifacts.read_json(artifacts.forecast_name(target))
        if payload is None:
            raise HTTPException(status_code=404, detail="no forecast has been published yet")
        return payload

    def require_table(name: str, tag: str = "") -> pd.DataFrame:
        table = results.table(name, tag)
        if table is None:
            raise HTTPException(status_code=404, detail=f"backtest table {name} not found")
        return table

    def require_predictions(tag: str) -> pd.DataFrame:
        predictions = results.predictions(tag)
        if predictions is None:
            raise HTTPException(status_code=404, detail="no backtest predictions found")
        return predictions

    @app.get("/health", response_model=schemas.Health)
    def health() -> schemas.Health:
        return schemas.Health(
            status="ok",
            forecast_available=artifacts.read_json(artifacts.forecast_name("load_kw")) is not None,
            backtest_available=results.table("metrics_overall") is not None,
            version=__version__,
        )

    @app.get("/api/forecast", response_model=schemas.ForecastResponse)
    def forecast(target: str = "load_kw") -> dict:
        return require_forecast(target)

    @app.get("/api/intervals", response_model=schemas.IntervalResponse)
    def intervals(target: str = "load_kw") -> schemas.IntervalResponse:
        payload = require_forecast(target)
        spec = payload["interval"]
        lower = np.array(payload["lower"])
        upper = np.array(payload["upper"])
        return schemas.IntervalResponse(
            target=target,
            origin=payload["origin"],
            nominal_coverage=round(spec["upper_level"] - spec["lower_level"], 4),
            timestamps=payload["timestamps"],
            lower=lower.tolist(),
            upper=upper.tolist(),
            width=(upper - lower).round(2).tolist(),
        )

    @app.get("/api/history")
    def history(target: str = "load_kw", hours: int = Query(168, ge=1, le=24 * 30)) -> dict:
        block = require_forecast(target)["history"]
        return {"timestamps": block["timestamps"][-hours:], "values": block["values"][-hours:]}

    @app.get("/api/backtest/summary")
    def backtest_summary(tag: str = "") -> dict:
        return {
            "meta": results.meta(tag),
            "metrics": records(require_table("metrics_overall", tag)),
            "coverage": records(results.table("coverage_overall", tag)),
            "timing": records(results.table("timing", tag)),
            "by_fold": records(results.table("metrics_by_fold", tag)),
        }

    @app.get("/api/backtest/horizon")
    def backtest_horizon(tag: str = "", model: str | None = None) -> dict:
        errors = require_table("metrics_by_horizon", tag)
        coverage = results.table("coverage_by_horizon", tag)
        if model:
            errors = errors[errors["model"] == model]
            if coverage is not None:
                coverage = coverage[coverage["model"] == model]
        return {"errors": records(errors), "coverage": records(coverage)}

    @app.get("/api/backtest/origins")
    def backtest_origins(tag: str = "", model: str | None = None) -> dict:
        predictions = require_predictions(tag)
        models = sorted(predictions["model"].unique())
        if model:
            predictions = predictions[predictions["model"] == model]
        origins = sorted(predictions["origin"].unique())
        return {"origins": [pd.Timestamp(o).isoformat() for o in origins], "models": models}

    @app.get("/api/backtest/series", response_model=schemas.BacktestSeries)
    def backtest_series(model: str, origin: str | None = None, tag: str = "") -> schemas.BacktestSeries:
        subset = require_predictions(tag)
        subset = subset[subset["model"] == model]
        if subset.empty:
            raise HTTPException(status_code=404, detail=f"model {model} not in backtest")
        chosen = pd.Timestamp(origin) if origin else subset["origin"].max()
        block = subset[subset["origin"] == chosen].sort_values("horizon")
        if block.empty:
            raise HTTPException(status_code=404, detail="origin not found")
        has_interval = {"q10", "q90"} <= set(block.columns)
        return schemas.BacktestSeries(
            model=model,
            origin=pd.Timestamp(chosen).isoformat(),
            horizon=block["horizon"].tolist(),
            actual=block["actual"].tolist(),
            point=block["point"].tolist(),
            lower=block["q10"].tolist() if has_interval else None,
            upper=block["q90"].tolist() if has_interval else None,
        )

    @app.get("/api/dispatch", response_model=schemas.DispatchResponse)
    def dispatch() -> dict:
        payload = artifacts.read_json(artifacts.dispatch_name())
        if payload is None:
            raise HTTPException(status_code=404, detail="no dispatch schedule has been published yet")
        return payload

    @app.get("/api/dispatch/savings")
    def dispatch_savings() -> dict:
        table = results.table("dispatch_savings")
        if table is None:
            raise HTTPException(status_code=404, detail="dispatch evaluation has not been run")
        return {"strategies": records(table)}

    @app.get("/api/models")
    def models() -> dict:
        return artifacts.read_json("retrain_latest.json") or {"runs": []}

    if dashboard_dir is not None and dashboard_dir.exists():
        app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")
    return app


def create_default_app() -> FastAPI:
    default_dashboard = Path(__file__).resolve().parents[2] / "dashboard"
    return create_app(dashboard_dir=default_dashboard)
