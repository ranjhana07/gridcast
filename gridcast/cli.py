from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from gridcast.backtest import BacktestConfig
from gridcast.benchmark import run_benchmark
from gridcast.config import get_settings
from gridcast.data import dataset
from gridcast.dispatch.battery import Battery
from gridcast.dispatch.evaluate import evaluate_dispatch, write_dispatch_report
from gridcast.dispatch.tariff import Tariff
from gridcast.flows import default_jobs, forecast_flow, retrain_flow
from gridcast.scheduler import serve as serve_jobs
from gridcast.zoo import MODEL_NAMES

app = typer.Typer(help="GridCast forecasting and dispatch toolkit", no_args_is_help=True)


@app.command("build-data")
def build_data(
    raw: Path = typer.Option(..., help="Path to LD2011_2014.txt"),
    clients: int = typer.Option(30, help="Number of meters to aggregate"),
    start: str = typer.Option("2012-01-01", help="First day to keep"),
    output: Path = typer.Option(None, help="Where to write the modelling frame"),
) -> None:
    settings = get_settings()
    frame = dataset.build_from_uci(raw, settings, n_clients=clients, start=start)
    target = output or settings.processed_dir / "microgrid_hourly.csv"
    dataset.save_frame(frame, target)
    typer.echo(f"wrote {len(frame)} rows to {target}")


@app.command("benchmark")
def benchmark(
    models: list[str] = typer.Option(list(MODEL_NAMES), "--model", "-m", help="Models to evaluate"),
    target: str = typer.Option("load_kw", help="load_kw or solar_kw"),
    folds: int = typer.Option(4, help="Number of rolling-origin folds"),
    fold_days: int = typer.Option(21, help="Length of each test fold in days"),
    stride: int = typer.Option(6, help="Hours between forecast origins"),
    data: Path = typer.Option(None, help="Modelling frame CSV"),
    output: Path = typer.Option(None, help="Directory for result tables"),
    tag: str = typer.Option("", help="Suffix for the result files"),
) -> None:
    settings = get_settings()
    frame = dataset.load_frame(data or settings.processed_dir / "microgrid_hourly.csv")
    config = BacktestConfig(
        target=target,
        horizon=settings.horizon,
        n_folds=folds,
        fold_hours=fold_days * 24,
        stride=stride,
    )
    tables = run_benchmark(frame, models, config, output or settings.results_dir, tag=tag)
    typer.echo(tables["metrics_overall"].to_string(index=False))
    typer.echo(tables["coverage_overall"].to_string(index=False))


@app.command("dispatch-eval")
def dispatch_eval(
    predictions: Path = typer.Option(None, help="predictions.csv.gz from a benchmark run"),
    data: Path = typer.Option(None, help="Modelling frame CSV"),
    model: list[str] = typer.Option(["lightgbm"], "--model", "-m", help="Forecast models to dispatch with"),
    reference: str = typer.Option("naive_168", help="Model used as the naive-forecast baseline"),
    output: Path = typer.Option(None, help="Where to write the comparison table"),
) -> None:
    settings = get_settings()
    frame = dataset.load_frame(data or settings.processed_dir / "microgrid_hourly.csv")
    forecasts = pd.read_csv(predictions or settings.results_dir / "predictions.csv.gz", parse_dates=["origin"])
    battery = Battery.sized_for(frame["load_kw"].to_numpy())
    tariff = Tariff()
    names = list(dict.fromkeys([reference, *model]))
    table = evaluate_dispatch(frame, forecasts, names, reference, tariff, battery)
    write_dispatch_report(table, battery, tariff, output or settings.results_dir / "dispatch_savings.csv")
    typer.echo(table.to_string(index=False))


@app.command("retrain")
def retrain(
    model: str = typer.Option("lightgbm", help="Model to train"),
    target: list[str] = typer.Option(["load_kw"], "--target", "-t"),
    tolerance: float = typer.Option(0.02, help="Allowed relative regression before promotion is refused"),
) -> None:
    reports = retrain_flow(get_settings(), tuple(target), model, tolerance)
    for report in reports:
        verdict = "promoted" if report["promoted"] else "kept production"
        typer.echo(f"{report['target']}: version {report['version']} {verdict} ({report['reason']})")


@app.command("forecast")
def forecast(
    model: str = typer.Option("lightgbm", help="Registered model to use"),
    target: list[str] = typer.Option(["load_kw"], "--target", "-t"),
    as_of: str = typer.Option(None, help="Forecast origin, defaults to the last observation"),
) -> None:
    origin = pd.Timestamp(as_of) if as_of else None
    payloads = forecast_flow(get_settings(), tuple(target), model, origin)
    for name, payload in payloads.items():
        typer.echo(f"{name}: {len(payload['point'])} hours from {payload['origin']}")


@app.command("schedule")
def schedule(poll_seconds: float = typer.Option(30.0)) -> None:
    serve_jobs(default_jobs(get_settings()), poll_seconds)


if __name__ == "__main__":
    app()
