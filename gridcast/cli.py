from __future__ import annotations

from pathlib import Path

import typer

from gridcast.backtest import BacktestConfig
from gridcast.benchmark import run_benchmark
from gridcast.config import get_settings
from gridcast.data import dataset
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


if __name__ == "__main__":
    app()
