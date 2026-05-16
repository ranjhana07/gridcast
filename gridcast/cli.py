from __future__ import annotations

from pathlib import Path

import typer

from gridcast.config import get_settings
from gridcast.data import dataset

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


if __name__ == "__main__":
    app()
