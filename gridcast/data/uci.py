from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

QUARTERS_PER_HOUR = 4


def read_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
        dtype=np.float32,
    )
    frame.index.name = "timestamp"
    return frame


def to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.resample("1h").mean()


def active_clients(hourly: pd.DataFrame, start: str, max_zero_share: float = 0.02) -> list[str]:
    window = hourly.loc[start:]
    zero_share = (window <= 0).mean()
    return [str(name) for name in zero_share[zero_share <= max_zero_share].index]


def aggregate(hourly: pd.DataFrame, clients: list[str], start: str) -> pd.Series:
    total = hourly.loc[start:, clients].sum(axis=1, min_count=len(clients))
    total.name = "load_kw"
    return total


def load_microgrid(path: Path, n_clients: int = 30, start: str = "2012-01-01") -> pd.Series:
    hourly = to_hourly(read_raw(path))
    chosen = active_clients(hourly, start)[:n_clients]
    return aggregate(hourly, chosen, start)
