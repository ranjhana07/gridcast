from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_pjm(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=[0], index_col=0)
    series = frame.iloc[:, 0].astype("float64")
    series.index.name = "timestamp"
    series = series[~series.index.duplicated(keep="first")].sort_index()
    series.name = "load_mw"
    return series
