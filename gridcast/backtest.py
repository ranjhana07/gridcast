from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.models.base import Forecaster

ModelFactory = Callable[[], Forecaster]


@dataclass(frozen=True)
class BacktestConfig:
    target: str = "load_kw"
    horizon: int = 24
    n_folds: int = 4
    fold_hours: int = 24 * 28
    stride: int = 6

    def __post_init__(self) -> None:
        if self.fold_hours <= self.horizon:
            raise ValueError("fold_hours must exceed the horizon")
        if self.stride < 1:
            raise ValueError("stride must be positive")


@dataclass(frozen=True)
class Fold:
    number: int
    train_end: int
    test_end: int

    def origin_positions(self, horizon: int, stride: int) -> np.ndarray:
        first = self.train_end - 1
        last = self.test_end - horizon - 1
        return np.arange(first, last + 1, stride)


def make_folds(n_rows: int, config: BacktestConfig) -> Iterator[Fold]:
    first_test = n_rows - config.n_folds * config.fold_hours
    if first_test <= 24 * 7 * 8:
        raise ValueError("not enough history for the requested folds")
    for number in range(config.n_folds):
        start = first_test + number * config.fold_hours
        yield Fold(number=number, train_end=start, test_end=start + config.fold_hours)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    timings: pd.DataFrame


def collect_rows(
    model_name: str,
    fold: Fold,
    frame: pd.DataFrame,
    target: str,
    positions: np.ndarray,
    forecast,
) -> pd.DataFrame:
    horizon = forecast.horizon
    values = frame[target].to_numpy()
    steps = np.arange(1, horizon + 1)
    target_positions = positions[:, None] + steps[None, :]
    data = {
        "model": model_name,
        "fold": fold.number,
        "origin": np.repeat(frame.index[positions], horizon),
        "horizon": np.tile(steps, len(positions)),
        "actual": values[target_positions].ravel(),
        "point": forecast.point.ravel(),
    }
    for level, array in forecast.quantiles.items():
        data[f"q{int(round(level * 100)):02d}"] = array.ravel()
    return pd.DataFrame(data)


def run_backtest(
    frame: pd.DataFrame,
    factories: dict[str, ModelFactory],
    config: BacktestConfig,
) -> BacktestResult:
    prediction_parts: list[pd.DataFrame] = []
    timing_rows: list[dict] = []
    for fold in make_folds(len(frame), config):
        positions = fold.origin_positions(config.horizon, config.stride)
        origins = frame.index[positions]
        train = frame.iloc[: fold.train_end]
        window = frame.iloc[: fold.test_end]
        for name, factory in factories.items():
            model = factory()
            started = time.perf_counter()
            model.fit(train)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            forecast = model.predict(window, origins)
            predict_seconds = time.perf_counter() - started
            prediction_parts.append(collect_rows(name, fold, window, config.target, positions, forecast))
            timing_rows.append(
                {
                    "model": name,
                    "fold": fold.number,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "origins": len(positions),
                }
            )
    return BacktestResult(pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(timing_rows))
