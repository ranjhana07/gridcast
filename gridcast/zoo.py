from __future__ import annotations

from collections.abc import Callable

from gridcast.features import FeatureSpec, solar_spec
from gridcast.models.base import Forecaster
from gridcast.models.calibrated import ConformalForecaster
from gridcast.models.deep import DeepConfig, QuantileSeq2Seq
from gridcast.models.ets import ETSForecaster
from gridcast.models.gbm import DirectLightGBM
from gridcast.models.naive import SeasonalNaive

MODEL_NAMES = ("naive_24", "naive_168", "ets", "lightgbm", "lstm_quantile")


def inner_factory(name: str, target: str, horizon: int) -> Callable[[], Forecaster]:
    if name == "naive_24":
        return lambda: SeasonalNaive(target, horizon, period=24)
    if name == "naive_168":
        return lambda: SeasonalNaive(target, horizon, period=168)
    if name == "ets":
        return lambda: ETSForecaster(target, horizon, window_days=14)
    if name == "lightgbm":
        spec = solar_spec(horizon) if target == "solar_kw" else FeatureSpec(target=target, horizon=horizon)
        return lambda: DirectLightGBM(spec)
    if name == "lstm_quantile":
        return lambda: QuantileSeq2Seq(target, horizon, DeepConfig())
    raise KeyError(f"unknown model {name!r}")


CALIBRATION_STRIDE = {"ets": 12}


def build_factories(
    names: tuple[str, ...] | list[str],
    target: str = "load_kw",
    horizon: int = 24,
    alpha: float = 0.2,
) -> dict[str, Callable[[], ConformalForecaster]]:
    factories = {}
    for name in names:
        inner = inner_factory(name, target, horizon)
        stride = CALIBRATION_STRIDE.get(name, 3)
        factories[name] = (
            lambda inner=inner, name=name, stride=stride: ConformalForecaster(
                inner, alpha=alpha, stride=stride, name=name
            )
        )
    return factories
