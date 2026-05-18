from __future__ import annotations

import numpy as np

EPSILON = 1e-9


def mae(actual: np.ndarray, predicted: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    return np.mean(np.abs(actual - predicted), axis=axis)


def rmse(actual: np.ndarray, predicted: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    return np.sqrt(np.mean((actual - predicted) ** 2, axis=axis))


def mape(actual: np.ndarray, predicted: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    return 100 * np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), EPSILON), axis=axis)


def smape(actual: np.ndarray, predicted: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    scale = (np.abs(actual) + np.abs(predicted)) / 2
    return 100 * np.mean(np.abs(actual - predicted) / np.maximum(scale, EPSILON), axis=axis)


def pinball(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    diff = actual - predicted
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((actual >= lower) & (actual <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


POINT_METRICS = {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}
