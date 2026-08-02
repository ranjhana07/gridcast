from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> np.ndarray:
    n = scores.shape[0]
    if n == 0:
        raise ValueError("calibration set is empty")
    rank = min(math.ceil((n + 1) * (1 - alpha)), n)
    return np.sort(scores, axis=0)[rank - 1]


class SplitConformal:
    def __init__(self, alpha: float = 0.2) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha
        self.radius: np.ndarray | None = None

    def fit(self, actual: np.ndarray, point: np.ndarray) -> "SplitConformal":
        self.radius = conformal_quantile(np.abs(actual - point), self.alpha)
        return self

    def interval(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.radius is None:
            raise RuntimeError("calibrator has not been fitted")
        return point - self.radius, point + self.radius


class ConformalizedQuantile:
    def __init__(self, alpha: float = 0.2) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha
        self.correction: np.ndarray | None = None

    def fit(self, actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> "ConformalizedQuantile":
        scores = np.maximum(lower - actual, actual - upper)
        self.correction = conformal_quantile(scores, self.alpha)
        return self

    def interval(self, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.correction is None:
            raise RuntimeError("calibrator has not been fitted")
        return lower - self.correction, upper + self.correction


def online_corrections(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    positions: np.ndarray,
    alpha: float,
    window: int = 400,
) -> np.ndarray:
    order = np.argsort(positions, kind="stable")
    if not np.array_equal(order, np.arange(len(positions))):
        raise ValueError("origins must be sorted in time")
    n_origins, horizon = test_scores.shape
    corrections = np.empty((n_origins, horizon))
    for index in range(n_origins):
        for step in range(1, horizon + 1):
            known = int(np.searchsorted(positions, positions[index] - step, side="right"))
            pool = np.concatenate([calibration_scores[:, step - 1], test_scores[:known, step - 1]])
            corrections[index, step - 1] = conformal_quantile(pool[-window:, None], alpha)[0]
    return corrections
