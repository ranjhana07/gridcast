from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.metrics import POINT_METRICS


def point_metrics(group: pd.DataFrame) -> pd.Series:
    actual = group["actual"].to_numpy()
    predicted = group["point"].to_numpy()
    return pd.Series({name: float(fn(actual, predicted)) for name, fn in POINT_METRICS.items()})


def metrics_by_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    table = predictions.groupby(["model", "horizon"]).apply(point_metrics, include_groups=False)
    return table.reset_index()


def metrics_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    table = predictions.groupby("model").apply(point_metrics, include_groups=False)
    return table.reset_index()


def metrics_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    table = predictions.groupby(["model", "fold"]).apply(point_metrics, include_groups=False)
    return table.reset_index()


def interval_columns(predictions: pd.DataFrame) -> tuple[str, str] | None:
    if {"q10", "q90"} <= set(predictions.columns):
        return "q10", "q90"
    return None


def coverage_table(predictions: pd.DataFrame, lower: str = "q10", upper: str = "q90") -> pd.DataFrame:
    rows = []
    frame = predictions.dropna(subset=[lower, upper])
    for (model, horizon), group in frame.groupby(["model", "horizon"]):
        inside = (group["actual"] >= group[lower]) & (group["actual"] <= group[upper])
        rows.append(
            {
                "model": model,
                "horizon": horizon,
                "coverage": float(inside.mean()),
                "width": float((group[upper] - group[lower]).mean()),
            }
        )
    return pd.DataFrame(rows)


def coverage_overall(predictions: pd.DataFrame, lower: str = "q10", upper: str = "q90") -> pd.DataFrame:
    table = coverage_table(predictions, lower, upper)
    if table.empty:
        return table
    return table.groupby("model")[["coverage", "width"]].mean().reset_index()


def timing_summary(timings: pd.DataFrame) -> pd.DataFrame:
    grouped = timings.groupby("model")
    summary = grouped[["fit_seconds", "predict_seconds"]].sum()
    summary["origins"] = grouped["origins"].sum()
    summary["predict_ms_per_origin"] = 1000 * summary["predict_seconds"] / summary["origins"].replace(0, np.nan)
    return summary.reset_index()
