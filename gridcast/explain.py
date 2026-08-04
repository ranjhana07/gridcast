from __future__ import annotations

import pandas as pd

from gridcast.features import CALENDAR_COLUMNS, WEATHER_COLUMNS, FeatureSpec
from gridcast.models.gbm import DirectLightGBM


def feature_family(name: str) -> str:
    if name.startswith("lag_") or name == "trend_24":
        return "recent lags"
    if name.startswith("roll_"):
        return "rolling statistics"
    if name.startswith("seasonal_"):
        return "seasonal lags"
    if name.startswith("origin_"):
        return "weather at origin"
    if name.startswith("future_"):
        base = name.removeprefix("future_")
        if base in WEATHER_COLUMNS:
            return "weather at target hour"
        if base in CALENDAR_COLUMNS:
            return "calendar at target hour"
        return "other known-future"
    return "other"


def importance_table(model: DirectLightGBM) -> pd.DataFrame:
    gain = model.importance()
    table = gain.rename("gain").reset_index().rename(columns={"index": "feature"})
    table["share"] = table["gain"] / table["gain"].sum()
    table["family"] = table["feature"].map(feature_family)
    return table


def family_table(table: pd.DataFrame) -> pd.DataFrame:
    grouped = table.groupby("family")["share"].sum().sort_values(ascending=False)
    return grouped.reset_index()


def fit_importance(frame: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    model = DirectLightGBM(spec or FeatureSpec()).fit(frame)
    return importance_table(model)
