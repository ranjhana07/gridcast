from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    rows: int
    missing: int
    duplicates: int
    non_positive: int
    outliers: int
    longest_gap: int
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def longest_gap(mask: np.ndarray) -> int:
    best = run = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def validate_series(
    series: pd.Series,
    freq: str = "1h",
    max_missing_share: float = 0.05,
    allow_non_positive: bool = False,
) -> ValidationReport:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValidationError("series must be indexed by timestamps")
    issues: list[str] = []
    duplicates = int(series.index.duplicated().sum())
    if duplicates:
        issues.append(f"{duplicates} duplicated timestamps")
    if not series.index.is_monotonic_increasing:
        issues.append("index is not sorted")
    expected = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    missing_mask = ~series.reindex(expected).notna().to_numpy()
    missing = int(missing_mask.sum())
    if missing / max(len(expected), 1) > max_missing_share:
        issues.append(f"{missing} of {len(expected)} hours missing")
    non_positive = 0 if allow_non_positive else int((series <= 0).sum())
    spread = series.std()
    outliers = 0
    if spread and not np.isnan(spread):
        outliers = int(((series - series.median()).abs() > 8 * spread).sum())
    return ValidationReport(
        rows=len(series),
        missing=missing,
        duplicates=duplicates,
        non_positive=non_positive,
        outliers=outliers,
        longest_gap=longest_gap(missing_mask),
        issues=issues,
    )


def clean_series(series: pd.Series, freq: str = "1h", allow_non_positive: bool = False) -> pd.Series:
    series = series[~series.index.duplicated(keep="first")].sort_index()
    if not allow_non_positive:
        series = series.mask(series <= 0)
    spread = series.std()
    if spread and not np.isnan(spread):
        series = series.mask((series - series.median()).abs() > 8 * spread)
    full = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    return series.reindex(full).rename_axis("timestamp").rename(series.name)


def fill_gaps(series: pd.Series, period: int = 24, max_interpolate: int = 3) -> pd.Series:
    filled = series.interpolate(limit=max_interpolate, limit_area="inside")
    remaining = filled.isna()
    if remaining.any():
        weekly = filled.shift(period * 7)
        daily = filled.shift(period)
        filled = filled.fillna(weekly).fillna(daily)
    filled = filled.interpolate(limit_area="inside").bfill().ffill()
    return filled


def prepare_series(series: pd.Series, freq: str = "1h", allow_non_positive: bool = False) -> pd.Series:
    cleaned = clean_series(series, freq, allow_non_positive)
    return fill_gaps(cleaned)
