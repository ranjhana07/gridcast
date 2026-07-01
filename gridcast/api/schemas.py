from __future__ import annotations

from pydantic import BaseModel


class Health(BaseModel):
    status: str
    forecast_available: bool
    backtest_available: bool
    version: str


class HistoryBlock(BaseModel):
    timestamps: list[str]
    values: list[float]


class IntervalSpec(BaseModel):
    lower_level: float
    upper_level: float


class ForecastResponse(BaseModel):
    target: str
    model: str
    origin: str
    interval: IntervalSpec
    timestamps: list[str]
    point: list[float]
    lower: list[float]
    upper: list[float]
    history: HistoryBlock


class IntervalResponse(BaseModel):
    target: str
    origin: str
    nominal_coverage: float
    timestamps: list[str]
    lower: list[float]
    upper: list[float]
    width: list[float]


class BacktestSeries(BaseModel):
    model: str
    origin: str
    horizon: list[int]
    actual: list[float]
    point: list[float]
    lower: list[float] | None
    upper: list[float] | None


class DispatchResponse(BaseModel):
    origin: str
    timestamps: list[str]
    load: list[float]
    price: list[float]
    charge: list[float]
    discharge: list[float]
    soc: list[float]
    grid: list[float]
    peak_without_kw: float
    peak_with_kw: float
    cost_without: float
    cost_with: float
    battery: dict[str, float]
