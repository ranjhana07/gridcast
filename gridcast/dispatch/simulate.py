from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.dispatch.battery import Battery
from gridcast.dispatch.optimizer import DispatchPlan, plan_dispatch
from gridcast.dispatch.tariff import Tariff


@dataclass
class Outcome:
    strategy: str
    energy_cost: float
    demand_cost: float
    peak_kw: float
    grid: pd.Series
    throughput_kwh: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.energy_cost + self.demand_cost


def demand_charge(grid: pd.Series, rate: float) -> float:
    total = 0.0
    for (year, month), block in grid.groupby([grid.index.year, grid.index.month]):
        days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month
        covered = len(block) / (24 * days_in_month)
        total += rate * float(block.max()) * min(covered, 1.0)
    return total


def execute_plan(
    plan: DispatchPlan,
    actual: np.ndarray,
    battery: Battery,
    soc: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    eta = battery.one_way_efficiency
    grid = np.empty(len(actual))
    net = np.empty(len(actual))
    for step, demand in enumerate(actual):
        charge = plan.charge[step]
        discharge = plan.discharge[step]
        discharge = min(discharge, max(soc - battery.min_soc_kwh, 0.0) * eta, demand + charge)
        charge = min(charge, max(battery.capacity_kwh - soc, 0.0) / eta)
        soc = soc + eta * charge - discharge / eta
        net[step] = charge - discharge
        grid[step] = demand + charge - discharge
    return grid, net, soc


def simulate_strategy(
    strategy: str,
    load: pd.Series,
    forecasts: dict[pd.Timestamp, np.ndarray] | None,
    tariff: Tariff,
    battery: Battery,
    window: int = 24,
) -> Outcome:
    if forecasts is None:
        starts = list(load.index[::window])
    else:
        starts = sorted(forecasts)
    grid_parts: list[pd.Series] = []
    soc = battery.initial_soc_kwh
    running_peak: dict[tuple[int, int], float] = {}
    throughput = 0.0
    for start in starts:
        index = load.index[(load.index >= start)][:window]
        if len(index) < window:
            break
        actual = load.loc[index].to_numpy()
        if forecasts is None:
            grid = actual
        else:
            prices = tariff.energy_prices(index)
            key = (index[0].year, index[0].month)
            plan = plan_dispatch(
                forecasts[start],
                prices,
                battery,
                tariff.demand_rate,
                initial_soc=soc,
                terminal_soc=battery.initial_soc_kwh,
                peak_floor=running_peak.get(key, 0.0),
            )
            grid, net, soc = execute_plan(plan, actual, battery, soc)
            throughput += float(np.abs(net).sum())
        grid_series = pd.Series(grid, index=index)
        for month_key, block in grid_series.groupby([grid_series.index.year, grid_series.index.month]):
            running_peak[month_key] = max(running_peak.get(month_key, 0.0), float(block.max()))
        grid_parts.append(grid_series)
    grid_all = pd.concat(grid_parts)
    prices_all = tariff.energy_prices(grid_all.index)
    return Outcome(
        strategy=strategy,
        energy_cost=float((grid_all.to_numpy() * prices_all).sum()),
        demand_cost=demand_charge(grid_all, tariff.demand_rate),
        peak_kw=float(grid_all.max()),
        grid=grid_all,
        throughput_kwh=throughput,
    )


def forecasts_from_predictions(
    predictions: pd.DataFrame,
    model: str,
    plan_hour: int = 0,
    horizon: int = 24,
) -> dict[pd.Timestamp, np.ndarray]:
    subset = predictions[predictions["model"] == model]
    forecasts: dict[pd.Timestamp, np.ndarray] = {}
    for origin, block in subset.groupby("origin"):
        origin = pd.Timestamp(origin)
        if origin.hour != plan_hour:
            continue
        ordered = block.sort_values("horizon")
        if len(ordered) != horizon:
            continue
        forecasts[origin + pd.Timedelta(hours=1)] = ordered["point"].to_numpy()
    return forecasts


def compare_strategies(
    load: pd.Series,
    forecasts_by_name: dict[str, dict[pd.Timestamp, np.ndarray]],
    tariff: Tariff,
    battery: Battery,
    reference: str,
) -> pd.DataFrame:
    common = sorted(set.intersection(*(set(f) for f in forecasts_by_name.values())))
    trimmed = load.loc[common[0] : common[-1] + pd.Timedelta(hours=23)]
    baseline = simulate_strategy("no_battery", trimmed, None, tariff, battery)
    outcomes = [baseline]
    oracle = {start: trimmed.loc[start : start + pd.Timedelta(hours=23)].to_numpy() for start in common}
    outcomes.append(simulate_strategy("perfect_foresight", trimmed, oracle, tariff, battery))
    for name, forecasts in forecasts_by_name.items():
        subset = {start: forecasts[start] for start in common}
        outcomes.append(simulate_strategy(name, trimmed, subset, tariff, battery))
    rows = []
    reference_cost = next(o.total_cost for o in outcomes if o.strategy == reference)
    for outcome in outcomes:
        rows.append(
            {
                "strategy": outcome.strategy,
                "days": len(common),
                "energy_cost": outcome.energy_cost,
                "demand_cost": outcome.demand_cost,
                "total_cost": outcome.total_cost,
                "peak_kw": outcome.peak_kw,
                "saving_vs_no_battery": baseline.total_cost - outcome.total_cost,
                "saving_pct_vs_no_battery": 100 * (baseline.total_cost - outcome.total_cost) / baseline.total_cost,
                "saving_vs_reference": reference_cost - outcome.total_cost,
                "throughput_kwh": outcome.throughput_kwh,
            }
        )
    return pd.DataFrame(rows)
