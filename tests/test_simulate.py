import numpy as np
import pandas as pd
import pytest

from gridcast.dispatch.battery import Battery
from gridcast.dispatch.optimizer import DispatchPlan
from gridcast.dispatch.simulate import (
    compare_strategies,
    demand_charge,
    execute_plan,
    forecasts_from_predictions,
    simulate_strategy,
)
from gridcast.dispatch.tariff import Tariff

BATTERY = Battery(capacity_kwh=400.0, power_kw=100.0, round_trip_efficiency=0.9)
TARIFF = Tariff()


def daily_load(days: int = 10) -> pd.Series:
    index = pd.date_range("2024-03-04", periods=24 * days, freq="1h")
    hour = index.hour.to_numpy()
    load = 500 + 200 * np.exp(-((hour - 19) ** 2) / 8)
    return pd.Series(load, index=index, name="load_kw")


def perfect_forecasts(load: pd.Series) -> dict:
    return {load.index[i]: load.iloc[i : i + 24].to_numpy() for i in range(0, len(load) - 23, 24)}


def test_tariff_prices_by_hour_and_weekday():
    index = pd.date_range("2024-03-04", periods=24 * 7, freq="1h")
    prices = pd.Series(TARIFF.energy_prices(index), index=index)
    assert prices.loc["2024-03-04 18:00"] == TARIFF.peak
    assert prices.loc["2024-03-04 03:00"] == TARIFF.off_peak
    assert prices.loc["2024-03-04 10:00"] == TARIFF.shoulder
    assert prices.loc["2024-03-09 18:00"] == TARIFF.off_peak


def test_demand_charge_is_prorated_to_the_covered_period():
    grid = pd.Series(100.0, index=pd.date_range("2024-03-01", periods=24 * 31, freq="1h"))
    assert demand_charge(grid, 10.0) == pytest.approx(1000.0)
    assert demand_charge(grid.iloc[: 24 * 15], 10.0) == pytest.approx(1000.0 * 15 / 31)


def test_execution_clips_to_state_of_charge_and_no_export():
    plan = DispatchPlan(np.zeros(3), np.full(3, 100.0), np.zeros(3), 0.0, 0.0)
    grid, net, soc = execute_plan(plan, np.array([30.0, 30.0, 30.0]), BATTERY, BATTERY.initial_soc_kwh)
    assert (grid >= -1e-9).all()
    assert soc >= BATTERY.min_soc_kwh - 1e-9


def test_perfect_foresight_cuts_cost():
    load = daily_load()
    base = simulate_strategy("none", load, None, TARIFF, BATTERY)
    shifted = simulate_strategy("oracle", load, perfect_forecasts(load), TARIFF, BATTERY)
    assert shifted.total_cost < base.total_cost
    assert shifted.peak_kw < base.peak_kw


def test_comparison_table_has_reference_columns():
    load = daily_load()
    forecasts = perfect_forecasts(load)
    noisy = {start: values * 1.1 for start, values in forecasts.items()}
    table = compare_strategies(load, {"good": forecasts, "biased": noisy}, TARIFF, BATTERY, reference="biased")
    assert set(table["strategy"]) == {"no_battery", "perfect_foresight", "good", "biased"}
    row = table.set_index("strategy")
    assert row.loc["no_battery", "saving_vs_no_battery"] == pytest.approx(0.0)
    assert row.loc["biased", "saving_vs_reference"] == pytest.approx(0.0)
    assert row.loc["perfect_foresight", "saving_vs_no_battery"] > 0


def test_forecasts_from_predictions_filters_plan_hour():
    origins = pd.date_range("2024-03-04 00:00", periods=3, freq="6h")
    rows = [
        {"model": "m", "origin": origin, "horizon": h, "point": float(h)}
        for origin in origins
        for h in range(1, 25)
    ]
    forecasts = forecasts_from_predictions(pd.DataFrame(rows), "m")
    assert list(forecasts) == [pd.Timestamp("2024-03-04 01:00")]
    assert len(forecasts[pd.Timestamp("2024-03-04 01:00")]) == 24
