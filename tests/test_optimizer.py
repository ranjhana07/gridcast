import numpy as np
import pytest

from gridcast.dispatch.battery import Battery
from gridcast.dispatch.optimizer import plan_dispatch

BATTERY = Battery(capacity_kwh=100.0, power_kw=50.0, round_trip_efficiency=1.0, min_soc_share=0.0, initial_soc_share=0.0, wear_cost_per_kwh=0.0)


def test_charges_when_cheap_and_discharges_when_expensive():
    load = np.full(4, 100.0)
    prices = np.array([0.1, 0.1, 0.5, 0.5])
    plan = plan_dispatch(load, prices, BATTERY, demand_rate=0.0, initial_soc=0.0, terminal_soc=0.0)
    assert plan.charge[:2].sum() == pytest.approx(100.0)
    assert plan.discharge[2:].sum() == pytest.approx(100.0)
    assert plan.discharge[:2].sum() == pytest.approx(0.0, abs=1e-6)
    saving = 100 * (0.5 - 0.1)
    assert plan.objective == pytest.approx(-saving)


def test_flat_prices_leave_battery_idle_when_losses_exist():
    lossy = Battery(100.0, 50.0, round_trip_efficiency=0.8, min_soc_share=0.0, initial_soc_share=0.0, wear_cost_per_kwh=0.0)
    plan = plan_dispatch(np.full(6, 80.0), np.full(6, 0.2), lossy, demand_rate=0.0, initial_soc=0.0, terminal_soc=0.0)
    assert plan.charge.sum() == pytest.approx(0.0, abs=1e-6)
    assert plan.discharge.sum() == pytest.approx(0.0, abs=1e-6)


def test_peak_shaving_flattens_the_spike():
    load = np.array([100.0, 100.0, 200.0, 100.0])
    prices = np.full(4, 0.0)
    battery = Battery(100.0, 50.0, 1.0, 0.0, 1.0, 0.0)
    plan = plan_dispatch(load, prices, battery, demand_rate=1.0, initial_soc=100.0, terminal_soc=0.0)
    assert plan.peak_kw == pytest.approx(150.0)
    assert (load + plan.net).max() == pytest.approx(150.0)


def test_never_exports_to_the_grid():
    load = np.array([10.0, 10.0, 10.0])
    prices = np.array([0.1, 0.9, 0.9])
    battery = Battery(100.0, 50.0, 1.0, 0.0, 1.0, 0.0)
    plan = plan_dispatch(load, prices, battery, demand_rate=0.0, initial_soc=100.0, terminal_soc=0.0)
    assert (load + plan.net >= -1e-7).all()


def test_state_of_charge_respects_bounds_and_terminal_target():
    rng = np.random.default_rng(0)
    load = rng.uniform(50, 150, 24)
    prices = rng.uniform(0.1, 0.4, 24)
    battery = Battery(120.0, 40.0, 0.9, 0.1, 0.5, 0.004)
    plan = plan_dispatch(load, prices, battery, demand_rate=2.0)
    assert plan.soc.min() >= battery.min_soc_kwh - 1e-6
    assert plan.soc.max() <= battery.capacity_kwh + 1e-6
    assert plan.soc[-1] >= battery.initial_soc_kwh - 1e-6


def test_peak_floor_prevents_paying_below_the_running_peak():
    load = np.full(4, 50.0)
    plan = plan_dispatch(load, np.full(4, 0.2), BATTERY, demand_rate=5.0, initial_soc=0.0, terminal_soc=0.0, peak_floor=80.0)
    assert plan.peak_kw == pytest.approx(80.0)


def test_mismatched_inputs_rejected():
    with pytest.raises(ValueError):
        plan_dispatch(np.ones(3), np.ones(4), BATTERY, 1.0)
