from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from gridcast.dispatch.battery import Battery


class DispatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class DispatchPlan:
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    peak_kw: float
    objective: float

    @property
    def net(self) -> np.ndarray:
        return self.charge - self.discharge


def plan_dispatch(
    load: np.ndarray,
    prices: np.ndarray,
    battery: Battery,
    demand_rate: float,
    initial_soc: float | None = None,
    terminal_soc: float | None = None,
    peak_floor: float = 0.0,
) -> DispatchPlan:
    load = np.asarray(load, dtype=float)
    prices = np.asarray(prices, dtype=float)
    horizon = len(load)
    if len(prices) != horizon:
        raise ValueError("load and prices must have the same length")
    soc0 = battery.initial_soc_kwh if initial_soc is None else initial_soc
    final = soc0 if terminal_soc is None else terminal_soc
    eta = battery.one_way_efficiency

    n_vars = 3 * horizon + 1
    charge = slice(0, horizon)
    discharge = slice(horizon, 2 * horizon)
    soc = slice(2 * horizon, 3 * horizon)
    peak = 3 * horizon

    cost = np.zeros(n_vars)
    cost[charge] = prices + battery.wear_cost_per_kwh
    cost[discharge] = -prices + battery.wear_cost_per_kwh
    cost[peak] = demand_rate

    identity = sparse.identity(horizon, format="csr")
    carry = sparse.diags([np.ones(horizon), -np.ones(horizon - 1)], [0, -1], format="csr")
    equality = sparse.hstack(
        [-eta * identity, identity / eta, carry, sparse.csr_matrix((horizon, 1))], format="csr"
    )
    equality_rhs = np.zeros(horizon)
    equality_rhs[0] = soc0

    no_export = sparse.hstack(
        [-identity, identity, sparse.csr_matrix((horizon, horizon)), sparse.csr_matrix((horizon, 1))],
        format="csr",
    )
    under_peak = sparse.hstack(
        [identity, -identity, sparse.csr_matrix((horizon, horizon)), -np.ones((horizon, 1))],
        format="csr",
    )
    inequality = sparse.vstack([no_export, under_peak], format="csr")
    inequality_rhs = np.concatenate([load, -load])

    bounds = (
        [(0.0, battery.power_kw)] * horizon
        + [(0.0, battery.power_kw)] * horizon
        + [(battery.min_soc_kwh, battery.capacity_kwh)] * (horizon - 1)
        + [(max(battery.min_soc_kwh, min(final, battery.capacity_kwh)), battery.capacity_kwh)]
        + [(max(peak_floor, 0.0), None)]
    )
    result = linprog(
        cost,
        A_ub=inequality,
        b_ub=inequality_rhs,
        A_eq=equality,
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise DispatchError(result.message)
    solution = result.x
    return DispatchPlan(
        charge=solution[charge],
        discharge=solution[discharge],
        soc=solution[soc],
        peak_kw=float(solution[peak]),
        objective=float(result.fun),
    )
