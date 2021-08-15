import numpy as np
import pytest
from scipy.optimize import linprog

from or_toolkit import solve_transportation


def reference(cost, supply, demand):
    cost = np.asarray(cost, float)
    m, n = cost.shape
    big = 1e6
    c = np.where(np.isfinite(cost), cost, big).ravel()
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for i in range(m):
        row = np.zeros((m, n)); row[i] = 1
        (A_ub if sum(supply) > sum(demand) else A_eq).append(row.ravel())
        (b_ub if sum(supply) > sum(demand) else b_eq).append(supply[i])
    for j in range(n):
        col = np.zeros((m, n)); col[:, j] = 1
        (A_ub if sum(supply) < sum(demand) else A_eq).append(col.ravel())
        (b_ub if sum(supply) < sum(demand) else b_eq).append(demand[j])
    # Whichever side is short must be fully used; the other side is <=.
    if sum(supply) > sum(demand):
        A_ub, b_ub = A_ub[:m], b_ub[:m]
    elif sum(supply) < sum(demand):
        A_ub, b_ub = A_ub[:n], b_ub[:n]
    r = linprog(c, A_ub=np.array(A_ub) if A_ub else None, b_ub=b_ub or None,
                A_eq=np.array(A_eq) if A_eq else None, b_eq=b_eq or None, method="highs")
    return r.fun


def test_classic_example():
    cost = [[19, 30, 50, 10], [70, 30, 40, 60], [40, 8, 70, 20]]
    supply, demand = [7, 9, 18], [5, 8, 7, 14]
    for method in ("vogel", "least_cost", "nwc"):
        r = solve_transportation(cost, supply, demand, method=method)
        assert r.success and r.total_cost == pytest.approx(743)
        np.testing.assert_allclose(r.allocation.sum(axis=1), supply)
        np.testing.assert_allclose(r.allocation.sum(axis=0), demand)


def test_vogel_start_is_never_worse_than_nwc_here():
    cost = [[19, 30, 50, 10], [70, 30, 40, 60], [40, 8, 70, 20]]
    supply, demand = [7, 9, 18], [5, 8, 7, 14]
    v = solve_transportation(cost, supply, demand, "vogel")
    n = solve_transportation(cost, supply, demand, "nwc")
    assert v.initial_cost <= n.initial_cost and v.iterations <= n.iterations


def test_unbalanced_uses_dummy():
    r = solve_transportation([[4, 6], [5, 3]], [30, 40], [20, 30])
    assert r.success and r.dummy == "sink" and r.dummy_flow.sum() == pytest.approx(20)


def test_blocked_route():
    r = solve_transportation([[np.inf, 2], [3, 4]], [10, 10], [10, 10])
    assert r.success and r.allocation[0, 0] == 0 and r.total_cost == pytest.approx(2 * 10 + 3 * 10)


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("method", ["vogel", "least_cost", "nwc"])
def test_random_vs_linprog(seed, method):
    rng = np.random.default_rng(seed)
    m, n = rng.integers(2, 6), rng.integers(2, 6)
    cost = rng.integers(1, 30, size=(m, n)).astype(float)
    supply = rng.integers(1, 20, size=m).astype(float)
    demand = rng.integers(1, 20, size=n).astype(float)
    if seed % 3 == 0:                          # balance exactly (highly degenerate instances)
        demand *= supply.sum() / demand.sum()
        demand = np.round(demand)
        demand[0] += supply.sum() - demand.sum()
        if demand.min() <= 0:
            demand = np.maximum(demand, 1); supply = supply.copy(); supply[0] += demand.sum() - supply.sum()
    r = solve_transportation(cost, supply, demand, method=method)
    assert r.success
    assert r.total_cost == pytest.approx(reference(cost, list(supply), list(demand)), abs=1e-6)
