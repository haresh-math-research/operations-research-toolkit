import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp

from or_toolkit import solve_ilp


def test_textbook_ilp():
    # max 5x + 4y  s.t. 6x + 4y <= 24, x + 2y <= 6  -> LP opt (3, 1.5); ILP opt 20 at (4, 0)
    r = solve_ilp([5, 4], [[6, 4], [1, 2]], [24, 6], maximize=True)
    assert r.success and r.objective == pytest.approx(20)
    assert r.root_bound == pytest.approx(21)


def test_infeasible_integer_program():
    r = solve_ilp([1], [[2], [2]], [1, 1], [">=", "<="], maximize=True)   # 2x = 1 has no integer solution
    assert r.status == "infeasible"


@pytest.mark.parametrize("seed", range(25))
def test_knapsack_like_vs_milp(seed):
    rng = np.random.default_rng(seed)
    n, m = 6, 3
    A = rng.integers(1, 9, size=(m, n)).astype(float)
    b = rng.integers(15, 40, size=m).astype(float)
    c = rng.integers(1, 12, size=n).astype(float)
    ub = np.full(n, 3.0)
    A_full = np.vstack([A, np.eye(n)])
    b_full = np.concatenate([b, ub])
    mine = solve_ilp(c, A_full, b_full, maximize=True)
    ref = milp(-c, constraints=LinearConstraint(A, -np.inf, b), integrality=np.ones(n), bounds=Bounds(0, ub))
    assert mine.success and ref.success
    assert mine.objective == pytest.approx(-ref.fun, abs=1e-6)


def test_mixed_integer():
    # only x0 integer
    r = solve_ilp([1, 1], [[2, 2]], [3], maximize=True, integer_vars=[0])
    assert r.success and r.objective == pytest.approx(1.5)
