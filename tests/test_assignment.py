import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from or_toolkit import solve_assignment


def test_small():
    r = solve_assignment([[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8], [7, 6, 9, 4]])
    assert r.success and r.total == 13


def test_forbidden_and_infeasible():
    assert solve_assignment([[np.inf, 1], [2, np.inf]]).total == 3
    assert solve_assignment([[np.inf, np.inf], [1, 2]]).status == "infeasible"


@pytest.mark.parametrize("seed", range(80))
def test_vs_scipy(seed):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(1, 9), rng.integers(1, 9)
    C = rng.integers(-20, 50, size=(n, m)).astype(float)
    maximize = bool(seed % 2)
    r = solve_assignment(C, maximize=maximize)
    ri, ci = linear_sum_assignment(C, maximize=maximize)
    assert r.success and r.total == pytest.approx(C[ri, ci].sum())
    assert len(r.pairs) == min(n, m)
    assert len({j for _, j in r.pairs}) == len(r.pairs)
