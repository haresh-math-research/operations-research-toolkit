import numpy as np
import pytest
from scipy.optimize import linprog

from or_toolkit import solve_lp


def test_original_repo_problem():
    r = solve_lp([3, 2], [[2, 1], [1, 1]], [100, 80], maximize=True)
    assert r.success
    np.testing.assert_allclose(r.x, [20, 60])
    assert r.objective == pytest.approx(180)
    np.testing.assert_allclose(r.shadow_prices, [1, 1])


def test_infeasible_and_unbounded():
    assert solve_lp([1, 1], [[1, 1], [1, 1]], [1, 3], ["<=", ">="]).status == "infeasible"
    assert solve_lp([1, 1], [[1, -1]], [1], maximize=True).status == "unbounded"


def test_beale_cycling_example():
    # Classic LP on which the pure Dantzig rule cycles; optimum is -5/4.
    A = [[0.25, -8, -1, 9], [0.5, -12, -0.5, 3], [0, 0, 1, 0]]
    r = solve_lp([-0.75, 20, -0.5, 6], A, [0, 0, 1])
    assert r.success and r.objective == pytest.approx(-1.25)


def test_redundant_equalities():
    r = solve_lp([1, 2], [[1, 1], [2, 2], [1, 0]], [4, 8, 1], ["==", "==", ">="], maximize=True)
    assert r.success and r.objective == pytest.approx(7)   # x=(1,3); second row is redundant
    r = solve_lp([1, 2], [[1, 1], [2, 2], [1, 0]], [4, 8, 1], ["==", "==", ">="])
    assert r.success and r.objective == pytest.approx(4)   # x=(4,0)


def random_lp(rng, m, n):
    A = rng.integers(-3, 6, size=(m, n)).astype(float)
    x0 = rng.integers(0, 5, size=n).astype(float)
    senses = rng.choice(["<=", ">=", "=="], size=m, p=[0.5, 0.3, 0.2]).tolist()
    b = A @ x0
    b = np.where(np.array(senses) == "<=", b + rng.integers(0, 3, m), b)
    b = np.where(np.array(senses) == ">=", b - rng.integers(0, 3, m), b)
    A = np.vstack([A, np.ones(n)])
    b = np.append(b, 50)
    senses.append("<=")                                     # keeps the problem bounded
    return rng.integers(-5, 6, size=n).astype(float), A, b, senses


def to_linprog(c, A, b, senses, maximize):
    A = np.asarray(A); b = np.asarray(b); s = np.array(senses)
    ub, ge, eq = s == "<=", s == ">=", s == "=="
    A_ub = np.vstack([A[ub], -A[ge]])
    b_ub = np.concatenate([b[ub], -b[ge]])
    return linprog(-np.asarray(c) if maximize else c, A_ub=A_ub, b_ub=b_ub,
                   A_eq=A[eq] if eq.any() else None, b_eq=b[eq] if eq.any() else None, method="highs")


@pytest.mark.parametrize("seed", range(60))
def test_matches_scipy(seed):
    rng = np.random.default_rng(seed)
    m, n = rng.integers(2, 7), rng.integers(2, 8)
    c, A, b, senses = random_lp(rng, m, n)
    maximize = bool(seed % 2)
    mine = solve_lp(c, A, b, senses, maximize=maximize)
    ref = to_linprog(c, A, b, senses, maximize)
    assert ref.status == 0 and mine.success
    assert mine.objective == pytest.approx(-ref.fun if maximize else ref.fun, abs=1e-6)
    act = np.asarray(A) @ mine.x
    for a, s, r in zip(act, senses, b):
        assert (a <= r + 1e-6) if s == "<=" else (a >= r - 1e-6) if s == ">=" else abs(a - r) < 1e-6


@pytest.mark.parametrize("seed", range(40))
def test_sensitivity_by_perturbation(seed):
    rng = np.random.default_rng(1000 + seed)
    c, A, b, senses = random_lp(rng, 4, 5)
    maximize = bool(seed % 2)
    base = solve_lp(c, A, b, senses, maximize=maximize)
    if not base.success:
        pytest.skip("unbounded/infeasible draw")
    # Shadow prices: inside the RHS range the objective moves linearly with slope = shadow price.
    for i in range(len(b)):
        lo, hi = base.rhs_ranges[i]
        for target in (lo, hi):
            if not np.isfinite(target) or abs(target - b[i]) < 1e-9:
                continue
            delta = 0.9 * (target - b[i])
            b2 = np.array(b, float)
            b2[i] += delta
            r2 = solve_lp(c, A, b2, senses, maximize=maximize)
            assert r2.success
            assert r2.objective == pytest.approx(base.objective + base.shadow_prices[i] * delta, abs=1e-6)
    # Cost ranging: inside the range the current x stays optimal.
    for j in range(len(c)):
        lo, hi = base.cost_ranges[j]
        for target in (lo, hi):
            if not np.isfinite(target) or abs(target - c[j]) < 1e-9:
                continue
            c2 = np.array(c, float)
            c2[j] += 0.9 * (target - c[j])
            r2 = solve_lp(c2, A, b, senses, maximize=maximize)
            assert r2.success
            assert r2.objective == pytest.approx(float(c2 @ base.x), abs=1e-6)
