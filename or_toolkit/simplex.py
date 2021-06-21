"""Two-phase tableau simplex method, implemented from scratch.

Solves      optimize  c^T x
            subject to  A[i] x  (<= | >= | ==)  b[i]      x >= 0

Features
--------
* Phase I / Phase II with artificial variables (handles <=, >=, == rows).
* Detects infeasibility and unboundedness.
* Dantzig's largest-coefficient rule, with automatic fallback to Bland's rule
  when a run of degenerate pivots is detected (guards against cycling).
* Post-optimal sensitivity analysis: shadow prices (duals), reduced costs,
  right-hand-side ranging and objective-coefficient ranging.

The dense tableau is O(m*n) per pivot, which is fine for teaching and for
problems up to a few hundred rows/columns.  It is not meant to compete with
HiGHS; the test-suite cross-checks it against ``scipy.optimize.linprog``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"
ITERATION_LIMIT = "iteration_limit"

_FLIP = {"<=": ">=", ">=": "<=", "==": "=="}


@dataclass
class LPResult:
    status: str
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    slack: Optional[np.ndarray] = None            # row slack/surplus (>= 0 when feasible)
    shadow_prices: Optional[np.ndarray] = None    # d(objective)/d(b_i)
    reduced_costs: Optional[np.ndarray] = None    # per structural variable
    basis: Optional[list] = None
    iterations: int = 0
    rhs_ranges: Optional[list] = None             # (lo, hi) of b_i keeping the basis optimal
    cost_ranges: Optional[list] = None            # (lo, hi) of c_j keeping the basis optimal

    @property
    def success(self) -> bool:
        return self.status == OPTIMAL

    def summary(self) -> str:
        if not self.success:
            return f"Status: {self.status}"
        lines = [
            f"Status: {self.status}  ({self.iterations} pivots)",
            f"Objective: {self.objective:.6g}",
            "x = " + np.array2string(self.x, precision=6, suppress_small=True),
        ]
        if self.shadow_prices is not None:
            lines.append("Shadow prices: " + np.array2string(self.shadow_prices, precision=6, suppress_small=True))
            lines.append("Reduced costs: " + np.array2string(self.reduced_costs, precision=6, suppress_small=True))
            lines.append("RHS ranges:  " + ", ".join(f"[{lo:.4g}, {hi:.4g}]" for lo, hi in self.rhs_ranges))
            lines.append("Cost ranges: " + ", ".join(f"[{lo:.4g}, {hi:.4g}]" for lo, hi in self.cost_ranges))
        return "\n".join(lines)


def _pivot(T: np.ndarray, basis: list, i: int, j: int) -> None:
    T[i] /= T[i, j]
    for k in range(T.shape[0]):
        if k != i and T[k, j] != 0.0:
            T[k] -= T[k, j] * T[i]
    T[np.abs(T) < 1e-13] = 0.0
    basis[i] = j


def _simplex_loop(T, basis, cost, allowed, tol, max_iter):
    """Run primal simplex on tableau T (in place).  Returns (status, pivots)."""
    m = T.shape[0]
    pivots = 0
    stall = 0
    while True:
        if pivots >= max_iter:
            return ITERATION_LIMIT, pivots
        r = cost - cost[basis] @ T[:, :-1]          # reduced costs (minimisation form)
        cand = np.flatnonzero((r < -tol) & allowed)
        if cand.size == 0:
            return OPTIMAL, pivots
        # Dantzig normally; Bland (lowest index) once we've stalled on degenerate pivots.
        j = cand[0] if stall > m else cand[np.argmin(r[cand])]
        col = T[:, j]
        pos = np.flatnonzero(col > tol)
        if pos.size == 0:
            return UNBOUNDED, pivots
        ratios = T[pos, -1] / col[pos]
        rmin = ratios.min()
        ties = pos[ratios <= rmin + tol]
        i = ties[np.argmin([basis[k] for k in ties])]   # Bland tie-break on leaving var
        stall = stall + 1 if rmin <= tol else 0
        _pivot(T, basis, i, j)
        pivots += 1


def solve_lp(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    senses: Optional[Sequence[str]] = None,
    maximize: bool = False,
    tol: float = 1e-9,
    max_iter: int = 50_000,
    sensitivity: bool = True,
) -> LPResult:
    """Solve an LP with the two-phase simplex method.  All variables are >= 0."""
    c = np.asarray(c, dtype=float)
    A0 = np.atleast_2d(np.asarray(A, dtype=float))
    b0 = np.asarray(b, dtype=float)
    m, n = A0.shape
    if c.shape != (n,) or b0.shape != (m,):
        raise ValueError("Shapes of c, A, b are inconsistent")
    senses = ["<="] * m if senses is None else list(senses)
    if len(senses) != m or any(s not in _FLIP for s in senses):
        raise ValueError("senses must be a list of '<=', '>=' or '==' with one entry per row")

    # Make every RHS non-negative.
    A_t, b_t, sen = A0.copy(), b0.copy(), list(senses)
    flipped = np.zeros(m, dtype=bool)
    for i in range(m):
        if b_t[i] < 0:
            A_t[i] *= -1
            b_t[i] *= -1
            sen[i] = _FLIP[sen[i]]
            flipped[i] = True

    n_slack = sum(s != "==" for s in sen)
    n_art = sum(s != "<=" for s in sen)
    art_start = n + n_slack
    N = art_start + n_art
    T = np.zeros((m, N + 1))
    T[:, :n] = A_t
    T[:, -1] = b_t
    basis = [0] * m
    ident = [0] * m                      # column that started as the identity column of row i
    sc, ac = n, art_start
    for i, s in enumerate(sen):
        if s == "<=":
            T[i, sc] = 1.0; basis[i] = ident[i] = sc; sc += 1
        elif s == ">=":
            T[i, sc] = -1.0; sc += 1
            T[i, ac] = 1.0; basis[i] = ident[i] = ac; ac += 1
        else:
            T[i, ac] = 1.0; basis[i] = ident[i] = ac; ac += 1

    c_min = -c if maximize else c
    total_pivots = 0

    # ---- Phase I: minimise the sum of artificials ------------------------------------
    if n_art:
        cost1 = np.zeros(N)
        cost1[art_start:] = 1.0
        allowed = np.zeros(N, dtype=bool)
        allowed[:art_start] = True
        status, p = _simplex_loop(T, basis, cost1, allowed, tol, max_iter)
        total_pivots += p
        if status == ITERATION_LIMIT:
            return LPResult(status, iterations=total_pivots)
        if cost1[basis] @ T[:, -1] > 1e-7 * max(1.0, np.abs(b_t).max()):
            return LPResult(INFEASIBLE, iterations=total_pivots)
        # Drive remaining (zero-level) artificials out of the basis where possible.
        for i in range(m):
            if basis[i] >= art_start:
                nz = np.flatnonzero(np.abs(T[i, :art_start]) > 1e-9)
                if nz.size:
                    _pivot(T, basis, i, nz[0])
                    total_pivots += 1
                # else: the row is redundant; the artificial stays basic at 0 and can never move.

    # ---- Phase II ---------------------------------------------------------------------
    cost2 = np.zeros(N)
    cost2[:n] = c_min
    allowed = np.zeros(N, dtype=bool)
    allowed[:art_start] = True
    status, p = _simplex_loop(T, basis, cost2, allowed, tol, max_iter)
    total_pivots += p
    if status != OPTIMAL:
        return LPResult(status, iterations=total_pivots)

    xfull = np.zeros(N)
    xfull[basis] = T[:, -1]
    x = xfull[:n]
    x = np.where(np.abs(x) < 1e-10, 0.0, x)
    obj = float(c @ x)

    act = A0 @ x
    slack = np.where(np.array(senses) == "<=", b0 - act, np.where(np.array(senses) == ">=", act - b0, 0.0))
    res = LPResult(OPTIMAL, x=x, objective=obj, slack=slack, basis=list(basis), iterations=total_pivots)
    if not sensitivity:
        return res

    # ---- Sensitivity analysis -----------------------------------------------------------
    sign_obj = -1.0 if maximize else 1.0
    y = cost2[basis] @ T[:, ident]                                   # duals of transformed problem
    y = np.where(np.abs(y) < 1e-12, 0.0, y)
    res.shadow_prices = y * np.where(flipped, -1.0, 1.0) * sign_obj + 0.0   # + 0.0 turns -0.0 into 0.0
    r = cost2 - cost2[basis] @ T[:, :-1]
    res.reduced_costs = np.where(np.abs(r[:n]) < 1e-10, 0.0, r[:n]) * sign_obj + 0.0

    # RHS ranging: keep B^-1 (b + delta e_i) >= 0.
    rhs = T[:, -1]
    rhs_ranges = []
    for i in range(m):
        col = T[:, ident[i]]
        with np.errstate(divide="ignore", invalid="ignore"):
            lim = -rhs / col
        lo = lim[col > tol].max() if np.any(col > tol) else -np.inf
        hi = lim[col < -tol].min() if np.any(col < -tol) else np.inf
        if flipped[i]:
            lo, hi = -hi, -lo
        rhs_ranges.append((b0[i] + lo, b0[i] + hi))
    res.rhs_ranges = rhs_ranges

    # Objective-coefficient ranging (minimisation form, then mapped back).
    nonbasic = [l for l in range(art_start) if l not in set(basis)]
    cost_ranges = []
    for j in range(n):
        if j in basis:
            k = basis.index(j)
            lo, hi = -np.inf, np.inf
            for l in nonbasic:
                alpha = T[k, l]
                if alpha > tol:
                    hi = min(hi, r[l] / alpha)
                elif alpha < -tol:
                    lo = max(lo, r[l] / alpha)
            rng = (c_min[j] + lo, c_min[j] + hi)
        else:
            rng = (c_min[j] - r[j], np.inf)
        cost_ranges.append((-rng[1], -rng[0]) if maximize else rng)
    res.cost_ranges = cost_ranges
    return res
