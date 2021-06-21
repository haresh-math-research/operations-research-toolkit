"""Transportation problem: initial basic feasible solution + MODI (u-v) optimisation.

* Initial solutions: north-west corner, least cost, Vogel's approximation.
* Optimisation: MODI / stepping-stone pivots on the basis spanning tree.
* Unbalanced problems are balanced with a zero-cost dummy source/sink.
* Forbidden routes: pass ``np.inf`` as the cost.
* Degeneracy is handled by keeping the basis a spanning tree with m+n-1 cells.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

EPS = 1e-9


@dataclass
class TransportResult:
    status: str
    allocation: Optional[np.ndarray] = None
    total_cost: Optional[float] = None
    u: Optional[np.ndarray] = None            # row potentials of the (balanced) problem
    v: Optional[np.ndarray] = None            # column potentials
    iterations: int = 0
    initial_method: str = ""
    initial_cost: Optional[float] = None
    alternative_optima: bool = False
    dummy: Optional[str] = None               # 'source' (unmet demand) / 'sink' (unused supply) / None
    dummy_flow: Optional[np.ndarray] = None   # per real destination / source amount going to the dummy

    @property
    def success(self) -> bool:
        return self.status == "optimal"

    def summary(self) -> str:
        if not self.success:
            return f"Status: {self.status}"
        s = (f"Status: optimal  initial ({self.initial_method}) cost={self.initial_cost:.6g} "
             f"-> optimal cost={self.total_cost:.6g} in {self.iterations} MODI iterations\n"
             f"Allocation:\n{np.array2string(self.allocation, precision=6, suppress_small=True)}")
        if self.alternative_optima:
            s += "\nNote: alternative optimal solutions exist."
        if self.dummy:
            s += f"\nBalanced with a dummy {self.dummy}; leftover: {np.array2string(self.dummy_flow, precision=6)}"
        return s


def _pick_cell(cost, s, d, R, C, method):
    if method == "nwc":
        return min(R), min(C)
    if method == "least_cost":
        return min(((cost[i, j], -min(s[i], d[j]), i, j) for i in R for j in C))[2:]
    if method == "vogel":
        def penalty(vals):
            vals = sorted(vals)
            return vals[1] - vals[0] if len(vals) > 1 else vals[0]
        best = None
        for i in sorted(R):
            key = (penalty([cost[i, j] for j in C]), 1)
            if best is None or key > best[0]:
                best = (key, ("r", i))
        for j in sorted(C):
            key = (penalty([cost[i, j] for i in R]), 0)
            if key > best[0]:
                best = (key, ("c", j))
        kind, k = best[1]
        if kind == "r":
            j = min(C, key=lambda jj: (cost[k, jj], -min(s[k], d[jj]), jj))
            return k, j
        i = min(R, key=lambda ii: (cost[ii, k], -min(s[ii], d[k]), ii))
        return i, k
    raise ValueError("method must be 'vogel', 'least_cost' or 'nwc'")


def _initial_bfs(cost, supply, demand, method):
    m, n = cost.shape
    s, d = supply.astype(float).copy(), demand.astype(float).copy()
    x = np.zeros((m, n))
    basis = set()
    R, C = set(range(m)), set(range(n))
    while R and C:
        i, j = _pick_cell(cost, s, d, R, C, method)
        q = min(s[i], d[j])
        x[i, j] = q
        basis.add((i, j))
        s[i] -= q
        d[j] -= q
        if s[i] <= EPS and d[j] <= EPS:
            # Degenerate: strike out only one line so the basis keeps m+n-1 cells.
            (R.remove(i) if len(R) > 1 else C.remove(j))
        elif s[i] <= EPS:
            R.remove(i)
        else:
            C.remove(j)
    return x, basis


def _adjacency(basis, m, n):
    adj_r = [[] for _ in range(m)]
    adj_c = [[] for _ in range(n)]
    for i, j in basis:
        adj_r[i].append(j)
        adj_c[j].append(i)
    return adj_r, adj_c


def _potentials(cost, basis, m, n):
    adj_r, adj_c = _adjacency(basis, m, n)
    u, v = [None] * m, [None] * n
    u[0] = 0.0
    stack = [("r", 0)]
    while stack:
        kind, k = stack.pop()
        if kind == "r":
            for j in adj_r[k]:
                if v[j] is None:
                    v[j] = cost[k, j] - u[k]
                    stack.append(("c", j))
        else:
            for i in adj_c[k]:
                if u[i] is None:
                    u[i] = cost[i, k] - v[k]
                    stack.append(("r", i))
    if any(t is None for t in u) or any(t is None for t in v):
        raise RuntimeError("Basis is not a spanning tree")
    return np.array(u), np.array(v)


def _cycle(basis, i0, j0, m, n):
    """Edges (basic cells) on the tree path from row i0 to column j0."""
    adj_r, adj_c = _adjacency(basis, m, n)
    start, goal = ("r", i0), ("c", j0)
    parent = {start: None}
    q = deque([start])
    while q:
        node = q.popleft()
        if node == goal:
            break
        kind, k = node
        nbrs = ([(("c", j), (k, j)) for j in adj_r[k]] if kind == "r"
                else [(("r", i), (i, k)) for i in adj_c[k]])
        for nb, edge in nbrs:
            if nb not in parent:
                parent[nb] = (node, edge)
                q.append(nb)
    path, node = [], goal
    while parent[node] is not None:
        node, edge = parent[node][0], parent[node][1]
        path.append(edge)
    return path[::-1]                         # path[0::2] lose flow, path[1::2] gain flow


def solve_transportation(cost, supply, demand, method: str = "vogel", max_iter: int = 10_000) -> TransportResult:
    """Minimise total shipping cost.  ``cost`` is (sources x destinations)."""
    cost = np.asarray(cost, dtype=float)
    supply = np.asarray(supply, dtype=float)
    demand = np.asarray(demand, dtype=float)
    m0, n0 = cost.shape
    if supply.shape != (m0,) or demand.shape != (n0,):
        raise ValueError("Shapes of cost, supply and demand are inconsistent")
    if np.any(supply < 0) or np.any(demand < 0):
        raise ValueError("Supply and demand must be non-negative")

    blocked = ~np.isfinite(cost)
    finite_max = cost[~blocked].max() if (~blocked).any() else 0.0
    big_m = (abs(finite_max) + 1.0) * (max(supply.sum(), demand.sum()) + 1.0)
    work = np.where(blocked, big_m, cost)

    # Balance the problem.
    dummy, diff = None, supply.sum() - demand.sum()
    if diff > EPS:                            # surplus supply -> dummy sink
        work = np.hstack([work, np.zeros((m0, 1))])
        demand = np.append(demand, diff)
        dummy = "sink"
    elif diff < -EPS:                         # shortage -> dummy source
        work = np.vstack([work, np.zeros((1, n0))])
        supply = np.append(supply, -diff)
        dummy = "source"
    m, n = work.shape

    x, basis = _initial_bfs(work, supply, demand, method)
    real_cost = lambda X: float(np.sum(X[:m0, :n0][~blocked] * cost[~blocked]))
    initial_cost = real_cost(x)

    iterations, alt = 0, False
    while True:
        u, v = _potentials(work, basis, m, n)
        red = work - u[:, None] - v[None, :]
        for cell in basis:
            red[cell] = 0.0
        nb_mask = np.ones((m, n), dtype=bool)
        for cell in basis:
            nb_mask[cell] = False
        masked = np.where(nb_mask, red, np.inf)
        i0, j0 = np.unravel_index(np.argmin(masked), masked.shape)
        if masked[i0, j0] >= -EPS:
            alt = bool(np.any(nb_mask & (np.abs(red) <= EPS)))
            break
        if iterations >= max_iter:
            return TransportResult("iteration_limit", iterations=iterations)
        path = _cycle(basis, i0, j0, m, n)
        minus, plus = path[0::2], path[1::2]
        theta = min(x[c] for c in minus)
        for c in minus:
            x[c] -= theta
        for c in plus:
            x[c] += theta
        x[i0, j0] += theta
        leaving = min(c for c in minus if x[c] <= EPS)
        basis.remove(leaving)
        basis.add((int(i0), int(j0)))
        x[leaving] = 0.0
        iterations += 1

    x[np.abs(x) < EPS] = 0.0
    if np.any(x[:m0, :n0][blocked] > EPS):
        return TransportResult("infeasible", iterations=iterations)

    dummy_flow = None
    if dummy == "sink":
        dummy_flow = x[:m0, n0].copy()
    elif dummy == "source":
        dummy_flow = x[m0, :n0].copy()
    return TransportResult(
        "optimal", allocation=x[:m0, :n0].copy(), total_cost=real_cost(x), u=u, v=v,
        iterations=iterations, initial_method=method, initial_cost=initial_cost,
        alternative_optima=alt, dummy=dummy, dummy_flow=dummy_flow,
    )
