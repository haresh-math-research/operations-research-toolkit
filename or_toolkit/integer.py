"""Branch-and-bound for (mixed-)integer linear programs.

Each node's LP relaxation is solved with this package's own simplex
(:func:`or_toolkit.simplex.solve_lp`).  Nodes are explored best-bound-first
and pruned against the incumbent.  Branching is on the most fractional variable.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .simplex import INFEASIBLE, OPTIMAL, solve_lp


@dataclass
class ILPResult:
    status: str
    x: Optional[np.ndarray] = None
    objective: Optional[float] = None
    root_bound: Optional[float] = None    # LP-relaxation objective at the root node
    nodes: int = 0
    gap: Optional[float] = None           # relative gap when stopped early (0 if proven optimal)

    @property
    def success(self) -> bool:
        return self.status == OPTIMAL

    def summary(self) -> str:
        if not self.success:
            return f"Status: {self.status} ({self.nodes} nodes)"
        return (f"Status: {self.status}  nodes={self.nodes}  root LP bound={self.root_bound:.6g}\n"
                f"Objective: {self.objective:.6g}\nx = {np.array2string(self.x, precision=6, suppress_small=True)}")


def solve_ilp(
    c: Sequence[float],
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    senses: Optional[Sequence[str]] = None,
    maximize: bool = False,
    integer_vars: Optional[Sequence[int]] = None,
    int_tol: float = 1e-6,
    max_nodes: int = 100_000,
) -> ILPResult:
    """Solve an ILP/MILP.  ``integer_vars`` defaults to every variable."""
    c = np.asarray(c, dtype=float)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    senses = ["<="] * m if senses is None else list(senses)
    int_idx = list(range(n)) if integer_vars is None else list(integer_vars)
    sgn = 1.0 if maximize else -1.0            # larger (sgn * obj) is better

    def relax(bounds):
        if bounds:
            rows = np.zeros((len(bounds), n))
            rhs, sen = [], []
            for k, (j, s, v) in enumerate(bounds):
                rows[k, j] = 1.0
                rhs.append(v)
                sen.append(s)
            return solve_lp(c, np.vstack([A, rows]), np.concatenate([b, rhs]),
                            senses + sen, maximize=maximize, sensitivity=False)
        return solve_lp(c, A, b, senses, maximize=maximize, sensitivity=False)

    root = relax([])
    if root.status != OPTIMAL:
        return ILPResult(root.status, nodes=1)

    best_x, best_val = None, -math.inf                       # best_val is in "sgn * obj" units
    counter = itertools.count()
    heap = [(-sgn * root.objective, next(counter), [], root)]
    nodes = 0
    while heap and nodes < max_nodes:
        _, _, bounds, lp = heapq.heappop(heap)
        nodes += 1
        if sgn * lp.objective <= best_val + 1e-9:            # cannot beat the incumbent
            continue
        frac = [(abs(lp.x[j] - round(lp.x[j])), j) for j in int_idx if abs(lp.x[j] - round(lp.x[j])) > int_tol]
        if not frac:
            best_x, best_val = lp.x.copy(), sgn * lp.objective
            for j in int_idx:
                best_x[j] = round(best_x[j])
            continue
        _, j = max(frac)
        v = lp.x[j]
        for new in ((j, "<=", math.floor(v)), (j, ">=", math.ceil(v))):
            child = relax(bounds + [new])
            if child.status == OPTIMAL and sgn * child.objective > best_val + 1e-9:
                heapq.heappush(heap, (-sgn * child.objective, next(counter), bounds + [new], child))

    if best_x is None:
        status = INFEASIBLE if not heap else "node_limit"
        return ILPResult(status, nodes=nodes, root_bound=root.objective)
    obj = float(c @ best_x)
    if heap:
        open_bound = max(sgn * h[3].objective for h in heap)
        gap = max(0.0, (open_bound - best_val) / max(1.0, abs(best_val)))
    else:
        gap = 0.0
    status = OPTIMAL if gap <= 1e-9 else "node_limit"
    return ILPResult(status, x=best_x, objective=obj, root_bound=root.objective, nodes=nodes, gap=gap)
