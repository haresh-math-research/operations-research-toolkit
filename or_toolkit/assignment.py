"""Assignment problem via the Hungarian (Kuhn-Munkres) algorithm, O(n^2 m).

Uses the shortest-augmenting-path formulation with dual potentials.  Handles
rectangular matrices (every row of the smaller side is matched), maximisation,
and forbidden pairs (``np.inf`` cost).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class AssignmentResult:
    status: str
    pairs: Optional[List[Tuple[int, int]]] = None   # (row, column), sorted by row
    total: Optional[float] = None
    unassigned_rows: Optional[List[int]] = None
    unassigned_cols: Optional[List[int]] = None

    @property
    def success(self) -> bool:
        return self.status == "optimal"

    def summary(self) -> str:
        if not self.success:
            return f"Status: {self.status}"
        return f"Status: optimal  total={self.total:.6g}\nPairs (row -> col): {self.pairs}"


def _hungarian_min(cost: np.ndarray) -> List[int]:
    """Rows <= cols.  Returns col assigned to each row (min total cost)."""
    n, m = cost.shape
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)      # p[j] = row matched to column j (1-based, 0 = none)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    match = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            match[p[j] - 1] = j - 1
    return match


def solve_assignment(cost, maximize: bool = False) -> AssignmentResult:
    """Optimal one-to-one assignment for a (possibly rectangular) cost/profit matrix."""
    C = np.asarray(cost, dtype=float)
    if C.ndim != 2 or C.size == 0:
        raise ValueError("cost must be a non-empty 2-D matrix")
    blocked = ~np.isfinite(C)
    finite = np.abs(C[~blocked]).max() if (~blocked).any() else 0.0
    big = (finite + 1.0) * (max(C.shape) + 1.0)
    W = np.where(blocked, big, C)
    if maximize:
        W = np.where(blocked, big, -C)

    transposed = W.shape[0] > W.shape[1]
    Wt = W.T if transposed else W
    match = _hungarian_min(Wt)
    pairs = [(j, i) if transposed else (i, j) for i, j in enumerate(match)]
    pairs.sort()
    if any(blocked[i, j] for i, j in pairs):
        return AssignmentResult("infeasible")
    total = float(sum(C[i, j] for i, j in pairs))
    rows = sorted(set(range(C.shape[0])) - {i for i, _ in pairs})
    cols = sorted(set(range(C.shape[1])) - {j for _, j in pairs})
    return AssignmentResult("optimal", pairs, total, rows, cols)
