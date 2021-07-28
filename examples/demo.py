"""Run `python examples/demo.py` (or `pip install -e .` first and run it from anywhere)."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from or_toolkit import solve_assignment, solve_ilp, solve_lp, solve_transportation


def banner(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


banner("1. Linear programme (production planning) with sensitivity analysis")
# max 3x + 2y  s.t. 2x + y <= 100 (machine hours), x + y <= 80 (labour), x <= 40
r = solve_lp([3, 2], [[2, 1], [1, 1], [1, 0]], [100, 80, 40], maximize=True)
print(r.summary())
print("-> One extra machine hour is worth", round(r.shadow_prices[0], 4), "; the x<=40 cap is slack by", r.slack[2])

banner("2. Integer programme (branch & bound)")
r = solve_ilp([5, 4], [[6, 4], [1, 2]], [24, 6], maximize=True)
print(r.summary())

banner("3. Transportation problem (Vogel start + MODI)")
cost = [[19, 30, 50, 10], [70, 30, 40, 60], [40, 8, 70, 20]]
for method in ("nwc", "least_cost", "vogel"):
    r = solve_transportation(cost, [7, 9, 18], [5, 8, 7, 14], method=method)
    print(f"{method:>10}: start cost {r.initial_cost:>5.0f} -> optimal {r.total_cost:.0f} after {r.iterations} MODI pivots")
print(r.allocation)

banner("4. Assignment problem (Hungarian), maximising profit on a 3x4 matrix")
profit = np.array([[7, 5, 9, 3], [8, 6, 4, 2], [3, 9, 8, 7]])
print(solve_assignment(profit, maximize=True).summary())
