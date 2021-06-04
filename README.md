# Operations Research Toolkit

From-scratch implementations of the core deterministic OR algorithms, with post-optimal
sensitivity analysis, validated against SciPy/HiGHS on randomised instances.

| Module | Algorithm | Highlights |
|---|---|---|
| `or_toolkit.simplex` | Two-phase tableau simplex | `<=`, `>=`, `==` rows; infeasible/unbounded detection; Dantzig rule with Bland fallback (anti-cycling); shadow prices, reduced costs, RHS ranging, cost ranging |
| `or_toolkit.integer` | Branch & bound (best-bound-first) | ILP and MILP, most-fractional branching, incumbent pruning, root bound and optimality gap |
| `or_toolkit.transportation` | NWC / least-cost / Vogel + MODI (u-v) | Degeneracy-safe spanning-tree basis, dummy source/sink for unbalanced problems, forbidden routes, alternative-optima detection |
| `or_toolkit.assignment` | Hungarian (Kuhn-Munkres), shortest-augmenting-path form | Rectangular matrices, maximisation, forbidden pairs |

## Quick start

```bash
pip install -r requirements.txt
python examples/demo.py
pytest                       # 398 tests
```

```python
from or_toolkit import solve_lp, solve_ilp, solve_transportation, solve_assignment

r = solve_lp([3, 2], [[2, 1], [1, 1], [1, 0]], [100, 80, 40], maximize=True)
print(r.summary())
# Objective: 180   x = [20. 60.]
# Shadow prices: [1. 1. 0.]      RHS ranges: [80, 120], [60, 100], [20, inf]
# Cost ranges: [2, 4], [1.5, 3]

solve_ilp([5, 4], [[6, 4], [1, 2]], [24, 6], maximize=True)              # LP bound 21 -> integer optimum 20
solve_transportation(cost, supply, demand, method="vogel")               # or "nwc", "least_cost"
solve_assignment(profit_matrix, maximize=True)
```

## Method notes

**Simplex.** Constraints are normalised to `b >= 0`, slack/surplus/artificial columns are added,
and Phase I minimises the sum of artificials. Artificials stuck in the basis at level zero
(redundant rows) are pivoted out or left inert. Duals are read from the final tableau as
`y = c_B B^-1` using the columns that started as the identity, then mapped back through any row
sign-flips and the max/min conversion. Ranging uses the ratio tests `B^-1(b + Δe_i) >= 0` and
`r_l - Δ·α_kl >= 0`. Ranges are valid while the current basis stays optimal (in degenerate
vertices they can be conservative).

**Branch & bound.** Each node adds a bound row (`x_j <= ⌊v⌋` or `x_j >= ⌈v⌉`) and re-solves the
relaxation with the same simplex. Nodes come off a heap ordered by LP bound, so the first
integer-feasible node that beats every open bound is provably optimal; on `max_nodes` the result
reports `node_limit` and the residual gap.

**Transportation.** Initial BFS keeps exactly `m + n - 1` basic cells even when supply and demand
exhaust simultaneously (only one line is struck out), so the basis is always a spanning tree.
MODI computes potentials by tree traversal, finds the entering cell with the most negative
`c_ij - u_i - v_j`, locates the unique stepping-stone cycle by BFS on the tree, and pivots.

**Hungarian.** O(n²m) with dual potentials `u, v`; rectangular problems are transposed so the
smaller side is fully matched.

## Testing

* 60 random LPs (mixed `<=`, `>=`, `==`) vs `scipy.optimize.linprog`; Beale's cycling example;
  redundant equalities; infeasible/unbounded detection.
* Sensitivity: 40 random LPs where every reported RHS/cost range endpoint is checked by
  perturbing the data and re-solving.
* 25 random ILPs vs `scipy.optimize.milp`.
* 180 random transportation instances (all three starts, balanced/unbalanced/degenerate) vs
  `linprog`; 80 random assignment problems vs `linear_sum_assignment`.

## Limitations

Dense tableau implementations for clarity: fine for hundreds of variables, not thousands. All LP
variables are `>= 0` (shift/split variables for other bounds). No revised simplex, dual simplex,
or cutting planes yet. See the roadmap below.

## Roadmap ideas

Revised simplex with LU updates · dual simplex for warm-started B&B · Gomory cuts · min-cost
network flow / shortest path · sensitivity report export (pandas/LaTeX).
