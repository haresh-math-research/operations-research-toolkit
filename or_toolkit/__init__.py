"""Operations Research Toolkit: from-scratch LP, ILP, transportation and assignment solvers."""
from .assignment import AssignmentResult, solve_assignment
from .integer import ILPResult, solve_ilp
from .simplex import LPResult, solve_lp
from .transportation import TransportResult, solve_transportation

__all__ = [
    "solve_lp", "LPResult",
    "solve_ilp", "ILPResult",
    "solve_transportation", "TransportResult",
    "solve_assignment", "AssignmentResult",
]
__version__ = "0.2.0"
