from typing import Callable, NamedTuple

import numpy as np

Dynamics = Callable[[np.ndarray, np.ndarray], np.ndarray]
Cost = Callable[[np.ndarray, np.ndarray], float]
Constraints = Callable[[np.ndarray, np.ndarray], np.ndarray]


class Guess(NamedTuple):
    """
    Initial guess for the state and control trajectories.
    The shape of these arrays implies the number of collocation points
    (N + 1) and the dimensions of state (nx) and control (nu).
    """

    x: np.ndarray
    u: np.ndarray


class ProblemSpec(NamedTuple):
    """Problem specification for trajectory optimization."""

    f: Dynamics
    cost: Cost
    constraints: Constraints
    # scalar list of additional constraints (e.g. boundary conditions)
    guess: Guess
    T: float
