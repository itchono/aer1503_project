from typing import Callable, NamedTuple

import numpy as np
from scipy.optimize import OptimizeResult, minimize

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
    f: Dynamics
    cost: Cost
    constraints: Constraints
    # scalar list of additional constraints (e.g. boundary conditions)
    guess: Guess
    T: float


def trapezoidal_collocation(
    problem: ProblemSpec,
    **optimizer_kwargs,
) -> tuple[np.ndarray, np.ndarray, OptimizeResult]:
    """
    Performs trajectory optimization using trapezoidal collocation.
    """
    # unpack and infer problem parameters
    f, cost, constraints, guess, T = problem

    x_guess, u_guess = guess
    N = x_guess.shape[0] - 1  # number of intervals
    nx = x_guess.shape[1]  # state dimension
    nu = u_guess.shape[1]  # control dimension

    h = T / N  # time step

    # variable marshalling
    def unpack(z: np.ndarray):
        """
        z = [x0, x1, ..., xN, u0, u1, ..., uN]
        """
        x = z[: (N + 1) * nx].reshape((N + 1, nx))
        u = z[(N + 1) * nx :].reshape((N + 1, nu))
        return x, u

    def pack(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.concatenate([x.flatten(), u.flatten()])

    # constraints
    def collocation_constraints(z: np.ndarray) -> np.ndarray:
        x, u = unpack(z)
        col_cons = []

        # Trapezoidal collocation on interior points
        for k in range(N):
            x_next_pred = x[k] + 0.5 * h * (f(x[k], u[k]) + f(x[k + 1], u[k + 1]))
            col_cons.append(x[k + 1] - x_next_pred)

        col_cons_flat = np.concatenate(col_cons).flatten()

        # Additional constraints
        additional_cons = constraints(*unpack(z))

        return np.concatenate([col_cons_flat, additional_cons])

    # call to SLSQP
    z0 = pack(x_guess, u_guess)

    result = minimize(
        lambda z: cost(*unpack(z)),
        z0,
        constraints={"type": "eq", "fun": collocation_constraints},
        method="SLSQP",
        **optimizer_kwargs,
    )

    x_opt, u_opt = unpack(result.x)
    return x_opt, u_opt, result
