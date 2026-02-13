from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from tqdm import tqdm

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


def trapezoidal_collocation(
    problem: ProblemSpec,
    **minimize_options,
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

    h: float = T / N  # time step

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
    @jax.jit
    def collocation_constraints(z: jnp.ndarray) -> jnp.ndarray:
        x, u = unpack(z)
        f_vec = jax.vmap(f, in_axes=(0, 0))

        # Trapezoidal collocation on interior points
        x_nxt = x[:-1] + h / 2 * (f_vec(x[:-1], u[:-1]) + f_vec(x[1:], u[1:]))
        collocation_conds = (x[1:] - x_nxt).flatten()

        return jnp.concatenate([collocation_conds, constraints(x)])

    # call to SLSQP
    z0 = pack(x_guess, u_guess)

    @jax.jit
    def objective(z: np.ndarray) -> float:
        x, u = unpack(z)
        return cost(x, u)

    def callback(intermediate_result: OptimizeResult):
        dv = intermediate_result.fun
        pbar.set_postfix_str(f"Cost: {dv:.6e}")
        pbar.update(1)

    # construct jax and hess functions for SLSQP using jax autograd
    jac_func = jax.jit(jax.grad(objective))

    pbar = tqdm(
        total=minimize_options.get("maxiter", 100), desc="Optimization Progress"
    )

    result = minimize(
        objective,
        z0,
        constraints={"type": "eq", "fun": collocation_constraints},
        method="SLSQP",
        options=minimize_options,
        jac=jac_func,
        callback=callback,
    )

    pbar.close()

    x_opt, u_opt = unpack(result.x)
    return x_opt, u_opt, result
