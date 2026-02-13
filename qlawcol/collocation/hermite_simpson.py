import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from tqdm import tqdm

from qlawcol.collocation.col_types import Dynamics, ProblemSpec


def hs_collocation(
    problem: ProblemSpec,
    **minimize_options,
) -> tuple[np.ndarray, np.ndarray, OptimizeResult]:
    """
    Performs trajectory optimization using Hermite-Simpson collocation.
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

        f_eval = f_vec(x, u)
        f_k = f_eval[:-1]
        f_k_plus_1 = f_eval[1:]

        # get midpoints
        x_c = (x[:-1] + x[1:]) / 2 + (h / 8) * (f_k - f_k_plus_1)
        u_c = (u[:-1] + u[1:]) / 2
        f_c = f_vec(x_c, u_c)

        # collocation condition
        x_nxt = x[:-1] + h / 6 * (f_k + 4 * f_c + f_k_plus_1)
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


def hs_interpolant(x_opt: np.ndarray, u_opt: np.ndarray, T: float, f: Dynamics):
    """
    Creates a cubic Hermite spline interpolant for the state and a linear
    interpolant for the control.
    """
    N = x_opt.shape[0] - 1
    h = T / N
    t_nodes = np.linspace(0, T, N + 1)
    f_vec = jax.vmap(f, in_axes=(0, 0))
    f_opt = f_vec(x_opt, u_opt)

    def interpolant(t: float | np.ndarray):
        """
        Interpolates the state and control at a given time t.
        """
        if isinstance(t, float):
            t = np.array([t])

        # Find the interval for each t
        interval_indices = np.searchsorted(t_nodes, t, side="right") - 1
        interval_indices = np.clip(interval_indices, 0, N - 1)

        # Normalize time in each interval
        tau = (t - t_nodes[interval_indices]) / h

        # Get interval start and end points
        x_k = x_opt[interval_indices]
        x_k_plus_1 = x_opt[interval_indices + 1]
        f_k = f_opt[interval_indices]
        f_k_plus_1 = f_opt[interval_indices + 1]
        u_k = u_opt[interval_indices]
        u_k_plus_1 = u_opt[interval_indices + 1]

        # Hermite basis functions
        H0 = 2 * tau**3 - 3 * tau**2 + 1
        H1 = tau**3 - 2 * tau**2 + tau
        H2 = -2 * tau**3 + 3 * tau**2
        H3 = tau**3 - tau**2

        # Interpolate state
        x_interp = (
            H0[:, None] * x_k
            + H1[:, None] * h * f_k
            + H2[:, None] * x_k_plus_1
            + H3[:, None] * h * f_k_plus_1
        )

        # Interpolate control (linear)
        u_interp = (1 - tau)[:, None] * u_k + tau[:, None] * u_k_plus_1

        if len(t) == 1:
            return x_interp[0], u_interp[0]
        return x_interp, u_interp

    return interpolant
