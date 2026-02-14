import jax
import jax.numpy as jnp
import numpy as np
import pyoptsparse
from scipy.optimize import OptimizeResult

from qlawcol.collocation.col_types import Dynamics, ProblemSpec


def trapezoidal_collocation(
    problem: ProblemSpec,
    **optimizer_options,
) -> tuple[np.ndarray, np.ndarray, OptimizeResult]:
    """
    Performs trajectory optimization using trapezoidal collocation with pyOptSparse and IPOPT.
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
    def collocation_constraints_jit(z: jnp.ndarray) -> jnp.ndarray:
        x, u = unpack(z)
        f_vec = jax.vmap(f, in_axes=(0, 0))

        f_eval = f_vec(x, u)
        f_k = f_eval[:-1]
        f_k_plus_1 = f_eval[1:]

        # Trapezoidal collocation on interior points
        x_nxt = x[:-1] + h / 2 * (f_k + f_k_plus_1)
        collocation_conds = (x[1:] - x_nxt).flatten()

        return jnp.concatenate([collocation_conds, constraints(x)])

    # call to IPOPT through pyoptsparse
    z0 = pack(x_guess, u_guess)
    n_vars = len(z0)
    n_cons = len(collocation_constraints_jit(z0))

    @jax.jit
    def objective_jit(z: np.ndarray) -> float:
        x, u = unpack(z)
        return cost(x, u)

    def objective_and_cons(z_dict):
        z = z_dict["z"]
        obj = objective_jit(z)
        cons = collocation_constraints_jit(z)
        return {"obj": obj, "con": cons}

    # construct jac and hess functions for IPOPT using jax autograd
    jac_cons_func = jax.jit(jax.jacfwd(collocation_constraints_jit))
    jac_obj_func = jax.jit(jax.grad(objective_jit))

    # get sparsity pattern
    jac_cons_sparsity = jnp.abs(jac_cons_func(z0)) > 1e-12

    def sens(z_dict, _):
        z = z_dict["z"]
        # jac_cons_values = jac_cons_func(z)[jac_cons_sparsity]

        return {
            # "con": {"z": jac_cons_values},
            "con": {"z": jac_cons_func(z)},
            "obj": {"z": jac_obj_func(z)},
        }

    opt_prob = pyoptsparse.Optimization("orbit_transfer", objective_and_cons)
    opt_prob.addVarGroup("z", n_vars, "c", value=z0)
    opt_prob.addConGroup(
        "con", n_cons, lower=0, upper=0, jac={"z": jac_cons_sparsity.astype(float)}
    )
    opt_prob.addObj("obj")

    opt = pyoptsparse.SLSQP(options=optimizer_options)
    result = opt(opt_prob, sens=sens)

    x_opt, u_opt = unpack(result.xStar["z"])
    return x_opt, u_opt, result


def trapezoidal_interpolant(
    x_opt: np.ndarray, u_opt: np.ndarray, T: float, f: Dynamics
):
    """
    Creates a quadratic interpolant for the state and a linear interpolant for the control
    based on the trapezoidal collocation solution.
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
        t_k = t_nodes[interval_indices]
        tau = ((t - t_k) / h)[:, None]

        # Quadratic interpolation for state
        x_k = x_opt[interval_indices]
        x_k_plus_1 = x_opt[interval_indices + 1]
        f_k = f_opt[interval_indices]
        f_k_plus_1 = f_opt[interval_indices + 1]
        x_interp = (
            (1 - tau) * x_k
            + tau * x_k_plus_1
            + tau * (1 - tau) * h / 8 * (f_k_plus_1 - f_k)
        )

        # Linear interpolation for control
        u_k = u_opt[interval_indices]
        u_k_plus_1 = u_opt[interval_indices + 1]
        u_interp = (1 - tau) * u_k + tau * u_k_plus_1

        if x_interp.shape[0] == 1:
            return x_interp[0], u_interp[0]
        return x_interp, u_interp

    return interpolant
