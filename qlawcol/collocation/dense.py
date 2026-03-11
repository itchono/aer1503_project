import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from tqdm import tqdm

from qlawcol.collocation.col_types import ProblemSpec
from qlawcol.collocation.lgl_utils import lgl_differentiation_matrix, lgl_nodes

from .constraints import hermite_simpson


def dense_collocation(
    problem: ProblemSpec,
    constraint_func=hermite_simpson,
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
    def combined_constraints(z: jnp.ndarray) -> jnp.ndarray:
        x, u = unpack(z)
        collocation_conds = constraint_func(x, u, h, f)

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
    jac_constraints = jax.jit(jax.jacfwd(combined_constraints))

    pbar = tqdm(
        total=minimize_options.get("maxiter", 100), desc="Optimization Progress"
    )

    result = minimize(
        objective,
        z0,
        constraints={
            "type": "eq",
            "fun": combined_constraints,
            "jac": jac_constraints,
        },
        method="SLSQP",
        options=minimize_options,
        jac=jac_func,
        callback=callback,
    )

    pbar.close()

    x_opt, u_opt = unpack(result.x)
    return x_opt, u_opt, result


def lgl_collocation(
    problem: ProblemSpec,
    **minimize_options,
) -> tuple[np.ndarray, np.ndarray, OptimizeResult]:
    """
    Performs trajectory optimization using LGL collocation.
    """
    # unpack and infer problem parameters
    f, cost, constraints, guess, T = problem
    t0 = 0

    x_guess, u_guess = guess
    N = x_guess.shape[0] - 1  # number of collocation points
    nx = x_guess.shape[1]  # state dimension
    nu = u_guess.shape[1]  # control dimension

    # LGL collocation specific items
    tau = lgl_nodes(N)
    D = lgl_differentiation_matrix(N, tau)

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
    def collocation_constraints(z: jnp.ndarray) -> jnp.ndarray:
        x, u = unpack(z)
        dx = D @ x

        f_eval = jax.vmap(f)(x, u)

        defects = dx - (T - t0) / 2 * f_eval
        return defects.flatten()

    @jax.jit
    def combined_constraints(z: jnp.ndarray) -> jnp.ndarray:
        x, u = unpack(z)
        collocation_conds = collocation_constraints(z)
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
    jac_constraints = jax.jit(jax.jacfwd(combined_constraints))

    pbar = tqdm(
        total=minimize_options.get("maxiter", 100), desc="Optimization Progress"
    )

    result = minimize(
        objective,
        z0,
        constraints={
            "type": "eq",
            "fun": combined_constraints,
            "jac": jac_constraints,
        },
        method="SLSQP",
        options=minimize_options,
        jac=jac_func,
        callback=callback,
    )

    pbar.close()

    x_opt, u_opt = unpack(result.x)
    return x_opt, u_opt, result


def hlgl_collocation(
    problem: ProblemSpec,
    m: int,
    N: int,
    **minimize_options,
) -> tuple[np.ndarray, np.ndarray, OptimizeResult]:
    """
    Performs trajectory optimization using Hermite-LGL collocation
    by splitting the time horizon into m segments and applying LGL collocation with N+1 points on each segment.

    There are redundant variables at the segment boundaries, so we
    enforce continuity constraints to ensure the solution is consistent across segments
    in both state and control.
    """
    # unpack and infer problem parameters
    f, cost, constraints, guess, T = problem

    # each interval spans a time T/m
    interval_time = T / m

    x_guess, u_guess = guess
    nx = x_guess.shape[2]  # state dimension
    nu = u_guess.shape[2]  # control dimension

    # LGL collocation specific items
    tau = lgl_nodes(N)
    D = lgl_differentiation_matrix(N, tau)

    # variable marshalling
    def unpack(z: np.ndarray):
        """
        z = [x0_0, x0_1, ..., x0_N, x1_0, ..., xm_N, u0_0, ..., um_N]
        where xk_j is the state at the j-th collocation point of the k-th segment.
        """
        x = z[: m * (N + 1) * nx].reshape((m, N + 1, nx))
        u = z[m * (N + 1) * nx :].reshape((m, N + 1, nu))
        return x, u

    def pack(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.concatenate([x.flatten(), u.flatten()])

    # constraints
    def collocation_constraints(z: jnp.ndarray) -> jnp.ndarray:
        x, u = unpack(z)
        defects = []
        for k in range(m):
            xk = x[k]
            uk = u[k]
            dxk = D @ xk
            fk_eval = jax.vmap(f)(xk, uk)
            defect_k = dxk - (interval_time) / 2 * fk_eval
            defects.append(defect_k.flatten())
        return jnp.concatenate(defects)

    def continuity_constraints(z: jnp.ndarray) -> jnp.ndarray:
        x, u = unpack(z)
        cont_conds = []
        for k in range(m - 1):
            # state continuity: x_k(N) == x_{k+1}(0)
            cont_conds.append(x[k, -1] - x[k + 1, 0])
            # control continuity: u_k(N) == u_{k+1}(0)
            cont_conds.append(u[k, -1] - u[k + 1, 0])
        return jnp.concatenate(cont_conds)

    @jax.jit
    def combined_constraints(z: jnp.ndarray) -> jnp.ndarray:
        x, u = unpack(z)
        collocation_conds = collocation_constraints(z)
        cont_conds = continuity_constraints(z)
        return jnp.concatenate([collocation_conds, cont_conds, constraints(x)])

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
    jac_constraints = jax.jit(jax.jacfwd(combined_constraints))
    pbar = tqdm(
        total=minimize_options.get("maxiter", 100), desc="Optimization Progress"
    )
    result = minimize(
        objective,
        z0,
        constraints={
            "type": "eq",
            "fun": combined_constraints,
            "jac": jac_constraints,
        },
        method="SLSQP",
        options=minimize_options,
        jac=jac_func,
        callback=callback,
    )
    pbar.close()

    x_opt, u_opt = unpack(result.x)
    return x_opt, u_opt, result
