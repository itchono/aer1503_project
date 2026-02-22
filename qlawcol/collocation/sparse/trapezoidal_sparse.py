import jax
import jax.numpy as jnp
import numpy as np
import pyoptsparse
from scipy.optimize import OptimizeResult

from qlawcol.collocation.col_types import ProblemSpec

from .sparse_utils import collocation_jac_sparsity, mask_to_sparse


def trapezoidal_collocation_sparse(
    problem: ProblemSpec,
    **optimizer_options,
) -> tuple[np.ndarray, np.ndarray, OptimizeResult]:
    """
    Performs trajectory optimization using trapezoidal collocation with pyOptSparse and IPOPT.
    """
    # unpack and infer problem parameters
    f, cost, constraints, guess, T = problem

    # infer problem sizes from initial guesses
    x_guess, u_guess = guess
    N = x_guess.shape[0] - 1  # number of intervals
    nx = x_guess.shape[1]  # state dimension
    nu = u_guess.shape[1]  # control dimension
    len_x = (N + 1) * nx
    len_u = (N + 1) * nu
    n_collocation_constr = N * nx
    n_additional_constr = constraints(x_guess).shape[0]

    h: float = T / N  # time step

    # constraints: collocation constraints + additional user-defined constraints
    @jax.jit
    def collocation_constraints(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        f_vec = jax.vmap(f, in_axes=(0, 0))

        f_eval = f_vec(x, u)
        f_k = f_eval[:-1]
        f_k_plus_1 = f_eval[1:]

        # Trapezoidal collocation on interior points
        x_nxt = x[:-1] + h / 2 * (f_k + f_k_plus_1)

        return (x[1:] - x_nxt).flatten() / h  # scaling by h to improve conditioning

    @jax.jit
    def objective_and_cons(xdict: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        x = xdict["x"].reshape((N + 1, nx))
        u = xdict["u"].reshape((N + 1, nu))
        obj = cost(x, u)

        return {
            "obj": obj,
            "collocation_constr": collocation_constraints(x, u),
            "additional_constr": constraints(x),
        }

    # get sparsity pattern of collocation constraint jacobian for efficient optimization
    jac_col_x_sparsity, jac_col_u_sparsity = collocation_jac_sparsity(N, nx, nu)

    # numerically probe jac of additional constraints
    jac_add_x_sparsity = (
        jnp.abs(jax.jacfwd(constraints, argnums=0)(x_guess)) > 1e-8
    ).reshape(-1, len_x)
    jac_add_x_sparsity = {
        "coo": [
            np.where(jac_add_x_sparsity)[0],
            np.where(jac_add_x_sparsity)[1],
            np.ones(np.sum(jac_add_x_sparsity)),
        ],
        "shape": [n_additional_constr, len_x],
    }

    @jax.jit
    def sens(xdict: dict[str, jnp.ndarray], _):
        x = xdict["x"].reshape((N + 1, nx))
        u = xdict["u"].reshape((N + 1, nu))

        # for minimum energy, we assume obj only has grad wrt u
        jac_obj_u = jax.grad(cost, argnums=1)(x, u).flatten()

        # collocation constraints
        # jacobians are always taller than wide so jacfwd is more efficient
        jac_colcon_x = jax.jacfwd(collocation_constraints, argnums=0)(x, u).reshape(
            -1, len_x
        )
        jac_colcon_u = jax.jacfwd(collocation_constraints, argnums=1)(x, u).reshape(
            -1, len_u
        )
        jac_add_x = jax.jacfwd(constraints, argnums=0)(x).reshape(-1, len_x)

        # apply sparsity pattern to all jacobians
        jac_colcon_x = mask_to_sparse(jac_col_x_sparsity, jac_colcon_x)
        jac_colcon_u = mask_to_sparse(jac_col_u_sparsity, jac_colcon_u)
        jac_addcon_x = mask_to_sparse(jac_add_x_sparsity, jac_add_x)

        return {
            "obj": {"u": jac_obj_u},
            "collocation_constr": {"x": jac_colcon_x, "u": jac_colcon_u},
            "additional_constr": {"x": jac_addcon_x},
        }

    opt_prob = pyoptsparse.Optimization("orbit_transfer", objective_and_cons)
    opt_prob.addVarGroup("x", len_x, value=x_guess.flatten())
    opt_prob.addVarGroup("u", len_u, value=u_guess.flatten())

    opt_prob.addConGroup(
        "collocation_constr",
        n_collocation_constr,
        lower=0,
        upper=0,
        jac={"x": jac_col_x_sparsity, "u": jac_col_u_sparsity},
    )
    opt_prob.addConGroup(
        "additional_constr",
        n_additional_constr,
        lower=0,
        upper=0,
        wrt=["x"],
        jac={"x": jac_add_x_sparsity},
    )

    opt_prob.addObj("obj")

    opt = pyoptsparse.IPOPT(options=optimizer_options)
    result = opt(opt_prob, sens=sens)

    x_opt = result.xStar["x"].reshape((N + 1, nx))
    u_opt = result.xStar["u"].reshape((N + 1, nu))
    return x_opt, u_opt, result
