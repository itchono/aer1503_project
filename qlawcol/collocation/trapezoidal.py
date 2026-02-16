import jax
import jax.numpy as jnp
import numpy as np
import pyoptsparse
from scipy.optimize import OptimizeResult

from qlawcol.collocation.col_types import Dynamics, ProblemSpec

PyOptSparseCOO = dict[str, list[float] | list[int]]
# mat = {'coo':[row, col, data], 'shape':[nrow, ncols]}


def trap_col_jac_sparsity(
    N: int, nx: int, nu: int
) -> tuple[PyOptSparseCOO, PyOptSparseCOO]:
    """
    construct sparsity pattern for collocation constraints
    collocation constraints depend on x_k, x_k+1, u_k, u_k+1
    x and u are each flattened i.e. x = [x1_0, x2_0, ..., x1_1, x2_1, ...]

    the constraint jacobian wrt both will be sparse since
    the i-th constraint depends only on xl_i, xl_i+1, ul_i, ul_i+1

    General form looks like (1 for nonzero, 0 for zero):
    [1 1 1 1 0 0 0 0 ...]
    [1 1 1 1 0 0 0 0 ...]
    [0 0 1 1 1 1 0 0 ...]
    [0 0 1 1 1 1 0 0 ...]

    - the width of each block is 2 nx for x and 2 nu for u
    - the height of each block is nx (corresponding to constraint dimension)
    """

    row_idx_x = np.repeat(np.arange(N * nx), 2 * nx)
    row_idx_u = np.repeat(np.arange(N * nx), 2 * nu)
    col_idx_x = []
    col_idx_u = []

    jac_x_shape = [N * nx, (N + 1) * nx]
    jac_u_shape = [N * nx, (N + 1) * nu]

    for i in range(N):
        # matrix is N * nx rows tall, so each iteration should
        # "add nx rows". Each row within each "block" is identical.

        # each "block": the nonzero indices move forward by nx and nu for each constraint
        # we have 2 nx and 2 nu nonzeros per row, and this is repeated for each of the nx rows in the block
        col_idx_x.extend(
            ([i * nx + j for j in range(nx)] + [(i + 1) * nx + j for j in range(nx)])
            * nx
        )
        col_idx_u.extend(
            ([i * nu + j for j in range(nu)] + [(i + 1) * nu + j for j in range(nu)])
            * nx
        )

    col_idx_x = np.array(col_idx_x)
    col_idx_u = np.array(col_idx_u)

    one_x = np.ones_like(row_idx_x, dtype=float)
    one_u = np.ones_like(row_idx_u, dtype=float)

    # return in PyOptSparse COO format
    jac_x = {
        "coo": [row_idx_x, col_idx_x, one_x],
        "shape": jac_x_shape,
    }

    jac_u = {
        "coo": [row_idx_u, col_idx_u, one_u],
        "shape": jac_u_shape,
    }

    return jac_x, jac_u


def mask_to_sparse(coo_mask: PyOptSparseCOO, data: np.ndarray) -> PyOptSparseCOO:
    """
    Takes a dense matrix and preserves only the entries corresponding to the nonzero pattern in coo_mask.
    """
    row_idx, col_idx, _ = coo_mask["coo"]
    sparse_data = data[row_idx, col_idx]

    return {
        "coo": [row_idx, col_idx, sparse_data],
        "shape": coo_mask["shape"],
    }


def trapezoidal_collocation(
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

        return (x[1:] - x_nxt).flatten()

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
    jac_col_x_sparsity, jac_col_u_sparsity = trap_col_jac_sparsity(N, nx, nu)

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
