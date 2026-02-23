import jax
import jax.numpy as jnp
import numpy as np
import pyoptsparse
import sparsejac

from qlawcol.collocation.col_types import ProblemSpec

from .pbar_utils import ipopt_pbar_from_file
from .sparse_utils import (
    collocation_jac_sparsity,
    detect_sparsity_pattern,
    jax_bcoo_to_pyoptsparse,
    pyoptsparse_to_jax_bcoo,
)


def hs_collocation_sparse(
    problem: ProblemSpec,
    **optimizer_options,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Performs trajectory optimization using Hermite-Simpson collocation with pyOptSparse and IPOPT.
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

        # get midpoints
        x_c = (x[:-1] + x[1:]) / 2 + (h / 8) * (f_k - f_k_plus_1)
        u_c = (u[:-1] + u[1:]) / 2
        f_c = f_vec(x_c, u_c)

        # collocation condition
        defect = (x[1:] - x[:-1]) / h - (1 / 6) * (f_k + 4 * f_c + f_k_plus_1)
        return defect.flatten()

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

    # convert to jax BCOO format to specify jacfwd sparsity in sparsejac
    def col_flat(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return collocation_constraints(x.reshape((N + 1, nx)), u.reshape((N + 1, nu)))

    def addcon_flat(x: np.ndarray) -> np.ndarray:
        return constraints(x.reshape((N + 1, nx)))

    # numerically probe jac of additional constraints
    jac_add_x_sparsity = detect_sparsity_pattern(
        jax.jacfwd(addcon_flat, argnums=0)(x_guess.flatten())
    )

    @jax.jit
    def sens(xdict: dict[str, jnp.ndarray], _):
        x = xdict["x"].reshape((N + 1, nx))
        u = xdict["u"].reshape((N + 1, nu))

        # for minimum energy, we assume obj only has grad wrt x (final mass)
        jac_obj_x = jax.grad(cost, argnums=0)(x, u).flatten()

        with jax.ensure_compile_time_eval():
            jccx_fn = sparsejac.jacfwd(
                col_flat,
                argnums=0,
                sparsity=pyoptsparse_to_jax_bcoo(jac_col_x_sparsity),
            )
            jccu_fn = sparsejac.jacfwd(
                col_flat,
                argnums=1,
                sparsity=pyoptsparse_to_jax_bcoo(jac_col_u_sparsity),
            )
            jadd_fn = sparsejac.jacfwd(
                addcon_flat,
                argnums=0,
                sparsity=pyoptsparse_to_jax_bcoo(jac_add_x_sparsity),
            )

        # collocation constraints
        jac_colcon_x = jccx_fn(x.flatten(), u.flatten())
        jac_colcon_u = jccu_fn(x.flatten(), u.flatten())
        jac_add_x = jadd_fn(x.flatten())

        # apply sparsity pattern to all jacobians
        jac_colcon_x = jax_bcoo_to_pyoptsparse(jac_colcon_x)
        jac_colcon_u = jax_bcoo_to_pyoptsparse(jac_colcon_u)
        jac_addcon_x = jax_bcoo_to_pyoptsparse(jac_add_x)

        return {
            "obj": {"x": jac_obj_x},
            "collocation_constr": {"x": jac_colcon_x, "u": jac_colcon_u},
            "additional_constr": {"x": jac_addcon_x},
        }

    opt_prob = pyoptsparse.Optimization("collocation", objective_and_cons)

    # bound state
    a_lb = np.zeros(N + 1)  # SMA must be positive
    a_ub = (
        np.ones(N + 1) * 20
    )  # some large number to effectively have no upper bound on SMA
    f_lb = np.ones(N + 1) * -0.99  # f can be in [-1, 1]
    f_ub = np.ones(N + 1) * 0.99
    g_lb = np.ones(N + 1) * -0.99  # g can be in [-1, 1]
    g_ub = np.ones(N + 1) * 0.99
    h_lb = np.ones(N + 1) * -10
    h_ub = (
        np.ones(N + 1) * 10
    )  # h can be large, but we set some bounds to help optimization
    k_lb = np.ones(N + 1) * -10
    k_ub = (
        np.ones(N + 1) * 10
    )  # k can be large, but we set some bounds to help optimization
    L_lb = np.ones(N + 1) * -np.inf
    L_ub = np.ones(N + 1) * np.inf
    mass_lb = np.zeros(N + 1)  # mass must be positive
    mass_ub = np.ones(N + 1) * 1  # mass cannot exceed initial mass

    x_lb = np.column_stack((a_lb, f_lb, g_lb, h_lb, k_lb, L_lb, mass_lb)).flatten()
    x_ub = np.column_stack((a_ub, f_ub, g_ub, h_ub, k_ub, L_ub, mass_ub)).flatten()

    opt_prob.addVarGroup("x", len_x, value=x_guess.flatten(), lower=x_lb, upper=x_ub)

    # bound control
    u_lb0 = np.zeros(N + 1)
    u_ub0 = np.ones(N + 1) * 1.0  # assume max throttle is 1.0
    u_lb12 = np.ones(N + 1) * -np.pi
    u_ub12 = np.ones(N + 1) * np.pi
    u_lb = np.column_stack((u_lb0, u_lb12, u_lb12)).flatten()
    u_ub = np.column_stack((u_ub0, u_ub12, u_ub12)).flatten()

    opt_prob.addVarGroup("u", len_u, value=u_guess.flatten(), lower=u_lb, upper=u_ub)

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
    with ipopt_pbar_from_file(max_iter=optimizer_options.get("max_iter", 1000)):
        result = opt(opt_prob, sens=sens)

    x_opt = result.xStar["x"].reshape((N + 1, nx))
    u_opt = result.xStar["u"].reshape((N + 1, nu))
    return x_opt, u_opt, result
