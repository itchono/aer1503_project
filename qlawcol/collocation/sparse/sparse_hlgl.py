import jax
import jax.numpy as jnp
import numpy as np
import pyoptsparse

from qlawcol.collocation.col_types import ProblemSpec
from qlawcol.collocation.lgl_utils import lgl_differentiation_matrix, lgl_nodes

from .pbar_utils import ipopt_pbar_from_file


def sparse_hlgl_collocation(
    problem: ProblemSpec,
    m: int,
    N: int,
    **optimizer_options,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Performs trajectory optimization using sparse Hermite-LGL collocation with pyOptSparse and IPOPT.
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

    # variable and constraint dimensions
    len_x_seg = (N + 1) * nx
    len_u_seg = (N + 1) * nu
    n_collocation_constr_seg = (N + 1) * nx
    n_continuity_constr_seg = nx + nu
    n_additional_constr = constraints(x_guess).shape[0]

    # --- Constraint and Objective Functions ---
    def collocation_constraints_seg(x_seg, u_seg):
        dxk = D @ x_seg
        fk_eval = jax.vmap(f)(x_seg, u_seg)
        defect_k = dxk - (interval_time / 2) * fk_eval
        return defect_k.flatten()

    def continuity_constraints_state(x_curr, x_next):
        return x_curr[-1] - x_next[0]

    def continuity_constraints_control(u_curr, u_next):
        return u_curr[-1] - u_next[0]

    @jax.jit
    def objective_and_cons(xdict: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        # unpack variables
        x = jnp.array([xdict[f"x_{k}"] for k in range(m)]).reshape((m, N + 1, nx))
        u = jnp.array([xdict[f"u_{k}"] for k in range(m)]).reshape((m, N + 1, nu))

        # objective
        obj = cost(x, u)

        # collocation constraints
        collocation_con = {}
        for k in range(m):
            x_seg = x[k]
            u_seg = u[k]
            collocation_con[f"collocation_{k}"] = collocation_constraints_seg(
                x_seg, u_seg
            )

        # continuity constraints
        continuity_con = {}
        for k in range(m - 1):
            continuity_con[f"continuity_state_{k}"] = continuity_constraints_state(
                x[k], x[k + 1]
            )
            continuity_con[f"continuity_control_{k}"] = continuity_constraints_control(
                u[k], u[k + 1]
            )

        # additional constraints
        additional_con = constraints(x)

        return {
            "obj": obj,
            **collocation_con,
            **continuity_con,
            "additional_constr": additional_con,
        }

    # --- Jacobian Computations (sens) ---
    @jax.jit
    def sens(xdict: dict[str, jnp.ndarray], _):
        # unpack variables
        x = jnp.array([xdict[f"x_{k}"] for k in range(m)]).reshape((m, N + 1, nx))
        u = jnp.array([xdict[f"u_{k}"] for k in range(m)]).reshape((m, N + 1, nu))

        # objective jacobian
        jac_obj_x = jax.grad(cost, argnums=0)(x, u).reshape(m, len_x_seg)
        jac_obj_u = jax.grad(cost, argnums=1)(x, u).reshape(m, len_u_seg)
        obj_jac = {}
        for k in range(m):
            obj_jac[f"x_{k}"] = jac_obj_x[k]
            obj_jac[f"u_{k}"] = jac_obj_u[k]

        # collocation jacobians
        collocation_jac = {}
        for k in range(m):
            x_seg = x[k]
            u_seg = u[k]
            jac_x = jax.jacfwd(collocation_constraints_seg, argnums=0)(x_seg, u_seg)
            jac_u = jax.jacfwd(collocation_constraints_seg, argnums=1)(x_seg, u_seg)
            collocation_jac[f"collocation_{k}"] = {
                f"x_{k}": jac_x.reshape(n_collocation_constr_seg, len_x_seg),
                f"u_{k}": jac_u.reshape(n_collocation_constr_seg, len_u_seg),
            }

        # continuity jacobians
        continuity_jac = {}
        for k in range(m - 1):
            # state
            jac_state_curr = jax.jacfwd(continuity_constraints_state, argnums=0)(
                x[k], x[k + 1]
            )
            jac_state_next = jax.jacfwd(continuity_constraints_state, argnums=1)(
                x[k], x[k + 1]
            )
            continuity_jac[f"continuity_state_{k}"] = {
                f"x_{k}": jac_state_curr.reshape(nx, len_x_seg),
                f"x_{k + 1}": jac_state_next.reshape(nx, len_x_seg),
            }
            # control
            jac_control_curr = jax.jacfwd(continuity_constraints_control, argnums=0)(
                u[k], u[k + 1]
            )
            jac_control_next = jax.jacfwd(continuity_constraints_control, argnums=1)(
                u[k], u[k + 1]
            )
            continuity_jac[f"continuity_control_{k}"] = {
                f"u_{k}": jac_control_curr.reshape(nu, len_u_seg),
                f"u_{k + 1}": jac_control_next.reshape(nu, len_u_seg),
            }

        # additional constraints jacobian
        jac_add_x = jax.jacfwd(constraints)(x).reshape(
            n_additional_constr, m * len_x_seg
        )
        add_con_jac = {
            "x_0": jac_add_x[:, :len_x_seg],
            f"x_{m - 1}": jac_add_x[:, -len_x_seg:],
        }

        return {
            "obj": obj_jac,
            **collocation_jac,
            **continuity_jac,
            "additional_constr": add_con_jac,
        }

    # --- pyOptSparse Problem Setup ---
    opt_prob = pyoptsparse.Optimization("sparse-hlgl-collocation", objective_and_cons)

    # Add variables (state and control for each segment)
    for k in range(m):
        opt_prob.addVarGroup(
            f"x_{k}", len_x_seg, value=x_guess[k].flatten(), lower=-np.inf, upper=np.inf
        )
        opt_prob.addVarGroup(
            f"u_{k}", len_u_seg, value=u_guess[k].flatten(), lower=-np.inf, upper=np.inf
        )

    # Add constraints
    # Collocation
    for k in range(m):
        opt_prob.addConGroup(
            f"collocation_{k}",
            n_collocation_constr_seg,
            lower=0,
            upper=0,
            wrt=[f"x_{k}", f"u_{k}"],
        )

    # Continuity
    for k in range(m - 1):
        opt_prob.addConGroup(
            f"continuity_state_{k}",
            nx,
            lower=0,
            upper=0,
            wrt=[f"x_{k}", f"x_{k + 1}"],
        )
        opt_prob.addConGroup(
            f"continuity_control_{k}",
            nu,
            lower=0,
            upper=0,
            wrt=[f"u_{k}", f"u_{k + 1}"],
        )

    # Additional (boundary)
    opt_prob.addConGroup(
        "additional_constr",
        n_additional_constr,
        lower=0,
        upper=0,
        wrt=[f"x_0", f"x_{m - 1}"],
    )

    # Add objective
    opt_prob.addObj("obj")

    # --- Solve ---
    opt = pyoptsparse.IPOPT(options=optimizer_options)
    with ipopt_pbar_from_file(max_iter=optimizer_options.get("max_iter", 1000)):
        result = opt(opt_prob, sens=sens)

    # --- Unpack Results ---
    x_opt = np.array([result.xStar[f"x_{k}"] for k in range(m)]).reshape((m, N + 1, nx))
    u_opt = np.array([result.xStar[f"u_{k}"] for k in range(m)]).reshape((m, N + 1, nu))

    return x_opt, u_opt, result
