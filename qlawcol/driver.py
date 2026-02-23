from typing import NamedTuple

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

from qlawcol.collocation import hs_collocation_sparse, hs_interpolant
from qlawcol.dynamics.conversion import keplerian_to_mee, mee_to_keplerian
from qlawcol.dynamics.gve import gve_mee
from qlawcol.dynamics.scaling import get_tu
from qlawcol.qlaw.control import QLawParams
from qlawcol.qlaw.sim import ODEArgs, simulate


class ProblemData(NamedTuple):
    initial_kep: np.ndarray
    initial_mass: float
    qlaw_params: QLawParams
    t_max: float
    thrust: float
    exhaust_velocity: float
    ode_maxsteps: int = 4096
    qlaw_tol: float = 1e-2
    col_segments_per_rev: int = 20


class Trajectory(NamedTuple):
    ts: np.ndarray
    mee: np.ndarray
    mass: np.ndarray
    control: np.ndarray

    def dump_to_file(self, fname: str):
        np.savez(fname, ts=self.ts, mee=self.mee, mass=self.mass, control=self.control)

    @staticmethod
    def load_from_file(fname: str) -> "Trajectory":
        data = np.load(fname)
        return Trajectory(
            ts=data["ts"], mee=data["mee"], mass=data["mass"], control=data["control"]
        )


class CollocationParams(NamedTuple):
    N: int
    T: float
    LU: float
    TU: float
    MASSU: float


class Result(NamedTuple):
    qlaw: Trajectory
    collocation: Trajectory
    message: str


def generate_initial_guess_for_collocation(
    problem_data: ProblemData, sol_q: Trajectory
) -> tuple[Trajectory, CollocationParams]:
    """
    Given a solution from the Q-law, generate an initial guess for collocation,
    returning the initial guess trajectory and the scalings.
    """
    T = sol_q.ts[-1]
    n_revs = max(sol_q.mee[:, 5]) / (2 * np.pi)
    N = int(np.ceil(n_revs * problem_data.col_segments_per_rev))

    LU = problem_data.initial_kep[0]
    TU = get_tu(LU)
    MASSU = problem_data.initial_mass

    col_params = CollocationParams(N=N, T=T, LU=LU, TU=TU, MASSU=MASSU)

    # nondimensionalize arrays before interpolation
    ts_nd = sol_q.ts / TU
    mee_nd = sol_q.mee / np.array([LU, 1, 1, 1, 1, 1])
    mass_nd = sol_q.mass / MASSU
    control_nd = sol_q.control * (LU / TU**2)

    # stack arrays for interpolation (simple linear interpolation)
    f_stacked = np.hstack((mee_nd, mass_nd[:, None], control_nd))

    ts_col = np.linspace(0, T, N + 1) / TU
    f_guess = np.array(
        [np.interp(ts_col, ts_nd, f_stacked[:, i]) for i in range(f_stacked.shape[1])]
    ).T

    # unstack interpolated arrays
    mee_guess = f_guess[:, :6]
    mass_guess = f_guess[:, 6]
    control_guess = f_guess[:, 7:]

    # change control arrays to required format (throttle + direction)
    throttle_guess = np.ones_like(control_guess[:, 0])  # full throttle
    control_dir_guess = control_guess / (
        np.linalg.norm(control_guess, axis=1, keepdims=True) + 1e-12
    )
    control_guess = np.hstack((throttle_guess[:, None], control_dir_guess))

    traj_guess_nd = Trajectory(
        ts=ts_col, mee=mee_guess, mass=mass_guess, control=control_guess
    )
    return traj_guess_nd, col_params


def collocate(
    problem_data: ProblemData,
    col_guess: Trajectory,
    col_params: CollocationParams,
    **collocation_kwargs,
) -> tuple[Trajectory, str]:
    initial_mee = keplerian_to_mee(problem_data.initial_kep)

    N, T, LU, TU, MASSU = col_params
    thrust_nd = problem_data.thrust / (MASSU * LU / TU**2)
    vex_nd = problem_data.exhaust_velocity / (LU / TU)
    h = T / N / TU  # nondimensional timestep

    def f(x: np.ndarray, u: np.ndarray):
        """
        Collocation variables
        x: (N+1, 7) array of state values at each node (6 orbital elements)
        u: (N+1, 4) array of throttle and direction at each node
        """
        mee, mass = x[:6], x[6]

        throttle, direction = u[0], u[1:]

        thrust_mag = throttle * thrust_nd

        thrust_vec = thrust_mag * direction / jnp.linalg.norm(direction + 1e-8)
        accel_vec = thrust_vec / mass

        A, b = gve_mee(mee)

        mee_dot = A @ accel_vec + b
        mass_dot = -thrust_mag / vex_nd
        return jnp.array([*mee_dot, mass_dot])

    def objective(x: np.ndarray, u: np.ndarray):
        # compute delta-vs
        # accel = u[:, 0] / x[:, 6]
        # return jnp.trapezoid(accel**2, dx=h)
        return -x[-1, 6]

    def constraints(x: np.ndarray) -> np.ndarray:
        # enforce BCs on state
        ic_constraints = jnp.array(
            [
                x[0, 0] - initial_mee[0] / LU,  # a(0) = a0
                x[0, 1] - initial_mee[1],  # f(0) = f0
                x[0, 2] - initial_mee[2],  # g(0) = g0
                x[0, 3] - initial_mee[3],  # h(0) = L0
                x[0, 4] - initial_mee[4],  # k(0) = h0
                x[0, 5] - initial_mee[5],  # L(0) = k0
                x[0, 6] - 1,  # m(0) = m0
            ]
        )

        terminal_kep_state = mee_to_keplerian(x[-1, :6])

        with jax.ensure_compile_time_eval():
            # add terminal constraints only if Q-law controls for them
            target_kep_nd = problem_data.qlaw_params.target / jnp.array(
                [LU, 1, 1, 1, 1]
            )
            w_qlaw = problem_data.qlaw_params.w_oe
            extras_list = []
            for i, w in enumerate(w_qlaw):
                if w > 0:
                    extras_list.append(target_kep_nd[i] - terminal_kep_state[i])

        return jnp.concatenate((ic_constraints, jnp.array(extras_list)))

    state_guess = np.hstack((col_guess.mee, col_guess.mass[:, None]))

    problem_args = (
        f,
        objective,
        constraints,
        (state_guess, col_guess.control),
        T / TU,
    )

    print(f"Initial guess objective: {objective(state_guess, col_guess.control):.4e}")

    x_opt, u_opt, res = hs_collocation_sparse(problem_args, **collocation_kwargs)

    collocation_interpolant = hs_interpolant(x_opt, u_opt, T / TU, f)
    t_interp = np.linspace(0, T / TU, N * 10)
    x_hist, u_hist = collocation_interpolant(t_interp)
    ts_col = t_interp * TU

    # convert to dimensional units
    mass_col = x_hist[:, 6] * MASSU
    mee_col = np.array(x_hist[:, :6])
    mee_col[:, 0] *= LU  # convert SMA back to dimensional units
    throttle_col = u_hist[:, 0]
    direction_col = u_hist[:, 1:] / np.linalg.norm(u_hist[:, 1:], axis=1, keepdims=True)
    control_col = (
        throttle_col[:, None]
        * direction_col
        * thrust_nd
        / x_hist[:, 6][:, None]
        * (LU / TU**2)
    )

    return Trajectory(ts=ts_col, mee=mee_col, mass=mass_col, control=control_col), res


def optimize_transfer(problem_data: ProblemData, **collocation_kwargs) -> Result:
    # simulate with Q-law
    ode_args = ODEArgs(
        qlaw_params=problem_data.qlaw_params.as_static(),
        thrust=problem_data.thrust,
        exhaust_velocity=problem_data.exhaust_velocity,
        convergence_tol=problem_data.qlaw_tol,
    )
    initial_mee = keplerian_to_mee(problem_data.initial_kep)

    ts_q, mee_q, mass_q, control_q, result, success = simulate(
        initial_mee,
        problem_data.initial_mass,
        ode_args,
        t_max=problem_data.t_max,
        max_steps=problem_data.ode_maxsteps,
    )
    print(f"Q-law simulation result: {dfx.RESULTS[result]}, success: {success}")

    # filter out any NaN values (in case of failure modes)
    valid_indices = np.where(np.isfinite(ts_q))
    ts_q = ts_q[valid_indices]
    mee_q = mee_q[valid_indices]
    mass_q = mass_q[valid_indices]
    control_q = control_q[valid_indices]

    q_solution = Trajectory(ts_q, mee_q, mass_q, control_q)

    # report on qlaw solution: # revs, final error
    n_revs = mee_q[-1, 5] / (2 * np.pi)
    final_kep = mee_to_keplerian(mee_q[-1])
    target_kep = problem_data.qlaw_params.target
    error = np.linalg.norm(
        (final_kep[:5] - target_kep)
        / np.array([problem_data.initial_kep[0], 1, 1, 1, 1])
        * problem_data.qlaw_params.w_oe
    )
    print(f"Q-law solution: {n_revs:.2f} revs, final error {error:.2e}")

    # run collocation
    col_guess, col_params = generate_initial_guess_for_collocation(
        problem_data, q_solution
    )
    col_solution, res = collocate(
        problem_data, col_guess, col_params, **collocation_kwargs
    )

    return Result(q_solution, col_solution, res)
