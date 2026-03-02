from typing import NamedTuple

import diffrax as dfx
import jax
import jax.numpy as jnp

from qlawcol.dynamics.conversion import mee_to_keplerian
from qlawcol.dynamics.gve import gve_mee
from qlawcol.dynamics.scaling import R_EARTH, get_tu
from qlawcol.qlaw.control import QLawParams, qlaw


class ODEState(NamedTuple):
    mee: jnp.ndarray
    mass: float


class ODEArgs(NamedTuple):
    qlaw_params: QLawParams
    thrust: float
    exhaust_velocity: float
    convergence_tol: float

    def as_static(self):
        return self._replace(
            qlaw_params=self.qlaw_params.as_static(),
        )


def sim_control(state: ODEState, args: ODEArgs) -> jnp.ndarray:
    """Compute the control acceleration for the current state."""
    qlaw_params = args.qlaw_params._replace(
        accel_mag=args.thrust / state.mass
    )  # update accel_mag based on current mass
    u = qlaw(mee_to_keplerian(state.mee), qlaw_params)

    return u / (jnp.linalg.norm(u) + 1e-12) * args.thrust / state.mass


def sim_ode(t: float, state: ODEState, args: ODEArgs) -> ODEState:
    """ODE function for simulating the spacecraft state."""
    # unpack state
    mee, mass = state

    # compute control
    u = sim_control(state, args)

    # compute GVE in MEE
    A, b = gve_mee(mee)

    # compute MEE derivatives
    mee_dot = A @ u + b

    # compute mass derivative
    accel_mag = jnp.linalg.norm(u)
    thrust_mag = accel_mag * mass

    mass_dot = -thrust_mag / args.exhaust_velocity  # mass loss due to thrust
    return ODEState(mee_dot, mass_dot)


def guidance_converged(_, state: ODEState, args: ODEArgs, **kwargs) -> bool:
    """Check if guidance has converged."""
    target = args.qlaw_params.target
    current_kep = mee_to_keplerian(state.mee)[:5]
    weighting = jnp.where(args.qlaw_params.w_oe > 0, 1, 0)

    error = current_kep - target
    # wrap Omega, omega using arccos(cos()) trick
    error = error.at[3:5].set(jnp.arccos(jnp.cos(error[3:5])))
    error_norm = jnp.linalg.norm(error * weighting)
    return error_norm < args.convergence_tol


def crashed_into_earth(_, state: ODEState, _args, **kwargs) -> bool:
    """Check if the spacecraft has crashed into Earth."""
    # compute current radius
    a, f, g, h, k, L = state.mee

    p = a * (1 - f**2 - g**2)
    q = 1 + f * jnp.cos(L) + g * jnp.sin(L)
    r = p / q

    return r < 1.0  # in nondimensional units, Earth radius is 1.0


@jax.jit(static_argnames=["args", "t_max", "max_steps"])
def simulate(
    initial_mee: jnp.ndarray,
    initial_mass: float,
    args: ODEArgs,
    t_max: float,
    max_steps: int = 4096,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Simulate the spacecraft trajectory under the Q-law.

    Parameters
    ----------
    initial_mee: array
        Initial mean equinoctial elements, in nondimensionalized units (i.e. SMA should be divided by Earth radius).
    initial_mass: float
        Initial mass of the spacecraft, in kg.
    args: ODEArgs
        Arguments for the ODE function, including Q-law parameters and physical parameters.
    t_max: float
        Maximum simulation time, in seconds.
    max_steps: int
        Maximum number of integration steps.

    Returns
    -------
    ts: array
        Time points of the simulation, in seconds.
    mees: array
        Mean equinoctial elements at each time point, in dimensional units.
    mass: array
        Mass of the spacecraft at each time point, in kg.
    control: array
        Control acceleration at each time point in the LVLH frame, in m/s^2.
    result: Enum
        ODE solver result code
    success: bool
        Whether the simulation ended due to successful convergence (True) or due to hitting max time/steps or crashing into Earth (False).

    """

    # reconstruct args with arrays etc. (for JIT compatibility)
    args = args._replace(
        qlaw_params=args.qlaw_params._replace(
            target=jnp.array(args.qlaw_params.target),
            w_oe=jnp.array(args.qlaw_params.w_oe),
            deadband=jnp.array(args.qlaw_params.deadband),
        )
    )

    # nondimensionalize initial conditions and parameters
    lu = R_EARTH  # use Earth radius as length unit
    tu = get_tu(lu)  # compute time unit
    massu = initial_mass  # use initial mass as mass unit

    t_max_nd = t_max / tu  # nondimensionalize max time
    thrust_nd = args.thrust * tu**2 / lu / massu  # nondimensionalize thrust
    ve_nd = args.exhaust_velocity * tu / lu  # nondimensionalize exhaust velocity
    args = args._replace(
        thrust=thrust_nd,
        exhaust_velocity=ve_nd,
        qlaw_params=args.qlaw_params._replace(
            target=args.qlaw_params.target.at[0].divide(
                lu
            )  # nondimensionalize target MEE
        ),
    )

    initial_mee = initial_mee.at[0].divide(lu)  # nondimensionalize SMA
    initial_mass = initial_mass / massu  # nondimensionalize mass

    # set up ODE solver
    ode_state0 = ODEState(initial_mee, initial_mass)

    controller = dfx.PIDController(rtol=1e-6, atol=1e-6)
    saveat = dfx.SaveAt(t0=True, steps=True)
    term = dfx.ODETerm(lambda t, y, _: sim_ode(t, y, args))
    solver = dfx.Tsit5()

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_max_nd,
        dt0=1e-3,
        y0=ode_state0,
        saveat=saveat,
        stepsize_controller=controller,
        event=dfx.Event(
            [
                lambda t, y, _, **kwargs: guidance_converged(t, y, args),
                lambda t, y, _, **kwargs: crashed_into_earth(t, y, args),
            ]
        ),
        throw=False,
        max_steps=max_steps,
    )

    # postprocess to get control solutions
    u = jax.vmap(sim_control, in_axes=(0, None))(sol.ys, args)

    # rescale solution
    ts = sol.ts * tu  # rescale time
    mee = sol.ys.mee.at[:, 0].multiply(lu)  # rescale MEE (only SMA needs rescaling)
    mass = sol.ys.mass * massu  # rescale mass to physical units
    result = sol.result
    u = u * lu / tu**2  # rescale control to physical units
    success = sol.event_mask[0]

    return ts, mee, mass, u, result, success
