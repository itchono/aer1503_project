from typing import NamedTuple

import diffrax as dfx
import jax
import jax.numpy as jnp

from qlawcol.dynamics.conversion import mee_to_keplerian
from qlawcol.dynamics.gve import gve_mee
from qlawcol.dynamics.scaling import R_EARTH, get_tu
from qlawcol.qlaw.control import QLawParams, qlaw_mee


class ODEState(NamedTuple):
    mee: jnp.ndarray
    mass: float


class ODEArgs(NamedTuple):
    qlaw_params: QLawParams
    thrust: float
    exhaust_velocity: float
    convergence_tol: float


def sim_ode(t: float, state: ODEState, args: ODEArgs) -> ODEState:
    """ODE function for simulating the spacecraft state."""
    # unpack state
    mee, mass = state

    # assemble qlaw params using current thrust and mass
    # (replace the accel field)
    qlaw_params = args.qlaw_params._replace(accel_mag=args.thrust / mass)

    # compute control
    u = qlaw_mee(mee, qlaw_params)
    u = u / (jnp.linalg.norm(u) + 1e-12) * args.thrust / mass  # convert to acceleration

    # compute GVE in MEE
    A, b = gve_mee(mee)

    # compute MEE derivatives
    mee_dot = A @ u + b
    mass_dot = -args.thrust / args.exhaust_velocity  # mass loss due to thrust
    return ODEState(mee_dot, mass_dot)


def guidance_converged(_, state: ODEState, args: ODEArgs, **kwargs) -> bool:
    """Check if guidance has converged."""
    target = args.qlaw_params.target
    current_mee = state.mee[:5]
    weighting = jnp.where(args.qlaw_params.w_oe > 0, 1, 0)
    error = (current_mee - target) * weighting
    error_norm = jnp.linalg.norm(error)
    return error_norm < args.convergence_tol


def crashed_into_earth(_, state: ODEState, _args, **kwargs) -> bool:
    """Check if the spacecraft has crashed into Earth."""
    # compute current radius
    a, f, g, h, k, L = state.mee

    p = a * (1 - f**2 - g**2)
    q = 1 + f * jnp.cos(L) + g * jnp.sin(L)
    r = p / q

    return r < 1.0  # in nondimensional units, Earth radius is 1.0


@jax.jit
def simulate(
    initial_mee: jnp.ndarray,
    initial_mass: float,
    args: ODEArgs,
    t_max: float,
) -> dfx.SaveAt:
    """Simulate the spacecraft trajectory under the Q-law."""

    # nondimensionalize
    lu = R_EARTH  # use Earth radius as length unit
    tu = get_tu(lu)  # compute time unit

    t_max_nd = t_max / tu  # nondimensionalize max time
    thrust_nd = args.thrust * tu**2 / lu  # nondimensionalize thrust
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

    # set up ODE solver
    ode_state0 = ODEState(initial_mee, initial_mass)

    controller = dfx.PIDController(rtol=1e-6, atol=1e-6)
    saveat = dfx.SaveAt(steps=True)
    term = dfx.ODETerm(sim_ode)
    solver = dfx.Tsit5()
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_max_nd,
        dt0=1e-3,
        y0=ode_state0,
        args=args,
        saveat=saveat,
        stepsize_controller=controller,
        event=dfx.Event([guidance_converged, crashed_into_earth]),
        max_steps=32768,
        throw=False,
    )

    # rescale solution
    ts = sol.ts * tu  # rescale time
    mee = sol.ys.mee.at[:, 0].multiply(lu)  # rescale MEE (only SMA needs rescaling)
    mass = sol.ys.mass  # mass is already in physical units
    result = sol.result

    return ts, mee, mass, result
