from code.qlawcol.qlaw.control import QLawParams, qlaw_kep
from typing import NamedTuple

import diffrax as dfx
import jax
import jax.numpy as jnp

from qlawcol.dynamics.conversion import keplerian_to_mee, mee_to_keplerian
from qlawcol.dynamics.gve import gve_mee
from qlawcol.dynamics.scaling import R_EARTH, get_tu


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
    kep = mee_to_keplerian(mee)

    # assemble qlaw params using current thrust and mass
    # (replace the accel field)
    qlaw_params = args.qlaw_params._replace(accel_mag=args.thrust / mass)

    # compute control in Keplerian elements
    u = qlaw_kep(kep, qlaw_params)
    u = u / (jnp.linalg.norm(u) + 1e-12) * args.thrust / mass  # convert to acceleration

    # compute GVE in MEE
    A, b = gve_mee(mee, mu=1.0)

    # compute MEE derivatives
    mee_dot = A @ u + b
    mass_dot = -args.thrust / args.exhaust_velocity  # mass loss due to thrust
    return ODEState(mee_dot, mass_dot)


def guidance_converged(_, state: ODEState, args: ODEArgs, **kwargs) -> bool:
    """Check if guidance has converged."""
    target = args.qlaw_params.target
    current_kep = mee_to_keplerian(state.mee)

    # take difference in Keplerian elements (q-law style)
    diff = current_kep[:5] - target
    diff = diff.at[3:].set(jnp.acos(jnp.cos(diff[3:])))  # wrap angles

    return jnp.linalg.norm(diff) < args.convergence_tol


def crashed_into_earth(_, state: ODEState, _args, **kwargs) -> bool:
    """Check if the spacecraft has crashed into Earth."""
    # compute current radius
    a, f, g, h, k, L = state.mee

    p = a * (1 - f**2 - g**2)
    q = 1 + f * jnp.cos(L) + g * jnp.sin(L)
    r = p / q

    return r < 1.0  # in nondimensional units, Earth radius is 1.0


def simulate(
    initial_kep: jnp.ndarray,
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
    args = args._replace(thrust=thrust_nd, exhaust_velocity=ve_nd)  # update args

    initial_kep = initial_kep.at[0].divide(lu)  # nondimensionalize SMA

    # convert initial state to MEE (also nondimensionalized)
    initial_mee = keplerian_to_mee(initial_kep)

    # set up ODE solver
    ode_state0 = ODEState(initial_mee, initial_mass)

    controller = dfx.PIDController(rtol=1e-6, atol=1e-9)
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
        throw=False,
        max_steps=16384,
    )

    # rescale solution
    ts = sol.ts * tu  # rescale time
    mee = sol.ys.mee.at[:, 0].multiply(lu)  # rescale MEE (only SMA needs rescaling)
    mass = sol.ys.mass  # mass is already in physical units

    # batch convert MEE to Keplerian for easier analysis
    keps = jax.vmap(mee_to_keplerian)(mee)

    return ts, keps, mass
