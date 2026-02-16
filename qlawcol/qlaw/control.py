from typing import NamedTuple

import jax
import jax.numpy as jnp

from qlawcol.dynamics.gve import gve_mee


class QLawParams(NamedTuple):
    target: jnp.ndarray
    w_oe: jnp.ndarray
    w_pen: float
    accel_mag: float
    rp_min: float
    k: float
    eta: float
    deadband: jnp.ndarray = jnp.zeros(5)
    m: int = 3
    n: int = 4
    r: int = 2


def oexx_mee(mee_5: jax.Array, accel_mag: float) -> jax.Array:
    """
    Computes maximum achievable rates of change of
    modified equinoctial elements when applying
    a constant acceleration of accel_mag.

    Uses a simplified approximate formulation for f and g.

    Parameters
    ----------
    mee_5: jax.Array
        Modified equinoctial elements in the form
        [a, f, g, h, k], which excludes true longitude.
    max_accel: float
        Maximum acceleration to be applied (in appropriate units).
    """
    a, f, g, h, k = mee_5

    # compute eccentricity and semi-latus rectum
    e = jnp.sqrt(f**2 + g**2)
    p = a * (1 - e**2)

    a_dot_max = 2 * a * jnp.sqrt(a) * jnp.sqrt((1 + e) / (1 - e))
    f_dot_max = 2 * jnp.sqrt(p)
    g_dot_max = 2 * jnp.sqrt(p)
    s_squared = 1 + h**2 + k**2
    h_dot_max = 1 / 2 * jnp.sqrt(p) * s_squared / (jnp.sqrt(1 - g**2) + f)
    k_dot_max = 1 / 2 * jnp.sqrt(p) * s_squared / (jnp.sqrt(1 - f**2) + g)

    oexx = jnp.array([a_dot_max, f_dot_max, g_dot_max, h_dot_max, k_dot_max])

    return oexx * accel_mag


def proximity_quotient(state: jnp.ndarray, params: QLawParams) -> float:
    target, w_oe, w_p, accel_mag, rp_min, k, eta, deadband, m, n, r = params

    dx = state[:5] - target[:5]

    w_oe = jnp.where(jnp.abs(dx) > deadband, w_oe, 0)

    s_a = (1 + ((state[0] - target[0]) / (m * target[0])) ** n) ** (1 / r)
    s_oe = jnp.array([s_a, 1, 1, 1, 1])

    a, e = state[:2]
    rp = a * (1 - e)
    penalty = jnp.exp(k * (1 - rp / rp_min))

    oexx = oexx_mee(state[:5], accel_mag)

    return (1 + w_p * penalty) * jnp.sum(w_oe * s_oe * (dx / oexx) ** 2)


def qdot_at_truelong(
    truelong: float, state: jnp.ndarray, params: QLawParams
) -> jnp.ndarray:
    state = state.at[-1].set(truelong)
    G = gve_mee(state)[0]
    grad_q = jax.grad(proximity_quotient)(state, params)

    return -(jnp.linalg.norm(G.T @ grad_q) ** 2)


def qdot_nn_gss(state: jnp.ndarray, params: QLawParams, tol: float = 1e-5) -> float:
    # find minimum qdot

    # Golden ratio
    phi = (1 + jnp.sqrt(5)) / 2
    res = 2 - phi  # reciprocal of golden ratio

    a = -jnp.pi
    b = jnp.pi

    def wh_body(val):
        a, b = val

        c = a + res * (b - a)
        d = b - res * (b - a)

        qdot_c = qdot_at_truelong(c, state, params)
        qdot_d = qdot_at_truelong(d, state, params)

        return jax.lax.cond(
            qdot_c < qdot_d,
            lambda: (a, d),
            lambda: (c, b),
        )

    def wh_cond(val):
        a, b = val
        return (b - a) > tol

    a, b = jax.lax.while_loop(wh_cond, wh_body, (a, b))
    return qdot_at_truelong((a + b) / 2, state, params)


def qlaw_mee(state: jnp.ndarray, params: QLawParams) -> jnp.ndarray:
    G = gve_mee(state)[0]
    grad_q = jax.grad(proximity_quotient)(state, params)

    u = -G.T @ grad_q

    qdot_n = qdot_at_truelong(state[-1], state, params)
    qdot_nn = qdot_nn_gss(state, params)
    eta_curr = qdot_n / qdot_nn

    return jax.lax.cond(
        eta_curr < params.eta,
        lambda: jnp.zeros_like(u),
        lambda: u,
    )
