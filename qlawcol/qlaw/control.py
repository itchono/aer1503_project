from typing import NamedTuple

import jax
import jax.numpy as jnp

from qlawcol.dynamics.gve import gve_kep


class QLawParams(NamedTuple):
    target: jnp.ndarray
    w_oe: jnp.ndarray
    w_pen: float = 0.0
    rp_min: float = 1.0  # nondimensionalized by Earth radius
    k: float = 100.0
    eta: float = 0.0
    accel_mag: float = None
    deadband: jnp.ndarray = jnp.zeros(5)  # extra deadband, added in the EQ-law (2017)
    m: int = 3
    n: int = 4
    r: int = 2

    def as_static(self):
        return self._replace(
            target=tuple(self.target.tolist()),
            w_oe=tuple(self.w_oe.tolist()),
            deadband=tuple(self.deadband.tolist()),
        )


def oexx_kep(kep: jnp.ndarray, f: float) -> jnp.ndarray:
    """
    Maximum rates of change per orbital element over the course of an orbit.

    Parameters
    ----------
    kep: array
        Keplerian elements [a, e, i, Omega, omega, theta].
        a should be expressed in nondimensionalized units (i.e. divided by Earth radius).
    f: float
        Maximum acceleration magnitude, expressed in nondimensionalized units.

    Returns
    -------
    oexx: array
        Maximum rates of change per orbital element, in the same order as kep.
        Time unit is nondimensional.
    """
    a, e, i, _, omega, _ = kep

    p = a * (1 - e**2)
    h = jnp.sqrt(p)

    a_xx = 2 * f * jnp.sqrt((a**3 * (1 + e)) / (1 - e))
    e_xx = 2 * p * f / h
    i_xx = (
        p
        * f
        / (h * (jnp.sqrt(1 - e**2 * jnp.sin(omega) ** 2) - e * jnp.abs(jnp.cos(omega))))
    )

    u = (1 - e**2) / (2 * e**3)
    v = jnp.sqrt(1 / 4 * ((1 - e**2) / e**3) ** 2 + 1 / 27)
    cos_th_xx = (u + v) ** (1 / 3) - (-u + v) ** (1 / 3) - 1 / e
    cos_th_xx_2 = cos_th_xx**2
    sin_th_xx_2 = 1 - cos_th_xx_2
    r_xx = p / (1 + e * cos_th_xx)

    omega_x = f / (e * h) * jnp.sqrt(p**2 * cos_th_xx_2 + (p + r_xx) ** 2 * sin_th_xx_2)

    Omega_xx = (
        p
        * f
        / (
            h
            * jnp.sin(i)
            * (jnp.sqrt(1 - e**2 * jnp.cos(omega) ** 2) - e * jnp.abs(jnp.sin(omega)))
        )
    )

    return jnp.array([a_xx, e_xx, i_xx, Omega_xx, omega_x])


def proximity_quotient(state: jnp.ndarray, params: QLawParams) -> float:
    """
    The Q function from the Q-law paper.

    Parameters
    ----------
    state: array
        State vector, consisting of [a, e, i, Omega, omega, theta].
        a should be expressed in nondimensionalized units (i.e. divided by Earth radius).
    params: QLawParams
        Parameters for the Q-law.

    Returns
    -------
    float
        The proximity quotient whose Lie derivative we seek to make as
        negative as possible on each timestep.
    """
    target, w_oe, w_p, rp_min, k, eta, accel_mag, deadband, m, n, r = params

    dx = state[:5] - target[:5]
    # wrap Omega, omega using arccos(cos()) trick
    dx = dx.at[3:5].set(jnp.arccos(jnp.cos(dx[3:5])))

    dx = jnp.where(jnp.abs(dx) > deadband, dx, 0)

    s_a = (1 + ((state[0] - target[0]) / (m * target[0])) ** n) ** (1 / r)
    s_oe = jnp.array([s_a, 1, 1, 1, 1])

    a, e = state[:2]
    rp = a * (1 - e)
    penalty = jnp.exp(k * (1 - rp / rp_min))

    oexx = oexx_kep(state, accel_mag)

    return (1 + w_p * penalty) * jnp.sum(w_oe * s_oe * (dx / oexx) ** 2)


def qdot_at_theta(theta: float, state: jnp.ndarray, params: QLawParams) -> jnp.ndarray:
    """
    Compute the Lie derivative of the proximity quotient at a given true anomaly.
    Helper function for determine when to cut off thrust.

    Parameters
    ----------
    theta: float
        True anomaly at which to evaluate qdot.
    state: array
        State vector, consisting of [a, e, i, Omega, omega, theta].
        a should be expressed in nondimensionalized units (i.e. divided by Earth radius).
    params: QLawParams
        Parameters for the Q-law.

    Returns
    -------
    float
        The Lie derivative of the proximity quotient at the given true anomaly.
    """
    state = state.at[-1].set(theta)
    G = gve_kep(state)[0]
    grad_q = jax.grad(proximity_quotient)(state, params)
    grad_q = jnp.where(jnp.isfinite(grad_q), grad_q, 0)
    accel_mag = params.accel_mag

    return -(jnp.linalg.norm(G.T @ grad_q) * accel_mag)


def qdot_nn_gss(state: jnp.ndarray, params: QLawParams, tol: float = 1e-6) -> float:
    """
    Find the most negative value of \dot{Q} over the course of an orbit, using the golden section search method.

    Note that the original Q-law paper uses a grid search instead of GSS.

    Parameters
    ----------
    state: array
        State vector, consisting of [a, e, i, Omega, omega, theta].
        a should be expressed in nondimensionalized units (i.e. divided by Earth radius).
    params: QLawParams
        Parameters for the Q-law.
    tol: float
        Tolerance for convergence of the golden section search.

    Returns
    -------
    float
        The most negative value of \dot{Q} over the course of an orbit.
    """
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

        qdot_c = qdot_at_theta(c, state, params)
        qdot_d = qdot_at_theta(d, state, params)

        return jax.lax.cond(
            qdot_c < qdot_d,
            lambda: (a, d),
            lambda: (c, b),
        )

    def wh_cond(val):
        a, b = val
        return (b - a) > tol

    a, b = jax.lax.while_loop(wh_cond, wh_body, (a, b))
    return qdot_at_theta((a + b) / 2, state, params)


def qlaw(state: jnp.ndarray, params: QLawParams) -> jnp.ndarray:
    """
    Compute the Q-law control acceleration for a given state and Q-law parameters.

    Parameters
    ----------
    state: array
        State vector, consisting of [a, e, i, Omega, omega, theta].
        a should be expressed in nondimensionalized units (i.e. divided by Earth radius).
    params: QLawParams
        Parameters for the Q-law.

    Returns
    -------
    control: array
        Control acceleration in the LVLH frame, expressed in nondimensionalized
        units
    """
    G = gve_kep(state)[0]
    grad_q = jax.grad(proximity_quotient)(state, params)
    # treat NaN or inf in grad_q as zero (i.e. if gradient is undefined, apply no control)
    grad_q = jnp.where(jnp.isfinite(grad_q), grad_q, 0)

    # "LgV control: guarantees maximum possible decrease in Q"
    u = -G.T @ grad_q

    # Cutoff decision
    qdot_n = qdot_at_theta(state[-1], state, params)
    qdot_nn = qdot_nn_gss(state, params)
    eta_curr = qdot_n / qdot_nn

    return jax.lax.cond(
        eta_curr < params.eta,
        lambda: jnp.zeros_like(u),
        lambda: u,
    )
