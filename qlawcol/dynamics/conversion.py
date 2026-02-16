import jax
import jax.numpy as jnp
from jax.numpy import arctan, arctan2, cos, sin


def mee_to_cartesian(mee: jax.Array) -> jax.Array:
    """
    Convert modified equinoctial elements to Cartesian elements.

    Parameters
    ----------
    mee : jax.Array
        Modified equinoctial elements [a, f, g, h, k, L(rad)].
        State must be in non-dimensional units such that mu = 1.

    Returns
    -------
    cart : Array
        Cartesian elements [x, y, z, vx, vy, vz] (LU and LU/TU).

    Notes
    -----
    Formulation from
    https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/EquinoctalElements-modified.pdf

    """
    # unpack state vector
    a, f, g, h, k, L = mee

    # convert SMA and ecc to p
    p = a * (1 - f**2 - g**2)

    # shorthand quantities defined in the document
    al_sq = h**2 - k**2
    s_sq = 1 + h**2 + k**2
    q = 1 + f * cos(L) + g * sin(L)
    r = p / q

    # states
    pos_x = cos(L) + al_sq * cos(L) + 2 * h * k * sin(L)
    pos_y = sin(L) - al_sq * sin(L) + 2 * h * k * cos(L)
    pos_z = 2 * (h * sin(L) - k * cos(L))
    pos = r / s_sq * jnp.array([pos_x, pos_y, pos_z])

    vel_x = -(
        sin(L) + al_sq * sin(L) - 2 * h * k * cos(L) + g - 2 * f * h * k + al_sq * g
    )
    vel_y = -(
        -cos(L) + al_sq * cos(L) + 2 * h * k * sin(L) - f + 2 * g * h * k + al_sq * f
    )
    vel_z = 2 * (h * cos(L) + k * sin(L) + f * h + g * k)
    vel = jnp.sqrt(1 / p) / s_sq * jnp.array([vel_x, vel_y, vel_z])

    return jnp.concatenate([pos, vel])


def keplerian_to_mee(kep: jax.Array) -> jax.Array:
    """
    Convert Keplerian elements to modified equinoctial elements.

    Parameters
    ----------
    kep : jax.Array
        Keplerian elements [a(m), e, i(rad), Omega(rad), omega(rad), theta(rad)].

    Returns
    -------
    mee : Array
        Modified equinoctial elements [a(m), f, g, h, k, L(rad)].

    """
    a, e, i, raan, aop, theta = kep

    # compute the equinoctial elements
    f = e * cos(aop + raan)
    g = e * sin(aop + raan)
    h = jnp.tan(i / 2) * cos(raan)
    k = jnp.tan(i / 2) * sin(raan)
    truelong = aop + theta + raan

    return jnp.array([a, f, g, h, k, truelong])


def mee_to_keplerian(mee: jax.Array) -> jax.Array:
    """
    Convert modified equinoctial elements to Keplerian elements.

    Parameters
    ----------
    mee : jax.Array
        Modified equinoctial elements [a(m), f, g, h, k, L(rad)].

    Returns
    -------
    kep : Array
        Keplerian elements [a(m), e, i(rad), Omega(rad), omega(rad), theta(rad)].

    """
    a, f, g, h, k, truelong = mee

    # compute the Keplerian elements
    e = jnp.sqrt(f**2 + g**2)
    i = 2 * arctan(jnp.sqrt(h**2 + k**2))
    raan = arctan2(k, h)
    aop = arctan2(g * h - f * k, f * h + g * k)
    theta = truelong - aop - raan

    return jnp.array([a, e, i, raan, aop, theta])
