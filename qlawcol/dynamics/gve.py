import jax.numpy as jnp
from jax.numpy import cos, sin, sqrt


def gve_2d_mee(mee: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes affine "matrices" A and b for the 2 dimensional
    Gauss variational equations in modified equinoctial elements (MEE).

    The modified equinoctial elements are defined as:
    [a, f, g, L]

    where:
    - a: semi-major axis
    - f: e * cos(omega + Omega)
    - g: e * sin(omega + Omega)
    - L: True longitude = Omega + omega + theta
    """
    # unpack state vector
    a, f, g, ell = mee

    # constructions
    p = a * (1 - f**2 - g**2)  # semi-latus rectum p = a(1 - e^2)
    q = 1 + f * jnp.cos(ell) + g * jnp.sin(ell)
    leading_coefficient = 1 / q * jnp.sqrt(p)

    # A
    A = leading_coefficient * jnp.array(
        [
            [
                2 * a * q * (f * jnp.sin(ell) - g * jnp.cos(ell)) / (1 - f**2 - g**2),
                2 * a * q**2 / (1 - f**2 - g**2),
            ],
            [
                q * jnp.sin(ell),
                (q + 1) * jnp.cos(ell) + f,
            ],
            [
                -q * jnp.cos(ell),
                (q + 1) * jnp.sin(ell) + g,
            ],
            [0, 0],
        ]
    )

    b = jnp.array([0, 0, 0, q**2 * jnp.sqrt(p) / p**2])

    return A, b


def gve_mee(state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gauss variational equation coefficients for
    a-modified equinoctial elements under no additional perturbations.

    i.e. [a f g h k L]

    Parameters
    ----------
    state : Array
        State vector in modified equinoctial elements.
    mu : float
        Gravitational parameter.

    Returns
    -------
    A : Array
        A-matrix for Gauss variational equation.
    b : Array
        b-vector for Gauss variational equation.

    """
    # unpack state vector
    a, f, g, h, k, L = state

    # convert SMA and ecc to p
    p = a * (1 - f**2 - g**2)  # semi-latus rectum p = a(1 - e^2)

    # shorthand quantities
    q = 1 + f * cos(L) + g * sin(L)

    leading_coefficient = 1 / q * sqrt(p)

    # A-matrix
    A = (
        jnp.array(
            [
                [
                    2 * a * q * (f * sin(L) - g * cos(L)) / (1 - f**2 - g**2),
                    2 * a * q**2 / (1 - f**2 - g**2),
                    0,
                ],
                [
                    q * sin(L),
                    (q + 1) * cos(L) + f,
                    -g * (h * sin(L) - k * cos(L)),
                ],
                [
                    -q * cos(L),
                    (q + 1) * sin(L) + g,
                    f * (h * sin(L) - k * cos(L)),
                ],
                [0, 0, cos(L) / 2 * (1 + h**2 + k**2)],
                [0, 0, sin(L) / 2 * (1 + h**2 + k**2)],
                [0, 0, h * sin(L) - k * cos(L)],
            ],
        )
        * leading_coefficient
    )

    # b-vector
    b = jnp.array([0, 0, 0, 0, 0, q**2 * sqrt(p) / p**2])

    return A, b
