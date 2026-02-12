import numpy as np


def gve_2d_mee(mee: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    q = 1 + f * np.cos(ell) + g * np.sin(ell)
    leading_coefficient = 1 / q * np.sqrt(p)

    # A
    A = leading_coefficient * np.array(
        [
            [
                2 * a * q * (f * np.sin(ell) - g * np.cos(ell)) / (1 - f**2 - g**2),
                2 * a * q**2 / (1 - f**2 - g**2),
            ],
            [
                q * np.sin(ell),
                (q + 1) * np.cos(ell) + f,
            ],
            [
                -q * np.cos(ell),
                (q + 1) * np.sin(ell) + g,
            ],
            [0, 0],
        ]
    )

    b = np.array([0, 0, 0, q**2 * np.sqrt(p) / p**2])

    return A, b
