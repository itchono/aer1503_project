import numpy as np


def linear_sma_guess(
    mee_start: np.ndarray,
    mee_end: np.ndarray,
    N: int,
    T: float,
) -> np.ndarray:
    """
    Generate an initial guess for the semi-major axis (sma) that changes linearly
    from mee_start to mee_end over N+1 points.

    T should be in nondimensional time units.
    """
    x_guess = np.zeros((N + 1, len(mee_start)))
    x_guess[:, 0] = np.linspace(mee_start[0], mee_end[0], N + 1)  # linear sma
    x_guess[:, -1] = np.linspace(0, T, N + 1)  # linear angle guess
    return x_guess


def linear_elements_guess(
    mee_start: np.ndarray,
    mee_end: np.ndarray,
    N: int,
    T: float,
) -> np.ndarray:
    """
    Generate an initial guess for the modified equinoctial elements (mee)
    that changes linearly from mee_start to mee_end over N+1 points.

    T should be in nondimensional time units.
    """
    x_guess = np.zeros((N + 1, len(mee_start)))
    s = np.linspace(0, 1, N + 1)
    x_guess[:, :-1] = mee_start[:-1] + np.outer(s, (mee_end - mee_start)[:-1])
    x_guess[:, -1] = np.linspace(0, T, N + 1)  # linear angle guess
    return x_guess


def q_law_guess(
    mee_start: np.ndarray,
    mee_end: np.ndarray,
    N: int,
    T: float,
) -> np.ndarray:
    """
    Generate an initial guess for the modified equinoctial elements (mee)
    using the Q-law.

    T should be in nondimensional time units.
    """
    pass
