import jax
import numpy as np

from qlawcol.collocation.col_types import Dynamics


def hs_interpolant(x_opt: np.ndarray, u_opt: np.ndarray, T: float, f: Dynamics):
    """
    Creates a cubic Hermite spline interpolant for the state and a linear
    interpolant for the control.
    """
    N = x_opt.shape[0] - 1
    h = T / N
    t_nodes = np.linspace(0, T, N + 1)
    f_vec = jax.vmap(f, in_axes=(0, 0))
    f_opt = f_vec(x_opt, u_opt)

    def interpolant(t: float | np.ndarray):
        """
        Interpolates the state and control at a given time t.
        """
        if isinstance(t, float):
            t = np.array([t])

        # Find the interval for each t
        interval_indices = np.searchsorted(t_nodes, t, side="right") - 1
        interval_indices = np.clip(interval_indices, 0, N - 1)

        # Normalize time in each interval
        tau = (t - t_nodes[interval_indices]) / h

        # Get interval start and end points
        x_k = x_opt[interval_indices]
        x_k_plus_1 = x_opt[interval_indices + 1]
        f_k = f_opt[interval_indices]
        f_k_plus_1 = f_opt[interval_indices + 1]
        u_k = u_opt[interval_indices]
        u_k_plus_1 = u_opt[interval_indices + 1]

        # Hermite basis functions
        H0 = 2 * tau**3 - 3 * tau**2 + 1
        H1 = tau**3 - 2 * tau**2 + tau
        H2 = -2 * tau**3 + 3 * tau**2
        H3 = tau**3 - tau**2

        # Interpolate state
        x_interp = (
            H0[:, None] * x_k
            + H1[:, None] * h * f_k
            + H2[:, None] * x_k_plus_1
            + H3[:, None] * h * f_k_plus_1
        )

        # Interpolate control (linear)
        u_interp = (1 - tau)[:, None] * u_k + tau[:, None] * u_k_plus_1

        if len(t) == 1:
            return x_interp[0], u_interp[0]
        return x_interp, u_interp

    return interpolant


def trapezoidal_interpolant(
    x_opt: np.ndarray, u_opt: np.ndarray, T: float, f: Dynamics
):
    """
    Creates a quadratic interpolant for the state and a linear interpolant for the control
    based on the trapezoidal collocation solution.
    """

    N = x_opt.shape[0] - 1
    h = T / N
    t_nodes = np.linspace(0, T, N + 1)
    f_vec = jax.vmap(f, in_axes=(0, 0))
    f_opt = f_vec(x_opt, u_opt)

    def interpolant(t: float | np.ndarray):
        """
        Interpolates the state and control at a given time t.
        """
        if isinstance(t, float):
            t = np.array([t])

        # Find the interval for each t
        interval_indices = np.searchsorted(t_nodes, t, side="right") - 1
        interval_indices = np.clip(interval_indices, 0, N - 1)
        t_k = t_nodes[interval_indices]
        tau = ((t - t_k) / h)[:, None]

        # Quadratic interpolation for state
        x_k = x_opt[interval_indices]
        x_k_plus_1 = x_opt[interval_indices + 1]
        f_k = f_opt[interval_indices]
        f_k_plus_1 = f_opt[interval_indices + 1]
        x_interp = (
            (1 - tau) * x_k
            + tau * x_k_plus_1
            + tau * (1 - tau) * h / 8 * (f_k_plus_1 - f_k)
        )

        # Linear interpolation for control
        u_k = u_opt[interval_indices]
        u_k_plus_1 = u_opt[interval_indices + 1]
        u_interp = (1 - tau) * u_k + tau * u_k_plus_1

        if x_interp.shape[0] == 1:
            return x_interp[0], u_interp[0]
        return x_interp, u_interp

    return interpolant
