import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from qlawcol.collocation import hlgl_interpolant, sparse_hlgl_collocation
from qlawcol.collocation.initial_guess import linear_sma_guess
from qlawcol.collocation.lgl_utils import lgl_nodes, lgl_weights
from qlawcol.dynamics.conversion import mee_to_cartesian
from qlawcol.dynamics.gve import gve_2d_mee
from qlawcol.dynamics.scaling import R_EARTH, get_tu

T = 40.0  # final time
N = 5  # degree of LGL polynomial (number of nodes - 1)
m = 10  # number of segments for Hermite-LGL collocation

nx = 4  # [a, f, g, L]
nu = 2  # [fr, ft]

LU = 8000e3
TU = get_tu(LU)

tau = lgl_nodes(N)
w = lgl_weights(N, tau)


mee_start = np.array([1.0, 0.0, 0.0, 0.0])
mee_end = np.array([2.0, 0.4, -0.1, 0.0])


def f(x: np.ndarray, u: np.ndarray):
    A, b = gve_2d_mee(x)
    return A @ u + b


def objective(x: np.ndarray, u: np.ndarray):
    cost = jnp.sum(w * (jnp.linalg.norm(u, axis=2) ** 2))
    return (T - 0) / 2 * cost / m


def constraints(x: np.ndarray) -> np.ndarray:
    # enforce BCs on a, f, g
    return jnp.array(
        [
            x[0, 0, 0] - mee_start[0],  # a(0) = a0
            x[0, 0, 1] - mee_start[1],  # f(0) = f0
            x[0, 0, 2] - mee_start[2],  # g(0) = g0
            # x[0, 3] - mee_start[3],  # L(0) = L0
            x[-1, -1, 0] - mee_end[0],  # a(T) = af
            x[-1, -1, 1] - mee_end[1],  # f(T) = ff
            x[-1, -1, 2] - mee_end[2],  # g(T) = gf
        ]
    )


# initial guess: linear orbital element change, T/TU rads of true
global_guess = linear_sma_guess(mee_start, mee_end, (N + 1) * m - 1, T)
# reshape to (m, N+1, nx)
x_guess = global_guess.reshape(m, N + 1, nx)

u_guess = np.ones((m, N + 1, nu)) * 1e-6  # avoid issues with jacobian eval


problem_args = (
    f,
    objective,
    constraints,
    (x_guess, u_guess),
    T,
)

x_opt, u_opt, res = sparse_hlgl_collocation(problem_args, m, N, max_iter=500)

# interpolate state
interpolant = hlgl_interpolant(x_opt, u_opt, T, tau)
t_interp = np.linspace(0, T, (N + 1) * m * 5)
x_hist, u_hist = interpolant(t_interp)
t_interp = t_interp * TU

dv = np.sum(w * np.linalg.norm(u_opt, axis=2)) * (T - 0) / 2 / m * LU / TU
print(f"Delta-V: {dv:.2f} m/s")

plt.style.use("qlawcol.clean_plot")

# plot orbital elements
t = np.linspace(0, T, N + 1) * TU
plt.figure(figsize=(9, 8))
plt.subplot(3, 1, 1)
plt.plot(t_interp, x_hist[:, 0] * LU)
plt.ylabel("a")
plt.subplot(3, 1, 2)
plt.plot(t_interp, x_hist[:, 1])
plt.ylabel("f")
plt.subplot(3, 1, 3)
plt.plot(t_interp, x_hist[:, 2])
plt.ylabel("g")
plt.xlabel("t (s)")


# plot controls
plt.figure(figsize=(9, 4))
plt.subplot(2, 1, 1)

u_mag = np.linalg.norm(u_hist, axis=1) * LU / TU**2
plt.plot(t_interp, u_mag)
plt.ylabel("u (m/s^2)")

u_dir = np.arctan2(u_hist[:, 1], u_hist[:, 0]) * 180 / np.pi - 90  # convert to degrees
plt.subplot(2, 1, 2)
plt.plot(t_interp, u_dir)
plt.ylabel("Direction (deg)")

plt.xlabel("t (s)")

# plot trajectory in Cartesian space
mee_array = np.zeros((x_hist.shape[0], 6))
mee_array[:, :3] = x_hist[:, :3]
mee_array[:, -1] = x_hist[:, -1]


cart_array = jax.vmap(mee_to_cartesian)(mee_array)
plt.figure(figsize=(6, 6))
plt.plot(cart_array[:, 0] * LU, cart_array[:, 1] * LU)
earth = Circle((0, 0), R_EARTH, color="blue", alpha=0.5)
plt.gca().add_patch(earth)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")

plt.show()
