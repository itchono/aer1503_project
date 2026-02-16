import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from qlawcol.collocation import hs_collocation, hs_interpolant
from qlawcol.dynamics.conversion import (
    keplerian_to_mee,
    mee_to_cartesian,
    mee_to_keplerian,
)
from qlawcol.dynamics.gve import gve_2d_mee
from qlawcol.dynamics.scaling import R_EARTH, get_tu
from qlawcol.qlaw.control import QLawParams
from qlawcol.qlaw.sim import ODEArgs, dfx, simulate
from scipy.interpolate import CubicSpline

initial_kep = jnp.array([7000e3, 0.01, 0, 0, 0, 0])
initial_mee = keplerian_to_mee(initial_kep)
target_orbit = jnp.array([8000e3, 0.0, 0.05, 0, 0])
qlaw_params = QLawParams(
    target=target_orbit,
    w_oe=jnp.array([1.0, 1.0, 1.0, 0.0, 0.0]),
    w_pen=0,
    rp_min=1,
    k=100,
    eta=0.5,
    accel_mag=1,
)
ode_args = ODEArgs(
    qlaw_params=qlaw_params,
    thrust=1,  # N
    exhaust_velocity=3100 * 9.81,  # m/s
    convergence_tol=1e-3,
)

ts, mee, mass, u, result = simulate(initial_mee, 300.0, ode_args, t_max=1e7)
print("Simulation result:", dfx.RESULTS[result])

# filter out any NaN values (in case of failure modes)
valid_indices = jnp.where(jnp.isfinite(ts))
ts = ts[valid_indices]
mee = mee[valid_indices]
mass = mass[valid_indices]
u = u[valid_indices]


kep = jax.vmap(mee_to_keplerian)(mee)

# interpolate mees before plotting
mee_interpolant = CubicSpline(ts, mee, axis=0)
control_interpolant = CubicSpline(ts, u, axis=0)
n_revs = max(mee[:, 5]) / (2 * jnp.pi)
ts_dense = jnp.linspace(ts[0], ts[-1], int(100 * n_revs))
mee_dense_nd = mee_interpolant(ts_dense)
mee_dense_nd[:, 0] /= R_EARTH

cart = jax.vmap(mee_to_cartesian)(mee_dense_nd)

# begin collocation
LU = 7000e3
TU = get_tu(LU)

T = max(ts) / TU
N = int(n_revs * 3.5)  # number of intervals
h = T / N

print(f"Using collocation with T={T:.2f} TU, N={N}, h={h:.4f} TU")


def f(x: np.ndarray, u: np.ndarray):
    A, b = gve_2d_mee(x)
    return A @ u + b


def objective(x: np.ndarray, u: np.ndarray):
    # integral of u^2 over time using trapezoidal rule
    return jnp.trapezoid(jnp.linalg.norm(u, axis=1) ** 2, dx=h)


def constraints(x: np.ndarray) -> np.ndarray:
    # enforce BCs on a, f, g
    return jnp.array(
        [
            x[0, 0] - initial_mee[0] / LU,  # a(0) = a0
            x[0, 1] - initial_mee[1],  # f(0) = f0
            x[0, 2] - initial_mee[2],  # g(0) = g0
            x[0, 3] - initial_mee[3],  # L(0) = L0
            x[-1, 0] - target_orbit[0] / LU,  # a(T) = af
            x[-1, 1] - target_orbit[1],  # f(T) = ff
            x[-1, 2] - target_orbit[2],  # g(T) = gf
        ]
    )


# initial guess: linear orbital element change, T/TU rads of true
t_guess = jnp.linspace(0, T, N + 1)

x_guess = mee_interpolant(t_guess * TU)[
    :,
    (0, 1, 2, 5),
]
x_guess[:, 0] /= LU  # convert SMA to nondimensional units
u_guess = control_interpolant(t_guess * TU)[:, (0, 1)] / (
    LU / TU**2
)  # convert back to nondimensional control

problem_args = (
    f,
    objective,
    constraints,
    (x_guess, u_guess),
    T,
)

x_opt, u_opt, res = hs_collocation(problem_args, maxiter=500, ftol=1e-9)

# interpolate state
mee_interpolant = hs_interpolant(x_opt, u_opt, T, f)
t_interp = np.linspace(0, T, N * 10)
x_hist, u_hist = mee_interpolant(t_interp)
t_interp = t_interp * TU


dv = np.trapezoid(jnp.linalg.norm(u_opt, axis=1) * LU / TU**2, dx=h * TU)
print(res)
print(f"Delta-V: {dv:.2f} m/s")

plt.style.use("qlawcol.clean_plot")

# plot orbital elements
plt.figure(figsize=(9, 8))
plt.subplot(3, 1, 1)
plt.plot(t_interp, x_hist[:, 0] * LU, label="Collocation")
plt.plot(ts_dense, mee_dense_nd[:, 0] * R_EARTH, label="Q-law")
plt.legend()
plt.ylabel("a")
plt.subplot(3, 1, 2)
plt.plot(t_interp, x_hist[:, 1], label="Collocation")
plt.plot(ts_dense, mee_dense_nd[:, 1], label="Q-law")
plt.ylabel("f")
plt.subplot(3, 1, 3)
plt.plot(t_interp, x_hist[:, 2], label="Collocation")
plt.plot(ts_dense, mee_dense_nd[:, 2], label="Q-law")
plt.ylabel("g")
plt.xlabel("t (s)")


# plot controls
plt.figure(figsize=(9, 4))
plt.subplot(2, 1, 1)

u_mag = np.linalg.norm(u_hist, axis=1) * LU / TU**2
u_qlaw = control_interpolant(ts_dense)[:, (0, 1)] * LU / TU**2
u_qlaw_mag = np.linalg.norm(u_qlaw, axis=1)
u_qlaw_dir = np.arctan2(u_qlaw[:, 1], u_qlaw[:, 0]) * 180 / np.pi - 90
plt.plot(t_interp, u_mag, label="Optimized (Collocation)")
plt.plot(ts_dense, u_qlaw_mag, label="Initial Guess (Q-law)")
plt.legend()
plt.ylabel("Control Magnitude (m/s^2)")

u_dir = np.arctan2(u_hist[:, 1], u_hist[:, 0]) * 180 / np.pi - 90  # convert to degrees
plt.subplot(2, 1, 2)
plt.plot(t_interp, u_dir, label="Collocation")
plt.plot(ts_dense, u_qlaw_dir, label="Q-law")
plt.ylabel("Control Direction (deg)")

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
