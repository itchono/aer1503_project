import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from qlawcol.collocation import hs_collocation_sparse, hs_interpolant
from qlawcol.dynamics.conversion import (
    keplerian_to_mee,
    mee_to_cartesian,
    mee_to_keplerian,
)
from qlawcol.dynamics.gve import gve_mee
from qlawcol.dynamics.scaling import R_EARTH, get_tu
from qlawcol.qlaw.control import QLawParams
from qlawcol.qlaw.sim import ODEArgs, dfx, simulate
from scipy.interpolate import CubicSpline

initial_kep = jnp.array([24505.9e3, 0.725, jnp.radians(7.05), 0, 0, 0])
initial_mee = keplerian_to_mee(initial_kep)
initial_mass = 2000.0
target_orbit = jnp.array([42165e3, 0.001, jnp.radians(0.05), 0, 0])
qlaw_params = QLawParams(
    target=target_orbit,
    w_oe=jnp.array([1.0, 1.0, 1.0, 0.0, 0.0]),
    w_pen=0,
    rp_min=1,
    k=100,
    eta=0.2,
    accel_mag=1,
)
ode_args = ODEArgs(
    qlaw_params=qlaw_params,
    thrust=5,  # N
    exhaust_velocity=2000 * 9.81,  # m/s
    convergence_tol=5e-3,
)

ts, mee, mass, control, result = simulate(
    initial_mee, initial_mass, ode_args.as_static(), t_max=200 * 86400, max_steps=20000
)
print("Simulation result:", dfx.RESULTS[result])

# filter out any NaN values (in case of failure modes)
valid_indices = np.where(np.isfinite(ts))
ts = ts[valid_indices]
mee = mee[valid_indices]
mass = mass[valid_indices]
control = control[valid_indices]

delta_v = np.log(mass[0] / mass[-1]) * ode_args.exhaust_velocity
n_revs = max(mee[:, 5]) / (2 * np.pi)
print(f"Timesteps: {len(ts)}")
print(f"ToF: {ts[-1] / 86400:.2f} days")
print(f"Total delta-v expended: {delta_v:.2f} m/s")
print(f"Number of revolutions: {n_revs:.2f}")


# begin collocation
LU = 7000e3
TU = get_tu(LU)
MASSU = initial_mass

T = max(ts) / TU
N = int(n_revs * 35)  # number of intervals per orbit
h = T / N
thrust_nd = ode_args.thrust / (MASSU * LU / TU**2)
vex_nd = ode_args.exhaust_velocity / (LU / TU)

print(f"Using collocation with T={T:.2f} TU, N={N}, h={h:.4f} TU")


def f(x: np.ndarray, u: np.ndarray):
    """
    Collocation variables
    x: (N+1, 7) array of state values at each node (6 orbital elements)
    u: (N+1, 4) array of throttle and direction at each node
    """
    mee, mass = x[:6], x[6]

    throttle, direction = u[0], u[1:]

    thrust_vec = throttle * thrust_nd * direction / jnp.linalg.norm(direction + 1e-8)
    accel_vec = thrust_vec / mass

    A, b = gve_mee(mee)

    mee_dot = A @ accel_vec + b
    mass_dot = -thrust_nd / vex_nd
    return jnp.array([*mee_dot, mass_dot])


def objective(x: np.ndarray, u: np.ndarray):
    # compute delta-vs
    accel = u[:, 0] * thrust_nd / x[:, 6]
    return jnp.trapezoid(accel**2, dx=h)


def constraints(x: np.ndarray) -> np.ndarray:
    # enforce BCs on state
    return jnp.array(
        [
            x[0, 0] - initial_mee[0] / LU,  # a(0) = a0
            x[0, 1] - initial_mee[1],  # f(0) = f0
            x[0, 2] - initial_mee[2],  # g(0) = g0
            x[0, 3] - initial_mee[3],  # h(0) = L0
            x[0, 4] - initial_mee[4],  # k(0) = h0
            x[0, 5] - initial_mee[5],  # L(0) = k0
            x[0, 6] - initial_mass / MASSU,  # m(0) = m0
            x[-1, 0] - target_orbit[0] / LU,  # a(T) = at
            mee_to_keplerian(x[-1, :6])[1] - target_orbit[1],  # e(T) = et
            mee_to_keplerian(x[-1, :6])[2] - target_orbit[2],  # i(T) = it
        ]
    )


# interpolate mees for initial guess
mee_interpolant = CubicSpline(ts, mee, axis=0)
mass_interpolant = CubicSpline(ts, mass, axis=0)


def control_interpolant(t: np.ndarray) -> np.ndarray:
    # linear interpolation of control history
    u_interp = np.zeros((len(t), 3))
    u_interp[:, 0] = np.interp(t, ts, control[:, 0])  # throttle
    u_interp[:, 1] = np.interp(t, ts, control[:, 1])  # alpha
    u_interp[:, 2] = np.interp(t, ts, control[:, 2])  # beta
    return u_interp


t_guess = np.linspace(0, T, N + 1) * TU
mass_guess = mass_interpolant(t_guess)
x_guess = mee_interpolant(t_guess)
x_guess[:, 0] /= LU  # convert SMA to nondimensional units
x_guess = np.hstack((x_guess, mass_guess[:, None] / MASSU))  # append mass to state
u_interp = control_interpolant(t_guess) / (LU / TU**2)

u_guess = np.zeros((N + 1, 4))
u_guess[:, 0] = 1.0  # full throttle
u_guess[:, 1:] = u_interp / (
    np.linalg.norm(u_interp[:, 1:3], axis=1, keepdims=True) + 1e-8
)

problem_args = (
    f,
    objective,
    constraints,
    (x_guess, u_guess),
    T,
)

print(f"Initial guess objective: {objective(x_guess, u_guess):.4e}")

x_opt, u_opt, res = hs_collocation_sparse(
    problem_args,
    max_iter=500,
    tol=1e-8,
    limited_memory_max_history=20,
    nlp_scaling_method="gradient-based",
)

# interpolate state
collocation_interpolant = hs_interpolant(x_opt, u_opt, T, f)
t_interp = np.linspace(0, T, N * 10)
x_hist, u_hist = collocation_interpolant(t_interp)
t_interp = t_interp * TU


dv = np.log(x_opt[0, 6] / x_opt[-1, 6]) * ode_args.exhaust_velocity
print(f"Delta-V: {dv:.2f} m/s")

dv_accel_based = (
    np.trapezoid(u_opt[:, 0] * thrust_nd / x_opt[:, 6], dx=h)
    * ode_args.exhaust_velocity
)
print(f"Delta-V (accel-based): {dv_accel_based:.2f} m/s")

plt.style.use("qlawcol.clean_plot")

ts_plot = np.linspace(0, T, int(n_revs * 100)) * TU
mee_dense = mee_interpolant(ts_plot)

# plot orbital elements
plt.figure(figsize=(9, 8))
plt.subplot(5, 1, 1)
plt.plot(t_interp, x_hist[:, 0] * LU, label="Collocation")
plt.plot(ts_plot, mee_dense[:, 0], label="Q-law")
plt.legend()
plt.ylabel("a")
plt.subplot(5, 1, 2)
plt.plot(t_interp, x_hist[:, 1], label="Collocation")
plt.plot(ts_plot, mee_dense[:, 1], label="Q-law")
plt.ylabel("f")
plt.subplot(5, 1, 3)
plt.plot(t_interp, x_hist[:, 2], label="Collocation")
plt.plot(ts_plot, mee_dense[:, 2], label="Q-law")
plt.ylabel("g")
plt.subplot(5, 1, 4)
plt.plot(t_interp, x_hist[:, 3], label="Collocation")
plt.plot(ts_plot, mee_dense[:, 3], label="Q-law")
plt.ylabel("h")
plt.subplot(5, 1, 5)
plt.plot(t_interp, x_hist[:, 4], label="Collocation")
plt.plot(ts_plot, mee_dense[:, 4], label="Q-law")
plt.ylabel("k")

plt.xlabel("t (s)")


# plot controls
plt.figure(figsize=(9, 4))
plt.subplot(3, 1, 1)

u_qlaw = control_interpolant(ts_plot)
mass_qlaw = mass_interpolant(ts_plot)
qlaw_throttle = np.linalg.norm(u_qlaw, axis=1) * mass_qlaw / ode_args.thrust
qlaw_alpha = np.arctan2(u_qlaw[:, 0], u_qlaw[:, 1]) * 180 / np.pi
qlaw_beta = (
    np.arctan2(u_qlaw[:, 2], np.linalg.norm(u_qlaw[:, :2], axis=1)) * 180 / np.pi
)
col_alpha = np.arctan2(u_hist[:, 1], u_hist[:, 2]) * 180 / np.pi
col_beta = (
    np.arctan2(u_hist[:, 3], np.linalg.norm(u_hist[:, 1:3], axis=1)) * 180 / np.pi
)
plt.plot(
    t_interp,
    u_hist[:, 0],
    label="Optimized (Collocation)",
)
plt.plot(ts_plot, qlaw_throttle, label="Initial Guess (Q-law)")
plt.legend()
plt.ylabel("Throttle")

plt.subplot(3, 1, 2)
plt.plot(t_interp, col_alpha, label="Collocation")
plt.plot(ts_plot, qlaw_alpha, label="Q-law")
plt.ylabel(r"$\alpha$ (deg)")

plt.subplot(3, 1, 3)
plt.plot(
    t_interp,
    col_beta,
    label="Collocation",
)
plt.plot(ts_plot, qlaw_beta, label="Q-law")
plt.ylabel(r"$\beta$ (deg)")
plt.xlabel("t (s)")

# plot trajectory in Cartesian space
mee_array = x_hist[:, :6]

cart_array = jax.vmap(mee_to_cartesian)(mee_array)
plt.figure(figsize=(6, 6))
plt.plot(cart_array[:, 0] * LU, cart_array[:, 1] * LU)
earth = Circle((0, 0), R_EARTH, color="blue", alpha=0.5)
plt.gca().add_patch(earth)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")

plt.show()
