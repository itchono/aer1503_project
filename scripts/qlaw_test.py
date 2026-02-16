import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from qlawcol.dynamics.conversion import (
    keplerian_to_mee,
    mee_to_cartesian,
    mee_to_keplerian,
)
from qlawcol.qlaw.control import QLawParams
from qlawcol.qlaw.sim import ODEArgs, dfx, simulate
from scipy.interpolate import CubicSpline

initial_kep = jnp.array([7000e3, 0.01, jnp.radians(0.05), 0, 0, 0])
initial_mee = keplerian_to_mee(initial_kep)
target_orbit = jnp.array([42000e3, 0.01, 0, 0, 0])
qlaw_params = QLawParams(
    target=target_orbit,
    w_oe=jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]),
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

ts, mee, mass, result = simulate(initial_mee, 300.0, ode_args, t_max=1e7)
print("Simulation result:", dfx.RESULTS[result])

# filter out any NaN values (in case of failure modes)
valid_indices = jnp.where(jnp.isfinite(ts))
ts = ts[valid_indices]
mee = mee[valid_indices]
mass = mass[valid_indices]


kep = jax.vmap(mee_to_keplerian)(mee)

# interpolate mees before plotting
interpolant = CubicSpline(ts, mee, axis=0)
n_revs = max(mee[:, 5]) / (2 * jnp.pi)
ts_dense = jnp.linspace(ts[0], ts[-1], int(100 * n_revs))
mee_dense_nd = interpolant(ts_dense)
mee_dense_nd[:, 0] /= 6378e3


cart = jax.vmap(mee_to_cartesian)(mee_dense_nd)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(ts, kep[:, 0] * 1e-3)
plt.axhline(target_orbit[0] * 1e-3, color="r", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Semi-major Axis (km)")
plt.subplot(2, 1, 2)
plt.plot(ts, kep[:, 1])
plt.axhline(target_orbit[1], color="r", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Eccentricity")
plt.tight_layout()

plt.figure()
plt.plot(cart[:, 0] * 6378e3, cart[:, 1] * 6378e3)
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.axis("equal")

plt.show()
