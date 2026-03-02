import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from qlawcol.dynamics.conversion import (
    keplerian_to_mee,
    mee_to_cartesian,
    mee_to_keplerian,
)
from qlawcol.dynamics.scaling import R_EARTH
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
    eta=0.0,
    accel_mag=1,
)
ode_args = ODEArgs(
    qlaw_params=qlaw_params,
    thrust=1,  # N
    exhaust_velocity=3100 * 9.81,  # m/s
    convergence_tol=6e-2,
)

ts, mee, mass, control, result, success = simulate(
    initial_mee, 300.0, ode_args.as_static(), t_max=20 * 86400, max_steps=1000
)
print("Simulation result:", dfx.RESULTS[result])
print("Simulation success:", success)

# filter out any NaN values (in case of failure modes)
valid_indices = jnp.where(jnp.isfinite(ts))
ts = ts[valid_indices]
mee = mee[valid_indices]
mass = mass[valid_indices]
control = control[valid_indices]

delta_v = jnp.log(mass[0] / mass[-1]) * ode_args.exhaust_velocity
print(f"Timesteps: {len(ts)}")
print(f"ToF: {ts[-1] / 86400:.2f} days")
print(f"Propellant Mass Used: {mass[0] - mass[-1]:.2f} kg")
print(f"Total delta-v expended: {delta_v:.2f} m/s")

kep = jax.vmap(mee_to_keplerian)(mee)

# interpolate mees before plotting
interpolant = CubicSpline(ts, mee, axis=0)
n_revs = max(mee[:, 5]) / (2 * jnp.pi)
ts_dense = jnp.linspace(ts[0], ts[-1], int(100 * n_revs))
mee_dense_nd = interpolant(ts_dense)
mee_dense_nd[:, 0] /= R_EARTH


cart = jax.vmap(mee_to_cartesian)(mee_dense_nd)

plt.style.use("qlawcol.clean_plot")
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(ts / 86400, kep[:, 0] * 1e-3)
plt.axhline(target_orbit[0] * 1e-3, color="r", linestyle="--")
plt.xlabel("Time (days)")
plt.ylabel("Semi-major Axis (km)")
plt.subplot(2, 1, 2)
plt.plot(ts / 86400, kep[:, 1])
plt.axhline(target_orbit[1], color="r", linestyle="--")
plt.xlabel("Time (days)")
plt.ylabel("Eccentricity")


plt.figure(figsize=(8.75, 7))
plt.plot(cart[:, 0] * R_EARTH * 1e-3, cart[:, 1] * R_EARTH * 1e-3, color="k", lw=0.5)
plt.xlabel("X (km)")
plt.ylabel("Y (km)")
plt.axis("equal")
# change ticks to scilimit (0, 0)
plt.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
plt.savefig("petr_a1_traj.pdf", bbox_inches="tight")

plt.show()
