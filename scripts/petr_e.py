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

initial_kep = jnp.array([25405.9e3, 0.725, jnp.radians(0.06), 0, 0, 0])
initial_mee = keplerian_to_mee(initial_kep)
target_orbit = jnp.array(
    [26500e3, 0.7, jnp.radians(116), jnp.radians(180), jnp.radians(270)]
)
qlaw_params = QLawParams(
    target=target_orbit,
    w_oe=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    w_pen=1,
    rp_min=6578 / 6378,
    k=100,
    eta=0,
    accel_mag=1,
)
ode_args = ODEArgs(
    qlaw_params=qlaw_params,
    thrust=2,  # N
    exhaust_velocity=2000 * 9.81,  # m/s
    convergence_tol=6e-2,
)

ts, mee, mass, control, result, success = simulate(
    initial_mee, 2000.0, ode_args.as_static(), t_max=150 * 86400, max_steps=30000
)
print("Simulation result:", dfx.RESULTS[result])

# filter out any NaN values (in case of failure modes)
valid_indices = jnp.where(jnp.isfinite(ts))
ts = ts[valid_indices]
mee = mee[valid_indices]
mass = mass[valid_indices]
control = control[valid_indices]

delta_v = jnp.log(mass[0] / mass[-1]) * ode_args.exhaust_velocity
delta_v_u = jnp.trapezoid(jnp.linalg.norm(control, axis=1), ts)
print(f"Timesteps: {len(ts)}")
print(f"ToF: {ts[-1] / 86400:.2f} days")
print(f"Propellant Mass Used: {mass[0] - mass[-1]:.2f} kg")
print(f"Total delta-v expended: {delta_v:.2f} m/s")
print(f"Total delta-v from control: {delta_v_u:.2f} m/s")


kep = jax.vmap(mee_to_keplerian)(mee)

# interpolate mees before plotting
interpolant = CubicSpline(ts, mee, axis=0)
n_revs = max(mee[:, 5]) / (2 * jnp.pi)
ts_dense = jnp.linspace(ts[0], ts[-1], int(100 * n_revs))
mee_dense_nd = interpolant(ts_dense)
mee_dense_nd[:, 0] /= R_EARTH


cart = jax.vmap(mee_to_cartesian)(mee_dense_nd)

r_a = kep[:, 0] * (1 + kep[:, 1])
r_p = kep[:, 0] * (1 - kep[:, 1])

inc = jnp.degrees(kep[:, 2])
raan = jnp.degrees(kep[:, 3]) % 360
argp = jnp.degrees(kep[:, 4]) % 360

plt.figure(figsize=(8.75, 9))
plt.subplot(5, 1, 1)
plt.plot(ts / 86400, r_a * 1e-3, color="k")
plt.ylabel("Apoapsis Radius (km)")
plt.subplot(5, 1, 2)
plt.plot(ts / 86400, r_p * 1e-3, color="k")
plt.ylabel("Periapsis Radius (km)")
plt.subplot(5, 1, 3)
plt.plot(ts / 86400, inc, color="k")
plt.ylabel("Inclination (deg)")
plt.subplot(5, 1, 4)
plt.plot(ts / 86400, raan, color="k")
plt.ylabel("RAAN (deg)")
plt.subplot(5, 1, 5)
plt.plot(ts / 86400, argp, color="k")
plt.ylabel("Argument of Periapsis (deg)")
plt.xlabel("Time (days)")
plt.tight_layout()

fig = plt.figure(figsize=(8.75, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    cart[:, 0] * R_EARTH * 1e-3,
    cart[:, 1] * R_EARTH * 1e-3,
    cart[:, 2] * R_EARTH * 1e-3,
    color="k",
    lw=0.5,
)
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
plt.axis("equal")
# change ticks to scilimit (0, 0)

plt.show()
