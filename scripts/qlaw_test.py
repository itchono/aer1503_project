from code.qlawcol.qlaw.control import QLawParams

import jax.numpy as jnp
from matplotlib import pyplot as plt
from qlawcol.qlaw.sim import ODEArgs, simulate

initial_orbit = jnp.array([7000e3, 0.01, jnp.radians(0.05), 0.01, 0.01, 0])
target_orbit = jnp.array([42000e3, 0.01, 0, 0, 0])
qlaw_params = QLawParams(
    target=target_orbit,
    w_oe=jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]),
    w_pen=0,
    rp_min=1,
    k=100,
    eta=0,
    accel_mag=1,
)
ode_args = ODEArgs(
    qlaw_params=qlaw_params,
    thrust=1,  # N
    exhaust_velocity=3100 * 9.81,  # m/s
    convergence_tol=1e-3,
)

ts, kep, mass = simulate(initial_orbit, 300.0, ode_args, t_max=1e7)

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
plt.show()
