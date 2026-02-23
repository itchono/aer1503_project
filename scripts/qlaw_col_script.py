import jax
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from qlawcol.driver import ProblemData, optimize_transfer
from qlawcol.dynamics.conversion import mee_to_cartesian
from qlawcol.qlaw.control import QLawParams

problem_data = ProblemData(
    initial_kep=np.array([7000e3, 0.01, np.radians(0.01), 0, 0, 0]),
    initial_mass=300.0,
    qlaw_params=QLawParams(
        target=np.array([20000e3, 0.3, np.radians(0.01), 0, 0]),
        w_oe=np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        eta=0.8,
    ),
    t_max=100 * 86400,
    thrust=10,
    exhaust_velocity=3000 * 9.81,
    ode_maxsteps=16384,
    col_segments_per_rev=30,
    qlaw_tol=5e-3,
)


res = optimize_transfer(problem_data, max_iter=500)

col_sol = res.collocation
qlaw_sol = res.qlaw

# save to file
col_sol.dump_to_file("col_sol.npz")
qlaw_sol.dump_to_file("qlaw_sol.npz")


dv_col = np.log(col_sol.mass[0] / col_sol.mass[-1]) * problem_data.exhaust_velocity
dv_q = np.log(qlaw_sol.mass[0] / qlaw_sol.mass[-1]) * problem_data.exhaust_velocity
print(f"Collocation Delta-V: {dv_col:.2f} m/s")
print(f"Q-law Delta-V: {dv_q:.2f} m/s")


plt.style.use("qlawcol.clean_plot")

# plot orbital elements
plt.figure(figsize=(9, 8))
plt.subplot(5, 1, 1)
plt.plot(col_sol.ts, col_sol.mee[:, 0], label="Collocation")
plt.plot(qlaw_sol.ts, qlaw_sol.mee[:, 0], label="Q-law")
plt.legend()
plt.ylabel("$a$ (m)")
plt.subplot(5, 1, 2)
plt.plot(col_sol.ts, col_sol.mee[:, 1], label="Collocation")
plt.plot(qlaw_sol.ts, qlaw_sol.mee[:, 1], label="Q-law")
plt.ylabel("$f$")
plt.subplot(5, 1, 3)
plt.plot(col_sol.ts, col_sol.mee[:, 2], label="Collocation")
plt.plot(qlaw_sol.ts, qlaw_sol.mee[:, 2], label="Q-law")
plt.ylabel("$g$")
plt.subplot(5, 1, 4)
plt.plot(col_sol.ts, col_sol.mee[:, 3], label="Collocation")
plt.plot(qlaw_sol.ts, qlaw_sol.mee[:, 3], label="Q-law")
plt.ylabel("$h$")
plt.subplot(5, 1, 5)
plt.plot(col_sol.ts, col_sol.mee[:, 4], label="Collocation")
plt.plot(qlaw_sol.ts, qlaw_sol.mee[:, 4], label="Q-law")
plt.ylabel("$k$")

plt.xlabel("$t$ (s)")

# plot mass
plt.figure(figsize=(9, 4))
plt.plot(col_sol.ts, col_sol.mass, label="Collocation")
plt.plot(qlaw_sol.ts, qlaw_sol.mass, label="Q-law")
plt.legend()
plt.ylabel("Mass (kg)")
plt.xlabel("$t$ (s)")


# plot controls
plt.figure(figsize=(9, 4))
plt.subplot(3, 1, 1)
plt.plot(col_sol.ts, col_sol.control[:, 0], label="Collocation")
plt.plot(qlaw_sol.ts, qlaw_sol.control[:, 0], label="Q-law")
plt.legend()
plt.ylabel("$u_r$")

plt.subplot(3, 1, 2)
plt.plot(col_sol.ts, col_sol.control[:, 1], label="Collocation")
plt.plot(qlaw_sol.ts, qlaw_sol.control[:, 1], label="Q-law")
plt.ylabel(r"$u_\theta$")

plt.subplot(3, 1, 3)
plt.plot(col_sol.ts, col_sol.control[:, 2], label="Collocation")
plt.plot(qlaw_sol.ts, qlaw_sol.control[:, 2], label="Q-law")
plt.ylabel(r"$u_z$")
plt.xlabel(r"$t$ (s)")

# plot trajectory in Cartesian space

length_scale = problem_data.initial_kep[0]

# interpolate qlaw mees more finely
qlaw_mee = np.array(
    [np.interp(col_sol.ts, qlaw_sol.ts, qlaw_sol.mee[:, i]) for i in range(6)]
)

cart_col = jax.vmap(mee_to_cartesian)(
    col_sol.mee / np.array([length_scale, 1, 1, 1, 1, 1])
)
cart_qlaw = jax.vmap(mee_to_cartesian)(
    qlaw_mee.T / np.array([length_scale, 1, 1, 1, 1, 1])
)


plt.figure(figsize=(6, 6))
plt.plot(
    cart_col[:, 0] * length_scale,
    cart_col[:, 1] * length_scale,
    label="Collocation",
    lw=0.5,
)
plt.plot(
    cart_qlaw[:, 0] * length_scale,
    cart_qlaw[:, 1] * length_scale,
    label="Q-law",
    lw=0.5,
)
earth = Circle((0, 0), 6378e3, color="blue", alpha=0.5)
plt.gca().add_patch(earth)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.legend()

plt.show()
