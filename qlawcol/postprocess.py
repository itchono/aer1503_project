import jax
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from qlawcol.driver import ProblemData, Trajectory
from qlawcol.dynamics.conversion import (
    mee_to_cartesian,
    mee_to_keplerian,
)


def plot_elems(col_sol: Trajectory, qlaw_sol: Trajectory):
    col_kep = np.array(jax.vmap(mee_to_keplerian)(col_sol.mee))
    qlaw_kep = np.array(jax.vmap(mee_to_keplerian)(qlaw_sol.mee))

    fig, ax = plt.subplots(5, 1, figsize=(5, 5), sharex=True)
    ax: list[plt.Axes]

    col_kep[:, 3:] = np.degrees(col_kep[:, 3:])
    qlaw_kep[:, 3:] = np.degrees(qlaw_kep[:, 3:])

    labels = ["$a$ (m)", "$e$", "$i$ (deg)", r"$\Omega$ (deg)", r"$\omega$ (deg)"]
    for i in range(5):
        ax[i].plot(col_sol.ts / 86400, col_kep[:, i], label="Collocation", c="r")
        ax[i].plot(qlaw_sol.ts / 86400, qlaw_kep[:, i], label="Q-law", c="k")
        ax[i].set_ylabel(labels[i])
    ax[0].legend()
    ax[-1].set_xlabel(r"$t$ (days)")


def plot_mass_dv(col_sol: Trajectory, qlaw_sol: Trajectory, problem_data: ProblemData):
    v_ex = problem_data.exhaust_velocity
    col_dv = v_ex * np.log(col_sol.mass[0] / col_sol.mass)
    qlaw_dv = v_ex * np.log(qlaw_sol.mass[0] / qlaw_sol.mass)

    fig, ax = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    ax[0].plot(col_sol.ts / 86400, col_sol.mass, label="Collocation", c="r")
    ax[0].plot(qlaw_sol.ts / 86400, qlaw_sol.mass, label="Q-law", c="k")
    ax[0].set_ylabel("Mass (kg)")
    ax[0].legend()

    ax[1].plot(col_sol.ts / 86400, col_dv / 1e3, label="Collocation", c="r")
    ax[1].plot(qlaw_sol.ts / 86400, qlaw_dv / 1e3, label="Q-law", c="k")
    ax[1].set_ylabel(r"$\Delta V$ (km/s)")
    ax[1].set_xlabel(r"$t$ (days)")
    ax[1].legend()


def plot_cart_2d(col_sol: Trajectory, qlaw_sol: Trajectory, problem_data: ProblemData):
    ls = problem_data.initial_kep[0]

    # interpolate qlaw mees more finely
    qlaw_mee = np.array(
        [np.interp(col_sol.ts, qlaw_sol.ts, qlaw_sol.mee[:, i]) for i in range(6)]
    ).T

    cart_col = jax.vmap(mee_to_cartesian)(col_sol.mee / np.array([ls, 1, 1, 1, 1, 1]))
    cart_qlaw = jax.vmap(mee_to_cartesian)(qlaw_mee / np.array([ls, 1, 1, 1, 1, 1]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        cart_col[:, 0] * ls, cart_col[:, 1] * ls, label="Collocation", lw=0.5, c="r"
    )
    ax.plot(cart_qlaw[:, 0] * ls, cart_qlaw[:, 1] * ls, label="Q-law", lw=0.5, c="k")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    earth = Circle((0, 0), 6378e3, color="blue")
    ax.add_patch(earth)
    ax.legend(loc="upper left")


def plot_cart_3d(col_sol: Trajectory, qlaw_sol: Trajectory, problem_data: ProblemData):
    ls = problem_data.initial_kep[0]

    # interpolate qlaw mees more finely
    qlaw_mee = np.array(
        [np.interp(col_sol.ts, qlaw_sol.ts, qlaw_sol.mee[:, i]) for i in range(6)]
    ).T

    cart_col = jax.vmap(mee_to_cartesian)(col_sol.mee / np.array([ls, 1, 1, 1, 1, 1]))
    cart_qlaw = jax.vmap(mee_to_cartesian)(qlaw_mee / np.array([ls, 1, 1, 1, 1, 1]))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        cart_col[:, 0] * ls,
        cart_col[:, 1] * ls,
        cart_col[:, 2] * ls,
        label="Collocation",
        lw=0.3,
        c="r",
    )
    ax.plot(
        cart_qlaw[:, 0] * ls,
        cart_qlaw[:, 1] * ls,
        cart_qlaw[:, 2] * ls,
        label="Q-law",
        lw=0.3,
        c="k",
    )
    ax.legend()
    ax.set_aspect("equal")

    # remove axes
    ax.set_axis_off()


def plot_control(col_sol: Trajectory, qlaw_sol: Trajectory):
    fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    labels = [r"$u_r$", r"$u_\theta$", r"$u_z$"]
    for i in range(3):
        ax[i].plot(col_sol.ts / 86400, col_sol.control[:, i], label="Collocation")
        ax[i].plot(qlaw_sol.ts / 86400, qlaw_sol.control[:, i], label="Q-law")
        ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel("$t$ (d)")
    ax[0].legend()


def plot_results(
    col_sol: Trajectory,
    qlaw_sol: Trajectory,
    problem_data: ProblemData,
    plot_prefix: str,
):
    plt.style.use("qlawcol.clean_plot")

    plot_elems(col_sol, qlaw_sol)
    plt.savefig(f"{plot_prefix}_elements.pdf", bbox_inches="tight")

    plot_mass_dv(col_sol, qlaw_sol, problem_data)
    plt.savefig(f"{plot_prefix}_mass_dv.pdf", bbox_inches="tight")

    plot_cart_2d(col_sol, qlaw_sol, problem_data)
    plt.savefig(f"{plot_prefix}_cart_2d.pdf", bbox_inches="tight")

    plot_cart_3d(col_sol, qlaw_sol, problem_data)
    plt.savefig(f"{plot_prefix}_cart_3d.pdf", bbox_inches="tight")

    plot_control(col_sol, qlaw_sol)
    plt.savefig(f"{plot_prefix}_control.pdf", bbox_inches="tight")
