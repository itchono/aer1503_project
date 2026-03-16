import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from qlawcol.collocation.interpolants import hlgl_interpolant
from qlawcol.collocation.lgl_utils import lgl_nodes, lgl_weights
from qlawcol.collocation.sparse.sparse_hlgl import sparse_hlgl_collocation

T = 1.0  # final time
N = 3  # degree of LGL polynomial (number of nodes - 1)
m = 10  # number of segments for Hermite-LGL collocation

tau = lgl_nodes(N)
w = lgl_weights(N, tau)

nx = 2  # [r, v]
nu = 1  # [u]

r0, v0 = 0.0, 0.0
rf, vf = 1.0, 0.0


def objective(x, u):
    cost = np.sum(np.sum(w * (u[:, :, 0] ** 2), axis=1), axis=0)

    return (T - 0) / 2 / m * cost


def f(x: jnp.ndarray, u: jnp.ndarray):
    r, v = x
    return jnp.array([v, u[0]])


def constraints(x: jnp.ndarray) -> jnp.ndarray:
    # enforce boundary conditions
    return jnp.array(
        [
            x[0, 0, 0] - r0,  # r(0) = r0
            x[0, 0, 1] - v0,  # v(0) = v0
            x[-1, -1, 0] - rf,  # r(T) = rf
            x[-1, -1, 1] - vf,  # v(T) = vf
        ]
    )


x_guess = np.zeros((m, N + 1, nx))
x_guess[:, :, 0] = np.linspace(r0, rf, N + 1)  # linear position guess
x_guess[:, :, 1] = 0.0

u_guess = np.ones((m, N + 1, nu)) * 1e-6

problem_args = (
    f,
    objective,
    constraints,
    (x_guess, u_guess),
    T,
)


x_opt, u_opt, res = sparse_hlgl_collocation(problem_args, m, N)

interpolant = hlgl_interpolant(x_opt, u_opt, T, tau)

t = np.linspace(0, T, 100)

x_interp, u_interp = interpolant(t)


plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, x_interp[:, 0])
plt.ylabel("r")

plt.subplot(3, 1, 2)
plt.plot(t, x_interp[:, 1])
plt.ylabel("v")

plt.subplot(3, 1, 3)
plt.plot(t, u_interp[:, 0])
plt.ylabel("u")
plt.xlabel("t")

plt.tight_layout()
plt.show()
