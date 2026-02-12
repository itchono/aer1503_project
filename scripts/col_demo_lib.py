import matplotlib.pyplot as plt
import numpy as np
from qlawcol.collocation.trapezoidal import trapezoidal_collocation

T = 1.0  # final time
N = 40  # number of intervals
h = T / N

nx = 2  # [r, v]
nu = 1  # [u]

r0, v0 = 0.0, 0.0
rf, vf = 1.0, 0.0


def f(x: np.ndarray, u: np.ndarray):
    r, v = x
    return np.array([v, u[0]])


def objective(x: np.ndarray, u: np.ndarray):
    J = 0.0
    for k in range(N):
        J += 0.5 * h * (u[k] ** 2 + u[k + 1] ** 2)
    return J.item()


def constraints(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    # enforce boundary conditions
    return np.array(
        [
            x[0, 0] - r0,  # r(0) = r0
            x[0, 1] - v0,  # v(0) = v0
            x[-1, 0] - rf,  # r(T) = rf
            x[-1, 1] - vf,  # v(T) = vf
        ]
    )


x_guess = np.zeros((N + 1, nx))
x_guess[:, 0] = np.linspace(r0, rf, N + 1)  # linear position guess
x_guess[:, 1] = 0.0

u_guess = np.zeros((N + 1, nu))


problem_args = (
    f,
    objective,
    constraints,
    (x_guess, u_guess),
    T,
)
slsqp_kwargs = {"options": {"maxiter": 1000, "ftol": 1e-9}}

x_opt, u_opt, res = trapezoidal_collocation(problem_args, **slsqp_kwargs)


print("Success:", res.success)
print("Cost:", res.fun)
print(res)

# -----------------------------
# Plot result
# -----------------------------

t = np.linspace(0, T, N + 1)

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, x_opt[:, 0])
plt.ylabel("r")

plt.subplot(3, 1, 2)
plt.plot(t, x_opt[:, 1])
plt.ylabel("v")

plt.subplot(3, 1, 3)
plt.plot(t, u_opt[:, 0])
plt.ylabel("u")
plt.xlabel("t")

plt.tight_layout()
plt.show()
