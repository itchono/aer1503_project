import matplotlib.pyplot as plt
import numpy as np
from qlawcol.collocation.trapezoidal import trapezoidal_collocation
from qlawcol.dynamics.gve import gve_2d_mee

T = 12.0  # final time
N = 40  # number of intervals
h = T / N

nx = 4  # [a, f, g, L]
nu = 2  # [fr, ft]


mee_start = np.array([1.0, 0.0, 0.0, 0.0])
mee_end = np.array([1.5, 0.0, 0.0, 0.0])


def f(x: np.ndarray, u: np.ndarray):
    A, b = gve_2d_mee(x)
    return A @ u + b


def objective(x: np.ndarray, u: np.ndarray):
    # integral of u^2 over time using trapezoidal rule
    return np.trapezoid(np.linalg.norm(u, axis=1) ** 2, dx=h)


def constraints(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    # enforce BCs on a, f, g
    return np.array(
        [
            x[0, 0] - mee_start[0],  # a(0) = a0
            x[0, 1] - mee_start[1],  # f(0) = f0
            x[0, 2] - mee_start[2],  # g(0) = g0
            x[-1, 0] - mee_end[0],  # a(T) = af
            x[-1, 1] - mee_end[1],  # f(T) = ff
            x[-1, 2] - mee_end[2],  # g(T) = gf
        ]
    )


x_guess = np.zeros((N + 1, nx))
x_guess[:, 0] = np.linspace(mee_start[0], mee_end[0], N + 1)  # linear position guess
x_guess[:, -1] = np.linspace(0, T, N + 1)  # linear angle guess
u_guess = np.zeros((N + 1, nu))


problem_args = (
    f,
    objective,
    constraints,
    (x_guess, u_guess),
    T,
)

x_opt, u_opt, res = trapezoidal_collocation(problem_args)


print("Success:", res.success)
print("Cost:", res.fun)
print(res)


# plot orbital elements
t = np.linspace(0, T, N + 1)
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(t, x_opt[:, 0])
plt.ylabel("a")
plt.subplot(3, 1, 2)
plt.plot(t, x_opt[:, 1])
plt.ylabel("f")
plt.subplot(3, 1, 3)
plt.plot(t, x_opt[:, 2])
plt.ylabel("g")


# plot controls
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t, u_opt[:, 0])
plt.ylabel("fr")
plt.subplot(2, 1, 2)
plt.plot(t, u_opt[:, 1])
plt.ylabel("ft")
plt.xlabel("t")

plt.show()
