import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# -----------------------------
# Problem setup
# -----------------------------

T = 1.0  # final time
N = 40  # number of intervals
h = T / N

nx = 2  # [r, v]
nu = 1  # [u]

# Boundary conditions
r0, v0 = 0.0, 0.0
rf, vf = 1.0, 0.0

# -----------------------------
# Dynamics
# -----------------------------


def f(x, u):
    r, v = x
    return np.array([v, u[0]])


# -----------------------------
# Decision variable packing
# -----------------------------


def unpack(z):
    """
    z = [x0, x1, ..., xN, u0, u1, ..., uN]
    """
    x = z[: (N + 1) * nx].reshape((N + 1, nx))
    u = z[(N + 1) * nx :].reshape((N + 1, nu))
    return x, u


# -----------------------------
# Objective function
# -----------------------------


def objective(z):
    x, u = unpack(z)
    J = 0.0
    for k in range(N):
        J += 0.5 * h * (u[k] ** 2 + u[k + 1] ** 2)
    return J.item()


# -----------------------------
# Collocation constraints
# -----------------------------


def constraints(z):
    x, u = unpack(z)
    cons = []

    # Trapezoidal collocation on interior points
    for k in range(N):
        fk = f(x[k], u[k])
        fk1 = f(x[k + 1], u[k + 1])
        defect = x[k + 1] - x[k] - 0.5 * h * (fk + fk1)
        cons.extend(defect.flatten().tolist())

    # Boundary conditions
    cons.append(x[0, 0] - r0)
    cons.append(x[0, 1] - v0)
    cons.append(x[-1, 0] - rf)
    cons.append(x[-1, 1] - vf)

    return np.array(cons)


# -----------------------------
# Initial guess
# -----------------------------

x_guess = np.zeros((N + 1, nx))
x_guess[:, 0] = np.linspace(r0, rf, N + 1)  # linear position guess
x_guess[:, 1] = 0.0

u_guess = np.zeros((N + 1, nu))

z0 = np.hstack([x_guess.flatten(), u_guess.flatten()])

# -----------------------------
# Solve NLP
# -----------------------------

cons = {"type": "eq", "fun": constraints}

res = minimize(
    objective,
    z0,
    method="SLSQP",
    constraints=cons,
    options={"ftol": 1e-9, "maxiter": 1000},
)

print("Success:", res.success)
print("Cost:", res.fun)
print(res)

# -----------------------------
# Plot result
# -----------------------------

x_opt, u_opt = unpack(res.x)
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
