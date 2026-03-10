import numpy as np
from qlawcol.collocation.interpolants import lgl_interpolant
from qlawcol.collocation.lgl_utils import (
    lgl_differentiation_matrix,
    lgl_nodes,
    lgl_weights,
)
from scipy.optimize import minimize

# -------------------------------------------------------
# Problem definition
# -------------------------------------------------------

nx = 2
nu = 1
t0 = 0
tf = 1


def dynamics(x, u):
    return np.array([x[1], u[0]])


# -------------------------------------------------------
# Build collocation problem
# -------------------------------------------------------

N = 20  # number of segments

tau = lgl_nodes(N)
w = lgl_weights(N, tau)
D = lgl_differentiation_matrix(N, tau)

# -------------------------------------------------------
# Helper unpack
# -------------------------------------------------------


def unpack(z):
    X = z[: (N + 1) * nx].reshape((N + 1, nx))
    U = z[(N + 1) * nx :].reshape((N + 1, nu))

    return X, U


# -------------------------------------------------------
# Objective
# -------------------------------------------------------


def objective(z):
    X, U = unpack(z)

    cost = np.sum(w * (U[:, 0] ** 2))

    return (tf - t0) / 2 * cost


# -------------------------------------------------------
# Collocation constraints
# -------------------------------------------------------


def collocation_constraints(z):
    X, U = unpack(z)

    DX = D @ X

    con = []

    for i in range(N + 1):
        f = dynamics(X[i], U[i])

        defect = DX[i] - (tf - t0) / 2 * f

        con.extend(defect)

    return np.array(con)


# -------------------------------------------------------
# Boundary conditions
# -------------------------------------------------------


def boundary_constraints(z):
    X, U = unpack(z)

    con = []

    con.extend(X[0] - np.array([0, 0]))
    con.extend(X[-1] - np.array([1, 0]))

    return np.array(con)


# -------------------------------------------------------
# Initial guess
# -------------------------------------------------------

X_guess = np.zeros((N + 1, nx))
U_guess = np.zeros((N + 1, nu))

X_guess[:, 0] = np.linspace(0, 1, N + 1)

z0 = np.concatenate([X_guess.flatten(), U_guess.flatten()])


# -------------------------------------------------------
# Build constraint list
# -------------------------------------------------------

cons = []

cons.append({"type": "eq", "fun": collocation_constraints})

cons.append({"type": "eq", "fun": boundary_constraints})


# -------------------------------------------------------
# Solve
# -------------------------------------------------------

sol = minimize(
    objective,
    z0,
    constraints=cons,
    method="SLSQP",
    options={"ftol": 1e-9, "maxiter": 1000, "disp": True},
)


X, U = unpack(sol.x)

print("Solved cost:", objective(sol.x))

# -------------------------------------------------------
# Plotting
# -------------------------------------------------------
import matplotlib.pyplot as plt

interpolant = lgl_interpolant(X, U, tf, tau)

t = np.linspace(0, 1, 100)

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
