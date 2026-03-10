import matplotlib.pyplot as plt
import numpy as np
from qlawcol.collocation.lgl_utils import (
    lgl_differentiation_matrix,
    lgl_nodes,
    lgl_weights,
)

# -------------------------------------------------------
# Problem definition
# -------------------------------------------------------

nx = 2
nu = 1
t0 = 0
tf = 1


def dynamics(x, u):
    r, v = x
    return np.array([v, u[0]])


# -------------------------------------------------------
# Build collocation problem
# -------------------------------------------------------

# grid size
N = 50


tau = lgl_nodes(N)
w = lgl_weights(N, tau)
D = lgl_differentiation_matrix(N, tau)

# initial and final point
x0 = 0
xf = 2 * np.pi

# domain transformation [-1,1]->[x0,xf]
x = tau * (xf - x0) / 2 + (xf + x0) / 2
# sample the function
y = np.sin(x)


# compute derivative from differentiation matrix
y_dot = 2 / (xf - x0) * D @ y
# plot function and its numerical and analytical derivative
plt.plot(x, y, label="sine")
plt.plot(x, y_dot, "-o", label="cosine (lgl)")
plt.plot(x, np.cos(x), label="cosine (analytical)")
# plt.plot(x,y_dot-np.cos(x))
plt.legend()
plt.grid()
plt.show()
print("quadrature:", w.T @ y)
