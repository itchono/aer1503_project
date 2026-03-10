import numpy as np
from numpy.polynomial.legendre import Legendre


def lgl_nodes(N: int):
    """
    Computes the Legendre-Gauss-Lobatto nodes for an Nth order polynomial,
    i.e. N+1 nodes including the endpoints -1 and 1.

    The LGL nodes are the roots of (1-x^2)*P'_N(x) = 0,
    i.e. the roots of P'_N(x) plus the endpoints -1 and 1.
    """
    poly = Legendre.basis(N).deriv()
    roots = poly.roots()
    return np.array([-1, *roots, 1])


def lgl_weights(N: int, tau: np.ndarray):
    """Given the LGL nodes, compute the corresponding quadrature weights."""
    Pn = Legendre.basis(N)
    return 2 / (N * (N + 1) * Pn(tau) ** 2)


def lgl_differentiation_matrix(N: int, tau: np.ndarray):
    """
    Computes the LGL differentiation matrix D, where D[i, j] = l_j'(tau_i)
    and l_j is the j-th Lagrange basis polynomial associated with the LGL nodes.
    """
    Pn = Legendre.basis(N)
    D = np.zeros((N + 1, N + 1))

    # off-diagonal entries
    off_diag_mask = ~np.eye(N + 1, dtype=bool)
    i_indices, j_indices = np.where(off_diag_mask)
    D[i_indices, j_indices] = Pn(tau[i_indices]) / (
        Pn(tau[j_indices]) * (tau[i_indices] - tau[j_indices])
    )

    for i in range(N + 1):
        D[i, i] = -np.sum(D[i, :]) + D[i, i]
    return D
