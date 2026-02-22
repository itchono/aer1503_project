import jax
import jax.numpy as jnp
import numpy as np
import pyoptsparse
from jax.experimental.sparse import BCOO

from qlawcol.collocation.col_types import ProblemSpec

PyOptSparseCOO = dict[str, list[float] | list[int]]
# mat = {'coo':[row, col, data], 'shape':[nrow, ncols]}


def collocation_jac_sparsity(
    N: int, nx: int, nu: int
) -> tuple[PyOptSparseCOO, PyOptSparseCOO]:
    """
    construct sparsity pattern for collocation constraints
    collocation constraints depend on x_k, x_k+1, u_k, u_k+1
    x and u are each flattened i.e. x = [x1_0, x2_0, ..., x1_1, x2_1, ...]

    the constraint jacobian wrt both will be sparse since
    the i-th constraint depends only on xl_i, xl_i+1, ul_i, ul_i+1

    General form looks like (1 for nonzero, 0 for zero):
    [1 1 1 1 0 0 0 0 ...]
    [1 1 1 1 0 0 0 0 ...]
    [0 0 1 1 1 1 0 0 ...]
    [0 0 1 1 1 1 0 0 ...]

    - the width of each block is 2 nx for x and 2 nu for u
    - the height of each block is nx (corresponding to constraint dimension)
    """

    row_idx_x = np.repeat(np.arange(N * nx), 2 * nx)
    row_idx_u = np.repeat(np.arange(N * nx), 2 * nu)
    col_idx_x = []
    col_idx_u = []

    jac_x_shape = [N * nx, (N + 1) * nx]
    jac_u_shape = [N * nx, (N + 1) * nu]

    for i in range(N):
        # matrix is N * nx rows tall, so each iteration should
        # "add nx rows". Each row within each "block" is identical.

        # each "block": the nonzero indices move forward by nx and nu for each constraint
        # we have 2 nx and 2 nu nonzeros per row, and this is repeated for each of the nx rows in the block
        col_idx_x.extend(
            ([i * nx + j for j in range(nx)] + [(i + 1) * nx + j for j in range(nx)])
            * nx
        )
        col_idx_u.extend(
            ([i * nu + j for j in range(nu)] + [(i + 1) * nu + j for j in range(nu)])
            * nx
        )

    col_idx_x = np.array(col_idx_x)
    col_idx_u = np.array(col_idx_u)

    one_x = np.ones_like(row_idx_x, dtype=float)
    one_u = np.ones_like(row_idx_u, dtype=float)

    # return in PyOptSparse COO format
    jac_x = {
        "coo": [row_idx_x, col_idx_x, one_x],
        "shape": jac_x_shape,
    }

    jac_u = {
        "coo": [row_idx_u, col_idx_u, one_u],
        "shape": jac_u_shape,
    }

    return jac_x, jac_u


def mask_to_sparse(coo_mask: PyOptSparseCOO, data: np.ndarray) -> PyOptSparseCOO:
    """
    Takes a dense matrix and preserves only the entries corresponding to the nonzero pattern in coo_mask.
    """
    row_idx, col_idx, _ = coo_mask["coo"]
    sparse_data = data[row_idx, col_idx]

    return {
        "coo": [row_idx, col_idx, sparse_data],
        "shape": coo_mask["shape"],
    }


def pyoptsparse_to_jax_bcoo(pyopt_coo: PyOptSparseCOO) -> BCOO:
    """
    Convert a PyOptSparse COO format to a JAX BCOO format for use in sparsejac.
    """
    row_idx, col_idx, data = pyopt_coo["coo"]
    shape = pyopt_coo["shape"]

    # convert to 0-based indexing for JAX
    row_idx = np.array(row_idx)
    col_idx = np.array(col_idx)

    # create BCOO sparse matrix
    bcoo = BCOO((data, np.stack([row_idx, col_idx], axis=1)), shape=shape)

    return bcoo


def jax_bcoo_to_pyoptsparse(bcoo: BCOO) -> PyOptSparseCOO:
    """
    Convert a JAX BCOO format to a PyOptSparse COO format for use in pyoptsparse.
    """
    data, indices = bcoo.data, bcoo.indices
    row_idx, col_idx = indices[:, 0], indices[:, 1]
    shape = bcoo.shape

    return {
        "coo": [row_idx, col_idx, data],
        "shape": shape,
    }


def detect_sparsity_pattern(jac: np.ndarray) -> PyOptSparseCOO:
    """
    Utility function to detect the sparsity pattern of a dense jacobian and convert it to PyOptSparse COO format.
    This can be used to probe the sparsity pattern of the additional constraints jacobian.
    """
    mask = jnp.abs(jac) > 1e-8

    return {
        "coo": [
            np.where(mask)[0],
            np.where(mask)[1],
            np.ones(np.sum(mask)),
        ],
        "shape": [mask.shape[0], mask.shape[1]],
    }
