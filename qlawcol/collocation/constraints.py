import jax
import jax.numpy as jnp


def hermite_simpson(x: jnp.ndarray, u: jnp.ndarray, h: float, f) -> jnp.ndarray:
    """
    Collocation constraints for Hermite-Simpson method.
    """
    f_vec = jax.vmap(f, in_axes=(0, 0))
    f_eval = f_vec(x, u)

    f_k = f_eval[:-1]
    f_k_plus_1 = f_eval[1:]

    # get midpoints
    x_c = (x[:-1] + x[1:]) / 2 + (h / 8) * (f_k - f_k_plus_1)
    u_c = (u[:-1] + u[1:]) / 2
    f_c = f_vec(x_c, u_c)

    # collocation condition
    defect = (x[1:] - x[:-1]) / h - (1 / 6) * (f_k + 4 * f_c + f_k_plus_1)
    return defect.flatten()


def trapezoidal(x: jnp.ndarray, u: jnp.ndarray, h: float, f) -> jnp.ndarray:
    """
    Collocation constraints for trapezoidal method.
    """
    f_vec = jax.vmap(f, in_axes=(0, 0))
    f_eval = f_vec(x, u)

    f_k = f_eval[:-1]
    f_k_plus_1 = f_eval[1:]

    # collocation condition
    defect = (x[1:] - x[:-1]) / h - 0.5 * (f_k + f_k_plus_1)
    return defect.flatten()
