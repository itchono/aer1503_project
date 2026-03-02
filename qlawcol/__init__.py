import jax

# enable 64 bit precision
jax.config.update(name="jax_enable_x64", val=True)
