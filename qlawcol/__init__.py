from pathlib import Path
from tempfile import gettempdir

import jax

# enable 64 bit precision
jax.config.update(name="jax_enable_x64", val=True)

cache_dir = Path(gettempdir()) / "jax_cache"

# enable JAX compilation cache
jax.config.update("jax_compilation_cache_dir", str(cache_dir))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 4096)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
