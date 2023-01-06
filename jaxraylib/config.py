"""_summary_
"""
# TODO type hints as well when jax.typing is supported
# https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html?highlight=type%20hints#static-type-annotations

import jax
import jax.numpy as jnp
from jax._src.typing import Array

xp = jnp
jit = jax.jit
jit_kwargs = {}
RAISE = False
NDArray = Array

del Array, jax, jnp
