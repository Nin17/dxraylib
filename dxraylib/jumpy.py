"""
Jump factors.
"""

from __future__ import annotations

from ._indexors import _index2d
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs

_JUMP = _load("jump")


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def JumpFactor(Z: ArrayLike, shell: ArrayLike) -> Array:
    """
    Jump factor.

    Parameters
    ----------
    Z : array_like
        atomic number
    shell : array_like
        shell-type macro

    Returns
    -------
    array
        jump factor
    """
    return _index2d(_JUMP, Z - 1, shell)
