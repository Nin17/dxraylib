"""
Jump factors.
"""

from __future__ import annotations
import os

from ._indexors import _index2d
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_DIRPATH = os.path.dirname(__file__)
_JUMP_PATH = os.path.join(_DIRPATH, "data/jump.npy")
_JUMP = xp.load(_JUMP_PATH)


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
