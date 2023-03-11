"""_summary_
"""
# TODO docstring

from __future__ import annotations
import os

from ._indexors import _index2d
from ._utilities import asarray, wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp

_DIRPATH = os.path.dirname(__file__)
_JUMP_PATH = os.path.join(_DIRPATH, "data/jump.npy")
_JUMP = xp.load(_JUMP_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def JumpFactor(Z: ArrayLike, shell: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        atomic number
    shell : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """

    return _index2d(_JUMP, Z - 1, shell)
