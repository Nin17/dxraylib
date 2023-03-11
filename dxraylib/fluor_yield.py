"""_summary_
"""
# TODO docstring

from __future__ import annotations
import os

from ._indexors import _index2d
from ._utilities import asarray, wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp

_DIRPATH = os.path.dirname(__file__)
_FY_PATH = os.path.join(_DIRPATH, "data/fluor_yield.npy")
_FY = xp.load(_FY_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def FluorYield(Z: ArrayLike, shell: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        _description_
    shell : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _index2d(_FY, Z - 3, shell)
