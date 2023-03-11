"""_summary_
"""
# TODO docstring

from __future__ import annotations
import os

from ._indexors import _index2d
from ._utilities import asarray, wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp

_DIRPATH = os.path.dirname(__file__)
_AU_PATH = os.path.join(_DIRPATH, "data/auger_rates.npy")
_AU = xp.load(_AU_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def AugerRate(Z: ArrayLike, auger_trans: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        _description_
    auger_trans : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _index2d(_AU, Z - 6, auger_trans)
