"""_summary_
"""
# TODO docstrings

import os

from .config import jit, xp
from ._splint import _splint

DIRPATH = os.path.dirname(__file__)

FI_PATH = os.path.join(DIRPATH, "data/fi.npy")
FII_PATH = os.path.join(DIRPATH, "data/fii.npy")

FI = xp.load(FI_PATH)
FII = xp.load(FII_PATH)

del DIRPATH, FI_PATH, FII_PATH


@jit
def Fi(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    return _splint(FI[Z - 1], E)


@jit
def Fii(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    return _splint(FII[Z - 1], E)
