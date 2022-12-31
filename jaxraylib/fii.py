"""_summary_
"""
# TODO docstrings

import os

from .config import jit, xp
from ._splint import _splint

DIRPATH = os.path.dirname(__file__)
FII_PATH = os.path.join(DIRPATH, "data/fii.npy")
FII = xp.load(FII_PATH)

del DIRPATH, FII_PATH


@jit
def _Fii(Z: int, E: float) -> float:
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
    return _Fii(Z, E)
