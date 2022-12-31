"""_summary_
"""
# TODO docstrings
import os

from .config import jit, xp
from ._splint import _splint

DIRPATH = os.path.dirname(__file__)
FI_PATH = os.path.join(DIRPATH, "data/fi.npy")
FI = xp.load(FI_PATH)


@jit
def _Fi(Z: int, E: float) -> float:
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
    return _Fi(Z, E)
