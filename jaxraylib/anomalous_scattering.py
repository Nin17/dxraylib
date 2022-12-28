"""_summary_
"""

import os

from ._io import _load
from ._splint import _splint

DIRPATH = os.path.dirname(__file__)

FI_PATH = os.path.join(DIRPATH, "xraylib/data/fi.dat")
FII_PATH = os.path.join(DIRPATH, "xraylib/data/fii.dat")

FI = _load(FI_PATH, "\t")
FII = _load(FII_PATH, "\t")


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
    x, y, y2 = FI[Z]
    return _splint(x, y, y2, len(x), E)


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
    x, y, y2 = FII[Z]
    return _splint(x, y, y2, len(x), E)
