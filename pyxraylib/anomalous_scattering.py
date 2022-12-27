"""_summary_
"""

import xraylib


def fi(atomic_number: int, energy: float) -> float:
    """_summary_

    Parameters
    ----------
    atomic_number : int
        _description_
    energy : float
        _description_

    Returns
    -------
    float
        _description_
    """
    return xraylib.Fi(atomic_number, energy)


def fii(atomic_number: int, energy: float) -> float:
    """_summary_

    Parameters
    ----------
    atomic_number : int
        _description_
    energy : float
        _description_

    Returns
    -------
    float
        _description_
    """
    return xraylib.Fii(atomic_number, energy)
