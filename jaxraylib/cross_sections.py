"""_summary_
"""

import os


from .config import xp
from ._io import _load
from ._splint import _splint

MEC2 = 511.0034  # electron rest mass (keV)
RE2 = 0.07940775  # square of classical electron radius (barn)
PI = xp.pi

DIRPATH = os.path.dirname(__file__)

CS_COMPT_PATH = os.path.join(DIRPATH, 'xraylib/data/CS_Compt.dat' )
CS_PHOTO_PATH = os.path.join(DIRPATH, 'xraylib/data/CS_Photo.dat')
CS_RAYL_PATH = os.path.join(DIRPATH, 'xraylib/data/CS_Rayl.dat')

CS_COMPT = _load(CS_COMPT_PATH)
CS_PHOTO = _load(CS_PHOTO_PATH)
CS_RAYL = _load(CS_RAYL_PATH)


def CS_Compt(Z: int, E: float) -> float:
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
    x, y, y2 = CS_COMPT[Z]
    return xp.exp(_splint(x, y, y2, len(x), xp.log(E * 1000)))


def CS_Energy():
    # !!! Not needed
    return


# TODO fix type hint
def CS_KN(E: float) -> float:
    """_summary_

    Parameters
    ----------
    E : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    if E < 0.0:
        raise ValueError("energy must be positive")
    a = E / MEC2
    a3 = a * a * a
    b = 1 + 2 * a
    b2 = b * b
    lb = xp.log(b)
    sigma = (
        2
        * PI
        * RE2
        * (
            (1 + a) / a3 * (2 * a * (1 + a) / b - lb)
            + 0.5 * lb / a
            - (1 + 3 * a) / b2
        )
    )
    return sigma


def CS_Photo(Z: int, E: float) -> float:
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
    x, y, y2 = CS_PHOTO[Z]
    return xp.exp(_splint(x, y, y2, len(x), xp.log(E * 1000)))


def CS_Rayl(Z: int, E: float) -> float:
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
    x, y, y2 = CS_RAYL[Z]
    return xp.exp(_splint(x, y, y2, len(x), xp.log(E * 1000)))


def CS_Total(Z: int, E: float) -> float:
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
    compton = CS_Compt(Z, E)
    photo = CS_Photo(Z, E)
    rayleigh = CS_Rayl(Z, E)

    return compton + photo + rayleigh
