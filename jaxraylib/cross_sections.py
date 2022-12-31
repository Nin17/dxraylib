"""_summary_
"""
# TODO cross sections in other units
import os

from .config import jit, xp
from ._splint import _splint

MEC2 = 511.0034  # electron rest mass (keV)
RE2 = 0.07940775  # square of classical electron radius (barn)
PI = xp.pi

DIRPATH = os.path.dirname(__file__)

CS_COMPT_PATH = os.path.join(DIRPATH, "data/cs_compt.npy")
CS_PHOTO_PATH = os.path.join(DIRPATH, "data/cs_photo.npy")
CS_RAYL_PATH = os.path.join(DIRPATH, "data/cs_rayl.npy")

CS_COMPT = xp.load(CS_COMPT_PATH)
CS_PHOTO = xp.load(CS_PHOTO_PATH)
CS_RAYL = xp.load(CS_RAYL_PATH)

del DIRPATH, CS_COMPT_PATH, CS_PHOTO_PATH, CS_RAYL_PATH


@jit
def CS_Compt(Z: int, E: float) -> float:
    """_summary_

    Parameters
    ----------z
    Z : int
        _description_
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
    return xp.exp(_splint(CS_COMPT[Z - 1], xp.log(E * 1000.0)))


def CS_Energy():
    # !!! Not needed
    return


# TODO fix type hint
@jit
def _CS_KN(E: float) -> float:
    """_summary_

    Parameters
    ----------
    E : float
        _description_

    Returns
    -------
    float
        _description_
    """
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
    if E < 0:
        raise ValueError("energy must be positive")
    return _CS_KN(E)


@jit
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
    return xp.exp(_splint(CS_PHOTO[Z - 1], xp.log(E * 1000.0)))


@jit
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
    return xp.exp(_splint(CS_RAYL[Z - 1], xp.log(E * 1000.0)))


@jit
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
