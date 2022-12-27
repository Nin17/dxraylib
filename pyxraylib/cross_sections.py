"""_summary_
"""

import xraylib

from .config import xp

MEC2 = 511.0034  # electron rest mass (keV)
RE2 = 0.07940775
PI = xp.pi  # square of classical electron radius (barn)


def cs_compt(atomic_number: int, energy: float) -> float:
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
    return xraylib.CS_Compt(atomic_number, energy)


def cs_energy():
    # !!! Not needed
    return


def cs_kn(energy: float) -> float:
    """_summary_

    Parameters
    ----------
    energy : float
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

    if energy < 0.0:
        raise ValueError("energy must be positive")
    a = energy / MEC2
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


def cs_photo(atomic_number: int, energy: float) -> float:
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
    return xraylib.CS_Photo(atomic_number, energy)


def cs_rayl(atomic_number: int, energy: float) -> float:
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
    return xraylib.CS_Rayl(atomic_number, energy)


def cs_total(atomic_number: int, energy: float) -> float:
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
    compton = cs_compt(atomic_number, energy)
    photo = cs_photo(atomic_number, energy)
    rayleigh = cs_rayl(atomic_number, energy)

    return compton + photo + rayleigh
