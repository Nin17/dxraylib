"""_summary_
"""

import xraylib


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
