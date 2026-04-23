"""Cross sections.

Total cross section  (cm2/g).
Photoelectric absorption cross section  (cm2/g).
Rayleigh scattering cross section  (cm2/g)
Compton scattering cross section  (cm2/g)
Mass energy-absorption coefficient (cm2/g)
"""
# TODO(nin17): issue with E = 1

from __future__ import annotations

__all__ = ["CS_Compt", "CS_Energy", "CS_Photo", "CS_Rayl", "CS_Total"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._interpolate import interpolate1d
from ._load import _load

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray


CS_COMPT_DATA = _load("cs_compt")
CS_ENERGY_DATA = _load("cs_energy")
CS_PHOTO_DATA = _load("cs_photo")
CS_RAYL_DATA = _load("cs_rayl")


def CS_Total(Z: NDArray[integer], E: NDArray) -> NDArray[floating]:
    """Total cross-section: Photoelectric + Compton + Rayleigh.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Total cross-section: Photoelectric + Compton + Rayleigh.
        (cm²/g)

    """
    compton = CS_Compt(Z, E)
    photo = CS_Photo(Z, E)
    rayleigh = CS_Rayl(Z, E)
    return compton + photo + rayleigh


def CS_Photo(Z: NDArray[integer], E: NDArray) -> NDArray[floating]:
    """Photoelectric absorption cross-section.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Photoelectric absorption cross-section
        (cm²/g)

    """
    xp = array_namespace(Z, E)
    data = xp.asarray(CS_PHOTO_DATA)
    return xp.exp(interpolate1d(data, Z, E, xp.log(E * 1000.0), xp=xp))


def CS_Rayl(Z: NDArray[integer], E: NDArray) -> NDArray[floating]:
    """Rayleigh scattering cross-section.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Rayleigh scattering cross-section
        (cm²/g)

    """
    xp = array_namespace(Z, E)
    data = xp.asarray(CS_RAYL_DATA)
    return xp.exp(interpolate1d(data, Z, E, xp.log(E * 1000.0), xp=xp))


def CS_Compt(Z: NDArray[integer], E: NDArray) -> NDArray[floating]:
    """Compton scattering cross-section.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Compton scattering cross-section
        (cm²/g)

    """
    xp = array_namespace(Z, E)
    data = xp.asarray(CS_COMPT_DATA)
    return xp.exp(interpolate1d(data, Z, E, xp.log(E * 1000.0), xp=xp))


def CS_Energy(Z: NDArray[integer], E: NDArray) -> NDArray[floating]:
    """Mass-energy absorption cross-section.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Mass-energy absorption cross-section
        (cm²/g)

    """
    xp = array_namespace(Z, E)
    data = xp.asarray(CS_ENERGY_DATA)
    return xp.exp(interpolate1d(data, Z, E, xp.log(E), xp=xp))
