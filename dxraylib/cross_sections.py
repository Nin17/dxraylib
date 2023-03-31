"""_summary_
"""
# TODO summary
# TODO issue with E = 1
# TODO docstrings

from __future__ import annotations

from ._interpolators import _interpolate
from ._load import _load
from ._utilities import asarray, wrapped_partial
from .config import Array, ArrayLike, jit, jit_kwargs, xp

_CS_COMPT = _load("cs_compt")
_CS_ENERGY = _load("cs_energy")
_CS_PHOTO = _load("cs_photo")
_CS_RAYL = _load("cs_rayl")


@wrapped_partial(jit, **jit_kwargs)
def CS_Total(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Total cross-section  (cm²/g): Photoelectric + Compton + Rayleigh.

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        Total cross-section  (cm²/g): Photoelectric + Compton + Rayleigh.
    """
    compton = CS_Compt(Z, E)
    photo = CS_Photo(Z, E)
    rayleigh = CS_Rayl(Z, E)
    return compton + photo + rayleigh


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CS_Photo(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Photoelectric absorption cross-section (cm²/g).

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        Photoelectric absorption cross-section (cm²/g)
    """
    return xp.exp(_interpolate(_CS_PHOTO, Z, E, xp.log(E * 1000.0)))


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CS_Rayl(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Rayleigh scattering cross-section (cm²/g)

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        Rayleigh scattering cross-section (cm²/g)
    """
    return xp.exp(_interpolate(_CS_RAYL, Z, E, xp.log(E * 1000.0)))


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CS_Compt(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Compton scattering cross-section (cm²/g)

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        Compton scattering cross-section (cm²/g)
    """
    return xp.exp(_interpolate(_CS_COMPT, Z, E, xp.log(E * 1000.0)))


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CS_Energy(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Mass-energy absorption cross-section (cm²/g).

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        Mass-energy absorption cross-section (cm²/g)
    """
    return xp.exp(_interpolate(_CS_ENERGY, Z, E, xp.log(E)))
