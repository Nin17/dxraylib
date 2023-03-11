"""_summary_
"""
# TODO issue with E = 1
# TODO docstrings

from __future__ import annotations
import os

from ._interpolators import _interpolate
from ._utilities import asarray, wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp

_DIRPATH = os.path.dirname(__file__)

_CS_COMPT_PATH = os.path.join(_DIRPATH, "data/cs_compt.npy")
_CS_ENERGY_PATH = os.path.join(_DIRPATH, "data/cs_energy.npy")
_CS_PHOTO_PATH = os.path.join(_DIRPATH, "data/cs_photo.npy")
_CS_RAYL_PATH = os.path.join(_DIRPATH, "data/cs_rayl.npy")

_CS_COMPT = xp.load(_CS_COMPT_PATH)
_CS_ENERGY = xp.load(_CS_ENERGY_PATH)
_CS_PHOTO = xp.load(_CS_PHOTO_PATH)
_CS_RAYL = xp.load(_CS_RAYL_PATH)


@wrapped_partial(jit, **jit_kwargs)
def CS_Total(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Total cross section  (cm2/g)
    (Photoelectric + Compton + Rayleigh)

    Parameters
    ----------
    Z : ArrayLike
        atomic number
    E : ArrayLike
        energy (keV)

    Returns
    -------
    NDArray
        Total cross section  (cm2/g)
        (Photoelectric + Compton + Rayleigh)
    """
    compton = CS_Compt(Z, E)
    photo = CS_Photo(Z, E)
    rayleigh = CS_Rayl(Z, E)
    return compton + photo + rayleigh


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CS_Photo(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Photoelectric absorption cross section  (cm2/g)

    Parameters
    ----------
    Z : ArrayLike
        atomic number
    E : ArrayLike
        energy (keV)

    Returns
    -------
    NDArray
        Photoelectric absorption cross section  (cm2/g)
    """
    return xp.exp(_interpolate(_CS_PHOTO, Z, E, xp.log(E * 1000.0)))


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CS_Rayl(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Rayleigh scattering cross section  (cm2/g)

    Parameters
    ----------
    Z : ArrayLike
        atomic number
    E : ArrayLike
        energy (keV)

    Returns
    -------
    NDArray
        Rayleigh scattering cross section  (cm2/g)
    """
    return xp.exp(_interpolate(_CS_RAYL, Z, E, xp.log(E * 1000.0)))


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CS_Compt(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Compton scattering cross section  (cm2/g)

    Parameters
    ----------
    Z : ArrayLike
        atomic number
    E : ArrayLike
        energy (keV)

    Returns
    -------
    NDArray
        Compton scattering cross section  (cm2/g)
    """
    return xp.exp(_interpolate(_CS_COMPT, Z, E, xp.log(E * 1000.0)))


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CS_Energy(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        atomic number
    E : ArrayLike
        energy (keV)

    Returns
    -------
    NDArray
        _description_
    """
    return xp.exp(_interpolate(_CS_ENERGY, Z, E, xp.log(E)))
