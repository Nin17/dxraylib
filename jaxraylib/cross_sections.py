"""_summary_
"""
# TODO cross sections in other units
# TODO redo with xp.where n stuff
# TODO issue with E = 1
from __future__ import annotations
import os
from typing import overload

from .config import jit, jit_kwargs, xp, NDArray
from ._splint import _splint
from ._utilities import value_error, wrapped_partial, output_type

PI = xp.pi

DIRPATH = os.path.dirname(__file__)

CS_COMPT_PATH = os.path.join(DIRPATH, "data/cs_compt.npy")
CS_ENERGY_PATH = os.path.join(DIRPATH, "data/cs_energy.npy")
CS_PHOTO_PATH = os.path.join(DIRPATH, "data/cs_photo.npy")
CS_RAYL_PATH = os.path.join(DIRPATH, "data/cs_rayl.npy")

CS_COMPT = xp.load(CS_COMPT_PATH)
CS_ENERGY = xp.load(CS_ENERGY_PATH)
CS_PHOTO = xp.load(CS_PHOTO_PATH)
CS_RAYL = xp.load(CS_RAYL_PATH)


@wrapped_partial(jit, **jit_kwargs)
def _CS_Total(
    Z: int | NDArray[int], E: float | NDArray[float]
) -> tuple[NDArray[float, bool]]:
    """
    Total cross section  (cm2/g)
    (Photoelectric + Compton + Rayleigh)

    Parameters
    ----------
    Z : int | Array
        atomic number
    E : float | Array
        energy (keV)

    Returns
    -------
    float | Array
        Total cross section  (cm2/g)
        (Photoelectric + Compton + Rayleigh)
    """
    compton = _CS_Compt(Z, E)[0]
    photo = _CS_Photo(Z, E)[0]
    rayleigh = _CS_Rayl(Z, E)[0]
    output = compton + photo + rayleigh
    return output, xp.isnan(output).any()


@overload
def CS_Total(Z: int, E: float) -> float:
    ...


@overload
def CS_Total(Z: NDArray[int], E: NDArray[float]) -> NDArray[float]:
    ...


@output_type
@value_error(" # TODO value error")
def CS_Total(Z, E):
    return _CS_Total(Z, E)


@wrapped_partial(jit, **jit_kwargs)
def _CS_Photo(
    Z: int | NDArray[int], E: float | NDArray[float]
) -> tuple[NDArray[float], bool]:
    """
    Photoelectric absorption cross section  (cm2/g)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        energy (keV)

    Returns
    -------
    Array
        Photoelectric absorption cross section  (cm2/g)
    """
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to CS_PHOTO[Z-1] when _splint is broadcast
    output = xp.where(
        (Z >= 1) & (Z < CS_PHOTO.shape[0]) & (E >= 0.0),
        xp.exp(_splint(CS_PHOTO[Z[0] - 1], xp.log(E * 1000.0))),
        xp.nan,
    )
    return output, xp.isnan(output).any()


@overload
def CS_Photo(Z: int, E: float) -> float:
    ...


@overload
def CS_Photo(Z: NDArray[int], E: NDArray[float]) -> NDArray[float]:
    ...


@output_type
@value_error("# TODO error message")
def CS_Photo(Z, E):
    """_summary_

    Parameters
    ----------
    Z : int | Array
        _description_
    E : float | Array
        _description_

    Returns
    -------
    float | Array
        _description_
    """
    return _CS_Photo(Z, E)


@wrapped_partial(jit, **jit_kwargs)
def _CS_Rayl(
    Z: int | NDArray[int], E: float | NDArray[float]
) -> tuple[NDArray[float], bool]:
    """
    Rayleigh scattering cross section  (cm2/g)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        energy (keV)

    Returns
    -------
    Array
        Rayleigh scattering cross section  (cm2/g)
    """
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to CS_RAYL[Z-1] when _splint is broadcast
    output = xp.where(
        (Z >= 1) & (Z < CS_RAYL.shape[0]) & (E >= 0.0),
        xp.exp(_splint(CS_RAYL[Z[0] - 1], xp.log(E * 1000.0))),
        xp.nan,
    )
    return output, xp.isnan(output).any()


@overload
def CS_Rayl(Z: int, E: float) -> float:
    ...


@overload
def CS_Rayl(Z: NDArray[int], E: NDArray[float]) -> NDArray[float]:
    ...


@output_type
@value_error(" # TODO error message")
def CS_Rayl(Z, E):
    """_summary_

    Parameters
    ----------
    Z : int | Array
        _description_
    E : float | Array
        _description_

    Returns
    -------
    float | Array
        _description_
    """
    return _CS_Rayl(Z, E)


@wrapped_partial(jit, **jit_kwargs)
def _CS_Compt(
    Z: int | NDArray[int], E: float | NDArray[float]
) -> tuple[NDArray[float], bool]:
    """
    Compton scattering cross section  (cm2/g)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        energy (keV)

    Returns
    -------
    Array
        Compton scattering cross section  (cm2/g)
    """
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to CS_COMPT[Z-1] when _splint is broadcast
    output = xp.where(
        (Z >= 1) & (Z < CS_COMPT.shape[0]) & (E >= 0.0),
        xp.exp(_splint(CS_COMPT[Z[0] - 1], xp.log(E * 1000.0))),
        xp.nan,
    )
    return output, xp.isnan(output).any()


@overload
def CS_Compt(Z: int, E: float) -> float:
    ...


@overload
def CS_Compt(Z: NDArray[int], E: NDArray[float]) -> NDArray[float]:
    ...


@output_type
@value_error(" # TODO value error")
def CS_Compt(Z, E):
    # TODO docstring
    """_summary_

    Parameters
    ----------
    Z : int | Array
        _description_
    E : float | Array
        _description_

    Returns
    -------
    float | Array
        _description_
    """
    return _CS_Compt(Z, E)


@wrapped_partial(jit, **jit_kwargs)
def _CS_Energy(
    Z: int | NDArray[int], E: float | NDArray[float]
) -> tuple[NDArray[float], bool]:
    # TODO docstring
    """_summary_

    Parameters
    ----------
    Z : int | NDArray[int]
        _description_
    E : float | NDArray[float]
        _description_

    Returns
    -------
    tuple[NDArray[float], bool]
        _description_
    """
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    # TODO change to CS_ENERGY[Z-1] when broadcast _splint
    output = xp.where(
        (Z >= 1) & (Z < CS_ENERGY.shape[0]) & (E >= 0.0),
        _splint(CS_ENERGY[Z[0] - 1], xp.log(E)),
        xp.nan,
    )
    return xp.exp(output), xp.isnan(output).any()


@overload
def CS_Energy(Z: int, E: float) -> float:
    ...


@overload
def CS_Energy(Z: NDArray[int], E: NDArray[float]) -> NDArray[float]:
    ...


CS_Energy_error = f"""Z out of range: 1 to {CS_ENERGY.shape[0]} |
 Energy must be strictly positive"""


@output_type
@value_error("".join(CS_Energy_error.splitlines()))
def CS_Energy(Z, E):
    """
    Mass energy-absorption coefficient (cm2/g)

    Parameters
    ----------
    Z : int | Array
        atomic number
    E : float | Array
        energy (keV)

    Returns
    -------
    float | Array
        Mass energy-absorption coefficient (cm2/g)

    Raises
    ------
    ValueError
        # TODO finish docstring
        _description_
    """
    return _CS_Energy(Z, E)
