"""_summary_
"""
# TODO cross sections in other units
# TODO redo with xp.where n stuff
# TODO issue with E = 1
from __future__ import annotations
import os

from .config import jit, jit_kwargs, xp, NDArray, ArrayLike
from ._splint import _splint
from ._utilities import wrapped_partial, xrl_xrlnp

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
def _CS_Total(Z: ArrayLike, E: ArrayLike) -> tuple[NDArray, bool]:
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
    tuple[NDArray, bool]
        Total cross section  (cm2/g)
        (Photoelectric + Compton + Rayleigh)
    """
    compton = _CS_Compt(Z, E)[0]
    photo = _CS_Photo(Z, E)[0]
    rayleigh = _CS_Rayl(Z, E)[0]
    output = compton + photo + rayleigh
    return output, xp.isnan(output).any()


@xrl_xrlnp(" # TODO error message")
def CS_Total(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CS_Total(Z, E)


@wrapped_partial(jit, **jit_kwargs)
def _CS_Photo(Z: ArrayLike, E: ArrayLike) -> tuple[NDArray, bool]:
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
    tuple[NDArray, bool]
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


@xrl_xrlnp(" # TODO error message")
def CS_Photo(Z: ArrayLike, E: ArrayLike) -> NDArray:
    # TODO docstring
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CS_Photo(Z, E)


@wrapped_partial(jit, **jit_kwargs)
def _CS_Rayl(Z: ArrayLike, E: ArrayLike) -> tuple[NDArray, bool]:
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
    tuple[NDArray, bool]
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


@xrl_xrlnp(" # TODO error message")
def CS_Rayl(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CS_Rayl(Z, E)


@wrapped_partial(jit, **jit_kwargs)
def _CS_Compt(Z: ArrayLike, E: ArrayLike) -> tuple[NDArray, bool]:
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
    tuple[NDArray, bool]
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


@xrl_xrlnp(" # TODO error message")
def CS_Compt(Z: ArrayLike, E: ArrayLike) -> NDArray:
    # TODO docstring
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CS_Compt(Z, E)


@wrapped_partial(jit, **jit_kwargs)
def _CS_Energy(Z: ArrayLike, E: ArrayLike) -> tuple[NDArray, bool]:
    # TODO docstring
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    tuple[NDArray, bool]
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


CS_Energy_error = f"""Z out of range: 1 to {CS_ENERGY.shape[0]} |
 Energy must be strictly positive"""


@xrl_xrlnp("".join(CS_Energy_error.splitlines()))
def CS_Energy(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Mass energy-absorption coefficient (cm2/g)

    Parameters
    ----------
    Z : ArrayLike
        atomic number
    E : ArrayLike
        energy (keV)

    Returns
    -------
    NDArray
        Mass energy-absorption coefficient (cm2/g)

    Raises
    ------
    ValueError
        # TODO finish docstring
        _description_
    """
    return _CS_Energy(Z, E)
