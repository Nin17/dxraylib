"""_summary_
"""
from __future__ import annotations
import os

from .atomicweight import _AtomicWeight
from .config import jit, jit_kwargs, xp, NDArray, ArrayLike
from .constants import AVOGNUM, KEV2ANGST, MEC2,  PI, RE2
from ._splint import _splint
from ._utilities import wrapped_partial, xrl_xrlnp

DIRPATH = os.path.dirname(__file__)

FF_RAYL_PATH = os.path.join(DIRPATH, "data/FF.npy")
FF_RAYL = xp.load(FF_RAYL_PATH)

SF_COMPT_PATH = os.path.join(DIRPATH, "data/SF.npy")
SF_COMPT = xp.load(SF_COMPT_PATH)


@wrapped_partial(jit, **jit_kwargs)
def _FF_Rayl(Z: ArrayLike, q: ArrayLike) -> tuple[NDArray[float], bool]:
    Z = xp.atleast_1d(xp.asarray(Z))
    q = xp.asarray(q)
    # TODO change to FF_RAYL[Z-1] when broadcast splint
    output = xp.where(
        (Z >= 1) & (Z <= FF_RAYL.shape[0]) & (q > 0),
        _splint(FF_RAYL[Z[0] - 1], q),
        xp.nan,
    )
    return output, xp.isnan(output).any()


# TODO docstring raise section
@xrl_xrlnp(f"Z out of range: 1 to {FF_RAYL.shape[0]} | q must be positive")
def FF_Rayl(Z: ArrayLike, q: ArrayLike) -> NDArray[float]:
    """
    Atomic form factor for Rayleigh scattering

    Parameters
    ----------
    Z : array_like
        atomic number
    q : array_like
        momentum transfer  (Å⁻¹)
        https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions#scattering-factors

    Returns
    -------
    Array
        Atomic form factor for Rayleigh scattering
    """
    return _FF_Rayl(Z, q)


@wrapped_partial(jit, **jit_kwargs)
def _SF_Compt(Z: ArrayLike, q: ArrayLike) -> tuple[NDArray, bool]:
    Z = xp.atleast_1d(xp.asarray(Z))
    q = xp.asarray(q)
    # TODO change to SF_COMPT[Z-1] when broadcast splint
    output = xp.where(
        (Z >= 1) & (Z <= SF_COMPT.shape[0]) & (q > 0),
        _splint(SF_COMPT[Z[0] - 1], q),
        xp.nan,
    )
    # TODO conditions on q
    return output, xp.isnan(output).any()


# TODO docstring raise section
@xrl_xrlnp(f"Z out of range: 1 to {SF_COMPT.shape[0]} | q must be positive")
def SF_Compt(Z: ArrayLike, q: ArrayLike) -> NDArray[float]:
    """
    Incoherent scattering function for Compton scattering

    Parameters
    ----------
    Z : array_like
        atomic number
    q : array_like
        momentum transfer  (Å⁻¹)
        https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions#scattering-factors

    Returns
    -------
    Array
        Incoherent scattering function for Compton scattering
    """
    return _SF_Compt(Z, q)


@wrapped_partial(jit, **jit_kwargs)
def _DCS_Thoms(theta: ArrayLike) -> NDArray[float]:
    theta = xp.asarray(theta)
    cos_theta = xp.cos(theta)
    return (RE2 / 2.0) * (1.0 + cos_theta * cos_theta), None


@xrl_xrlnp()
def DCS_Thoms(theta: ArrayLike) -> NDArray:
    """
    Thomson differential scattering cross section (barn)

    Parameters
    ----------
    theta : array_like
        scattering polar angle (rad)

    Returns
    -------
    Array
        Thomson differential scattering cross section (barn)
    """
    return _DCS_Thoms(theta)


@wrapped_partial(jit, **jit_kwargs)
def _DCS_KN(E: ArrayLike, theta: ArrayLike) -> tuple[NDArray[float], bool]:
    E = xp.atleast_1d(xp.asarray(E))
    theta = xp.atleast_1d(xp.asarray(theta))
    cos_theta = xp.cos(theta)
    t1 = xp.where(E > 0, (1.0 - cos_theta) * E / MEC2, xp.nan)
    t2 = 1.0 + t1
    output = (
        (RE2 / 2.0) * (1.0 + cos_theta * cos_theta + t1 * t1 / t2) / t2 / t2
    )
    return output, xp.isnan(output).any()


@xrl_xrlnp("Energy must be strictly positive")
def DCS_KN(E: ArrayLike, theta: ArrayLike) -> NDArray[float]:
    """
    Klein-Nishina differential scattering cross section (barn)

    Parameters
    ----------
    E : array_like
        Energy (keV)
    theta : array_like
        scattering polar angle (rad)

    Returns
    -------
    Array
        Klein-Nishina differential scattering cross section (barn)

    Raises
    ------
    ValueError
        _description_
    """
    return _DCS_KN(E, theta)


@wrapped_partial(jit, **jit_kwargs)
def _DCS_Rayl(
    Z: ArrayLike, E: ArrayLike, theta: ArrayLike
) -> tuple[NDArray, bool]:
    Z = xp.asarray(Z)
    E = xp.asarray(E)
    theta = xp.asarray(theta)
    q = _MomentTransf(E, theta)[0]
    F = _FF_Rayl(Z, q)[0]
    output = AVOGNUM / _AtomicWeight(Z)[0] * F * F * _DCS_Thoms(theta)[0]
    return output, xp.isnan(output).any()


# TODO raise section
@xrl_xrlnp("# TODO error message")
def DCS_Rayl(Z: ArrayLike, E: ArrayLike, theta: ArrayLike) -> NDArray[float]:
    """
    Differential Rayleigh scattering cross section (cm2/g/sterad)

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        Energy (keV)
    theta : Array
        scattering polar angle (rad)

    Returns
    -------
    Array
        Differential Rayleigh scattering cross section (cm2/g/sterad)
    """
    return _DCS_Rayl(Z, E, theta)


@wrapped_partial(jit, **jit_kwargs)
def _DCS_Compt(
    Z: ArrayLike, E: ArrayLike, theta: ArrayLike
) -> tuple[NDArray[float], bool]:
    Z = xp.asarray(Z)
    E = xp.asarray(E)
    theta = xp.asarray(theta)
    q = _MomentTransf(E, theta)[0]
    S = _SF_Compt(Z, q)[0]
    output = AVOGNUM / _AtomicWeight(Z)[0] * S * _DCS_KN(E, theta)[0]
    return output, xp.isnan(output).any()


# TODO raise section
# TODO value_error after updating _splint
@xrl_xrlnp("# TODO error message")
def DCS_Compt(Z: ArrayLike, E: ArrayLike, theta: ArrayLike) -> NDArray[float]:
    """
    Differential Compton scattering cross section (cm2/g/sterad)

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        Energy (keV)
    theta : array_like
        scattering polar angle (rad)

    Returns
    -------
    Array
        Differential Compton scattering cross section (cm2/g/sterad)
    """
    return _DCS_Compt(Z, E, theta)


@wrapped_partial(jit, **jit_kwargs)
def _MomentTransf(
    E: ArrayLike, theta: ArrayLike
) -> tuple[NDArray[float], bool]:
    E = xp.atleast_1d(xp.asarray(E))
    theta = xp.atleast_1d(xp.asarray(theta))
    output = xp.where(E > 0, E / KEV2ANGST * xp.sin(theta / 2.0), xp.nan)
    return output, xp.isnan(output).any()


@xrl_xrlnp("Energy must be strictly positive")
def MomentTransf(E: ArrayLike, theta: ArrayLike) -> NDArray[float]:
    """
    Momentum transfer for X-ray photon scattering (Å⁻¹)

    Parameters
    ----------
    E : array_like
        Energy (keV)
    theta : array_like
        scattering polar angle (rad)

    Returns
    -------
    Array
        Momentum transfer for X-ray photon scattering (Å⁻¹)

    Raises
    ------
    ValueError
        If energy is not strictly positive
    """
    return _MomentTransf(E, theta)


@wrapped_partial(jit, **jit_kwargs)
def _CS_KN(E: ArrayLike) -> tuple[NDArray[float], bool]:
    E = xp.atleast_1d(xp.asarray(E))
    a = xp.where(E > 0, E / MEC2, xp.nan)
    a3 = a * a * a
    b = 1 + 2 * a
    b2 = b * b
    lb = xp.log(b)
    output = (
        2
        * PI
        * RE2
        * (
            (1 + a) / a3 * (2 * a * (1 + a) / b - lb)
            + 0.5 * lb / a
            - (1 + 3 * a) / b2
        )
    )
    return output, xp.isnan(output).any()


@xrl_xrlnp("Energy must be strictly positive")
def CS_KN(E: ArrayLike) -> NDArray[float]:
    """
    Total klein-Nishina cross section (barn)

    Parameters
    ----------
    E : array_like
        Energy  (keV)

    Returns
    -------
    Array
        Total klein-Nishina cross section (barn)

    Raises
    ------
    ValueError
        If energy is not strictly positive
    """
    return _CS_KN(E)


@wrapped_partial(jit, **jit_kwargs)
def _ComptonEnergy(E0: ArrayLike, theta: ArrayLike) -> tuple[NDArray, bool]:
    E0 = xp.atleast_1d(xp.asarray(E0))
    theta = xp.atleast_1d(xp.asarray(theta))
    cos_theta = xp.cos(theta)
    alpha = xp.where(E0 > 0, E0 / MEC2, xp.nan)
    output = E0 / (1 + alpha * (1 - cos_theta))
    return output, xp.isnan(output).any()


@xrl_xrlnp("Energy must be strictly positive")
def ComptonEnergy(E0: ArrayLike, theta: ArrayLike) -> NDArray[float]:
    """
    Photon energy after Compton scattering (keV)

    Parameters
    ----------
    E0 : array_like
        Photon Energy before scattering (keV)
    theta : array_like
        scattering polar angle (rad)

    Returns
    -------
    Array
        Photon energy after Compton scattering (keV)

    Raises
    ------
    ValueError
        If energy is not strictly positive
    """

    return _ComptonEnergy(E0, theta)
