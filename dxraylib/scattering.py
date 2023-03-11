"""_summary_
"""
# TODO sensible variable names
from __future__ import annotations
import os

from ._interpolators import _interpolate
from ._utilities import asarray, wrapped_partial
from .atomicweight import AtomicWeight as _AtomicWeight
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp
from .constants import AVOGNUM, KEV2ANGST, MEC2, PI, RE2

_DIRPATH = os.path.dirname(__file__)

_FF_RAYL_PATH = os.path.join(_DIRPATH, "data/FF.npy")
_FF_RAYL = xp.load(_FF_RAYL_PATH)

_SF_COMPT_PATH = os.path.join(_DIRPATH, "data/SF.npy")
_SF_COMPT = xp.load(_SF_COMPT_PATH)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def FF_Rayl(Z: ArrayLike, q: ArrayLike) -> NDArray:
    """
    Atomic form factor for Rayleigh scattering

    Parameters
    ----------
    Z : array_like
        atomic number
    q : array_like
        momentum transfer (Å⁻¹)

    Returns
    -------
    array
        Atomic form factor for Rayleigh scattering
    """
    return _interpolate(_FF_RAYL, Z, q, q)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def SF_Compt(Z: ArrayLike, q: ArrayLike) -> NDArray:
    """
    Incoherent scattering function for Compton scattering

    Parameters
    ----------
    Z : array_like
        atomic number
    q : array_like
        momentum transfer  (Å⁻¹)

    Returns
    -------
    array
        Incoherent scattering function for Compton scattering
    """
    return _interpolate(_SF_COMPT, Z, q, q)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def DCS_Thoms(theta: ArrayLike) -> NDArray:
    """
    Thomson differential scattering cross section (barn)

    Parameters
    ----------
    theta : array_like
        scattering polar angle (rad)

    Returns
    -------
    array
        Thomson differential scattering cross section (barn)
    """
    cos_theta = xp.cos(theta)
    return (RE2 / 2.0) * (1.0 + cos_theta * cos_theta)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def DCS_KN(E: ArrayLike, theta: ArrayLike) -> NDArray:
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
    array
        Klein-Nishina differential scattering cross section (barn)
    """
    e = E.reshape((*E.shape, *(1,) * theta.ndim))
    cos_theta = xp.cos(theta).reshape((*(1,) * E.ndim, *theta.shape))
    t_1 = xp.where(e > 0, (1.0 - cos_theta) * e / MEC2, xp.nan)
    t_2 = 1.0 + t_1
    output = (
        (RE2 / 2.0)
        * (1.0 + cos_theta * cos_theta + t_1 * t_1 / t_2)
        / t_2
        / t_2
    )
    return output


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def DCS_Rayl(Z: ArrayLike, E: ArrayLike, theta: ArrayLike) -> NDArray:
    """
    Differential Rayleigh scattering cross section (cm2/g/sterad)

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
    array
        Differential Rayleigh scattering cross section (cm2/g/sterad)
    """
    q = MomentTransf(E, theta)
    ff_rayl = FF_Rayl(Z, q)
    a_w = _AtomicWeight(Z).reshape((*Z.shape, *(1,) * (E.ndim + theta.ndim)))
    dcs_thoms = DCS_Thoms(theta).reshape(
        (*(1,) * (Z.ndim + E.ndim), *theta.shape)
    )
    return AVOGNUM / a_w * ff_rayl * ff_rayl * dcs_thoms


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def DCS_Compt(Z: ArrayLike, E: ArrayLike, theta: ArrayLike) -> NDArray:
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
    array
        Differential Compton scattering cross section (cm2/g/sterad)
    """
    q = MomentTransf(E, theta)
    sf_compt = SF_Compt(Z, q)
    a_w = _AtomicWeight(Z).reshape((*Z.shape, *(1,) * (E.ndim + theta.ndim)))
    dcs_kn = DCS_KN(E, theta).reshape((*(1,) * Z.ndim, *E.shape, *theta.shape))
    return AVOGNUM / a_w * sf_compt * dcs_kn


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def MomentTransf(E: ArrayLike, theta: ArrayLike) -> NDArray:
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
    array
        Momentum transfer for X-ray photon scattering (Å⁻¹)
    """
    e = E.reshape((*E.shape, *(1,) * theta.ndim))
    sin_theta = xp.sin(theta / 2.0).reshape((*(1,) * E.ndim, *theta.shape))
    return xp.where(e > 0.0, e / KEV2ANGST * sin_theta, xp.nan)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def CS_KN(E: ArrayLike) -> NDArray:
    """
    Total klein-Nishina cross section (barn)

    Parameters
    ----------
    E : array_like
        Energy  (keV)

    Returns
    -------
    array
        Total klein-Nishina cross section (barn)

    """
    a = xp.where(E > 0.0, E / MEC2, xp.nan)
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
    return output


@wrapped_partial(jit, **jit_kwargs)
@asarray()
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
    array
        Photon energy after Compton scattering (keV)

    """
    _e0 = E0.reshape((*E0.shape, *(1,) * theta.ndim))
    cos_theta = xp.cos(theta).reshape((*(1,) * E0.ndim, *theta.shape))
    alpha = xp.where(_e0 > 0.0, _e0 / MEC2, xp.nan)
    return _e0 / (1 + alpha * (1 - cos_theta))
