"""
Differential scattering cross-sections for polarized beams
"""

from __future__ import annotations

from ._utilities import asarray, wrapped_partial
from .atomicweight import AtomicWeight as _AtomicWeight
from .config import Array, ArrayLike, jit, jit_kwargs, xp
from .constants import AVOGNUM, MEC2, RE2
from .scattering import (
    MomentTransf as _MomentTransf,
    FF_Rayl as _FF_Rayl,
    SF_Compt as _SF_Compt,
)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def DCSP_Rayl(
    Z: ArrayLike, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> Array:
    """
    Rayleigh differential scattering cross-section for a polarized beam
    (cm²/g/sr).

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)
    theta : array_like
        scattering polar angle (rad)
    phi : array
        scattering azimuthal angle (rad)

    Returns
    -------
    array
        Rayleigh differential scattering cross-section for a polarized beam
        (cm²/g/sr)
    """
    q = _MomentTransf(E, theta)
    ff_rayl = _FF_Rayl(Z, q).reshape(
        (*Z.shape, *E.shape, *theta.shape, *(1,) * phi.ndim)
    )
    a_w = _AtomicWeight(Z).reshape(
        (*Z.shape, *(1,) * (E.ndim + theta.ndim + phi.ndim))
    )
    dcsp_thoms = DCSP_Thoms(theta, phi).reshape(
        (*(1,) * (Z.ndim + E.ndim), *theta.shape, *phi.shape)
    )
    return AVOGNUM / a_w * ff_rayl * ff_rayl * dcsp_thoms


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def DCSP_Compt(
    Z: ArrayLike, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> Array:
    """
    Compton differential scattering cross-section for a polarized beam
    (cm²/g/sr).

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)
    theta : array_like
        scattering polar angle (rad)
    phi : array_like
        scattering azimuthal angle (rad)

    Returns
    -------
    array
        Compton differential scattering cross-section for a polarized beam
        (cm²/g/sr)
    """
    q = _MomentTransf(E, theta)
    sf_compt = _SF_Compt(Z, q).reshape(
        (*Z.shape, *E.shape, *theta.shape, *(1,) * phi.ndim)
    )
    a_w = _AtomicWeight(Z).reshape(
        (*Z.shape, *(1,) * (E.ndim + theta.ndim + phi.ndim))
    )

    dcsp_kn = DCSP_KN(E, theta, phi).reshape(
        (*(1,) * Z.ndim, *E.shape, *theta.shape, *phi.shape)
    )
    return AVOGNUM / a_w * sf_compt * dcsp_kn


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def DCSP_KN(E: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> Array:
    """
    Klein-Nishina differential scattering cross-section for a polarized beam
    (barn).

    Parameters
    ----------
    E : array_like
        energy (keV)
    theta : array_like
        scattering polar angle (rad)
    phi : array_like
        scattering azimuthal angle (rad)

    Returns
    -------
    array
        Klein-Nishina differential scattering cross-section for a polarized
        beam (barn)
    """
    e = E.reshape((*E.shape, *(1,) * (theta.ndim + phi.ndim)))
    _theta = theta.reshape(
        (
            *(1,) * E.ndim,
            *theta.shape,
            *(1,) * phi.ndim,
        )
    )
    _phi = phi.reshape((*(1,) * (E.ndim + theta.ndim), *phi.shape))
    cos_th = xp.cos(_theta)
    sin_th = xp.sin(_theta)
    cos_phi = xp.cos(_phi)
    k0_k = xp.where(e > 0, 1.0 + (1.0 - cos_th) * e / MEC2, xp.nan)
    k_k0 = 1.0 / k0_k
    k_k0_2 = k_k0 * k_k0

    output = (
        (RE2 / 2.0)
        * k_k0_2
        * (k_k0 + k0_k - 2 * sin_th * sin_th * cos_phi * cos_phi)
    )
    return output


@wrapped_partial(jit, **jit_kwargs)
@asarray()
def DCSP_Thoms(theta: ArrayLike, phi: ArrayLike) -> Array:
    """
    Thomson differential scattering cross-section for a polarized beam (barn).

    Parameters
    ----------
    theta : array_like
        scattering polar angle (rad)
    phi : array_like
        scattering azimuthal angle (rad)

    Returns
    -------
    array
        Thomson differential scattering cross-section for a polarized beam
        (barn)
    """
    sin_th = xp.sin(theta.reshape((*theta.shape, *(1,) * phi.ndim)))
    cos_phi = xp.cos(phi.reshape((*(1,) * theta.ndim, *phi.shape)))
    return RE2 * (1 - sin_th * sin_th * cos_phi * cos_phi)
