"""Differential scattering cross-sections for polarized beams."""

from __future__ import annotations

__all__ = ["DCSP_KN", "DCSP_Compt", "DCSP_Rayl", "DCSP_Thoms"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace
from xraylib import AVOGNUM, MEC2, RE2

from .atomicweight import AtomicWeight
from .scattering import FF_Rayl, MomentTransf, SF_Compt

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray


def DCSP_Rayl(
    Z: NDArray[integer],
    E: NDArray[floating],
    theta: NDArray[floating],
    phi: NDArray[floating],
) -> NDArray[floating]:
    """Rayleigh differential scattering cross-section for a polarized beam.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray[floating]
        energy (keV)
    theta : NDArray[floating]
        scattering polar angle (rad)
    phi : NDArray[floating]
        scattering azimuthal angle (rad)

    Returns
    -------
    NDArray[floating]
        Rayleigh differential scattering cross-section for a polarized beam
        (cm²/g/sr)

    """
    xp = array_namespace(Z, E, theta, phi)
    dims = (*range(Z.ndim + E.ndim + theta.ndim + phi.ndim),)

    q = MomentTransf(E, theta)
    ff_rayl = xp.expand_dims(FF_Rayl(Z, q), dims[-phi.ndim :])
    a_w = xp.expand_dims(AtomicWeight(Z), dims[Z.ndim :])
    dcsp_thoms = xp.expand_dims(DCSP_Thoms(theta, phi), dims[: Z.ndim + E.ndim])
    return AVOGNUM / a_w * ff_rayl * ff_rayl * dcsp_thoms


def DCSP_Compt(
    Z: NDArray[integer],
    E: NDArray[floating],
    theta: NDArray[floating],
    phi: NDArray[floating],
) -> NDArray[floating]:
    """Compton differential scattering cross-section for a polarized beam.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray[floating]
        energy (keV)
    theta : NDArray[floating]
        scattering polar angle (rad)
    phi : NDArray[floating]
        scattering azimuthal angle (rad)

    Returns
    -------
    NDArray[floating]
        Compton differential scattering cross-section for a polarized beam
        (cm²/g/sr)

    """
    xp = array_namespace(Z, E, theta, phi)
    dims = (*range(Z.ndim + E.ndim + theta.ndim + phi.ndim),)

    q = MomentTransf(E, theta)
    sf_compt = xp.expand_dims(SF_Compt(Z, q), dims[-phi.ndim :])
    a_w = xp.expand_dims(AtomicWeight(Z), dims[Z.ndim :])
    dcsp_kn = xp.expand_dims(DCSP_KN(E, theta, phi), dims[: Z.ndim])

    return AVOGNUM / a_w * sf_compt * dcsp_kn


def DCSP_KN(
    E: NDArray[floating],
    theta: NDArray[floating],
    phi: NDArray[floating],
) -> NDArray[floating]:
    """Klein-Nishina differential scattering cross-section for a polarized beam.

    Parameters
    ----------
    E : NDArray[floating]
        energy (keV)
    theta : NDArray[floating]
        scattering polar angle (rad)
    phi : NDArray[floating]
        scattering azimuthal angle (rad)

    Returns
    -------
    NDArray[floating]
        Klein-Nishina differential scattering cross-section for a polarized beam
        (barn)

    """
    xp = array_namespace(E, theta, phi)
    dims = (*range(E.ndim + theta.ndim + phi.ndim),)

    e = xp.expand_dims(E, dims[E.ndim :])
    costh = xp.expand_dims(xp.cos(theta), dims[: E.ndim] + dims[-phi.ndim :])
    sinth = xp.expand_dims(xp.sin(theta), dims[: E.ndim] + dims[-phi.ndim :])
    cosphi = xp.expand_dims(xp.cos(phi), dims[: -phi.ndim])
    k0_k = xp.where(e > 0, 1.0 + (1.0 - costh) * e / MEC2, xp.nan)
    k_k0 = 1.0 / k0_k
    k_k0_2 = k_k0 * k_k0

    return (RE2 / 2.0) * k_k0_2 * (k_k0 + k0_k - 2 * sinth * sinth * cosphi * cosphi)


def DCSP_Thoms(theta: NDArray[floating], phi: NDArray[floating]) -> NDArray[floating]:
    """Thomson differential scattering cross-section for a polarized beam.

    Parameters
    ----------
    theta : NDArray[floating]
        scattering polar angle (rad)
    phi : NDArray[floating]
        scattering azimuthal angle (rad)

    Returns
    -------
    NDArray[floating]
        Thomson differential scattering cross-section for a polarized beam
        (barn)

    """
    dims = (*range(theta.ndim + phi.ndim),)
    xp = array_namespace(theta, phi)
    sin_theta = xp.expand_dims(xp.sin(theta), dims[theta.ndim :])
    cos_phi = xp.expand_dims(xp.cos(phi), dims[: theta.ndim])
    return RE2 * (1 - sin_theta * sin_theta * cos_phi * cos_phi)
