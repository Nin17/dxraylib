"""Scattering cross-sections."""

from __future__ import annotations

__all__ = [
    "CS_KN",
    "DCS_KN",
    "ComptonEnergy",
    "DCS_Compt",
    "DCS_Rayl",
    "DCS_Thoms",
    "FF_Rayl",
    "MomentTransf",
    "SF_Compt",
]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._interpolate import interpolate1d
from ._load import _load
from .atomicweight import AtomicWeight
from .constants import AVOGNUM, KEV2ANGST, MEC2, PI, RE2  # ??? import from xraylib

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray

FFRAYL_DATA = _load("FF")
SFCOMPT_DATA = _load("SF")


# TODO fix small values ~finfo(float64) & test
def FF_Rayl(Z: NDArray[integer], q: NDArray[floating]) -> NDArray[floating]:
    """Atomic form factor for Rayleigh scattering.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    q : NDArray[floating]
        momentum transfer (Å⁻¹)

    Returns
    -------
    NDArray[floating]
        atomic form factor for Rayleigh scattering

    """
    xp = array_namespace(Z, q)
    data = xp.asarray(FFRAYL_DATA)
    return interpolate1d(data, Z, q, q, xp=xp)


def SF_Compt(Z: NDArray[integer], q: NDArray[floating]) -> NDArray[floating]:
    """Incoherent scattering function for Compton scattering.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    q : NDArray[floating]
        momentum transfer  (Å⁻¹)

    Returns
    -------
    NDArray[floating]
        incoherent scattering function for Compton scattering

    """
    xp = array_namespace(Z, q)
    data = xp.asarray(SFCOMPT_DATA)
    return interpolate1d(data, Z, q, q, xp=xp)


def DCS_Thoms(theta: NDArray[floating]) -> NDArray[floating]:
    """Thomson differential scattering cross-section.

    Parameters
    ----------
    theta : NDArray[floating]
        scattering polar angle (rad)

    Returns
    -------
    NDArray[floating]
        Thomson differential scattering cross-section
        (barn)

    """
    xp = array_namespace(theta)
    cos_theta = xp.cos(theta)
    return (RE2 / 2.0) * (1.0 + cos_theta * cos_theta)


def DCS_KN(E: NDArray[floating], theta: NDArray[floating]) -> NDArray[floating]:
    """Klein-Nishina differential scattering cross-section.

    Parameters
    ----------
    E : NDArray[floating]
        Energy (keV)
    theta : NDArray[floating]
        scattering polar angle (rad)

    Returns
    -------
    NDArray[floating]
        Klein-Nishina differential scattering cross-section
        (barn)

    """
    xp = array_namespace(E, theta)
    dims = (*range(E.ndim + theta.ndim),)
    e = xp.expand_dims(E, dims[E.ndim :])
    cos_theta = xp.expand_dims(xp.cos(theta), dims[: E.ndim])
    t_1 = xp.where(e > 0, (1.0 - cos_theta) * e / MEC2, xp.nan)
    t_2 = 1.0 + t_1
    return (RE2 / 2.0) * (1.0 + cos_theta * cos_theta + t_1 * t_1 / t_2) / t_2 / t_2


def DCS_Rayl(
    Z: NDArray[integer],
    E: NDArray[floating],
    theta: NDArray[floating],
) -> NDArray[floating]:
    """Rayleigh differential scattering cross-section.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray[floating]
        energy (keV)
    theta : NDArray[floating]
        scattering polar angle (rad)

    Returns
    -------
    NDArray[floating]
        Rayleigh differential scattering cross-section
        (cm²/g/sr)

    """
    xp = array_namespace(Z, E, theta)
    dims = (*range(Z.ndim + E.ndim + theta.ndim),)
    q = MomentTransf(E, theta)
    ff_rayl2 = FF_Rayl(Z, q) ** 2
    a_w = xp.expand_dims(AtomicWeight(Z), dims[Z.ndim :])
    dcs_thoms = xp.expand_dims(DCS_Thoms(theta), dims[: -theta.ndim])
    return AVOGNUM / a_w * ff_rayl2 * dcs_thoms


def DCS_Compt(
    Z: NDArray[integer],
    E: NDArray[floating],
    theta: NDArray[floating],
) -> NDArray[floating]:
    """Compton differential scattering cross-section.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray[floating]
        energy (keV)
    theta : NDArray[floating]
        scattering polar angle (rad)

    Returns
    -------
    NDArray[floating]
        Compton differential scattering cross-section
        (cm²/g/sr)

    """
    xp = array_namespace(Z, E, theta)
    dims = (*range(Z.ndim + E.ndim + theta.ndim),)
    q = MomentTransf(E, theta)
    sf_compt = SF_Compt(Z, q)
    a_w = xp.expand_dims(AtomicWeight(Z), dims[Z.ndim :])
    dcs_kn = xp.expand_dims(DCS_KN(E, theta), dims[: Z.ndim])
    return AVOGNUM / a_w * sf_compt * dcs_kn


def MomentTransf(E: NDArray[floating], theta: NDArray[floating]) -> NDArray[floating]:
    """Momentum transfer for X-ray photon scattering.

    Parameters
    ----------
    E : NDArray[floating]
        energy (keV)
    theta : NDArray[floating]
        scattering polar angle (rad)

    Returns
    -------
    NDArray[floating]
        momentum transfer for X-ray photon scattering
        (Å⁻¹)

    """
    xp = array_namespace(E, theta)
    dims = (*range(E.ndim + theta.ndim),)
    e = xp.expand_dims(E, dims[E.ndim :])
    sin_theta = xp.expand_dims(xp.sin(theta / 2.0), dims[: E.ndim])
    return xp.where(e > 0.0, e / KEV2ANGST * sin_theta, xp.nan)


def CS_KN(E: NDArray[floating]) -> NDArray[floating]:
    """Total Klein-Nishina cross-section.

    Parameters
    ----------
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        total Klein-Nishina cross-section
        (barn)

    """
    xp = array_namespace(E)
    a = xp.where(E > 0.0, E / MEC2, xp.nan)
    a3 = a* a *a
    b = 1.0 + 2.0 * a
    b2 = b * b
    lb = xp.log(b)
    c: float = 2 * PI * RE2
    return c * (
        (1 + a) / a3 * (2 * a * (1 + a) / b - lb) + 0.5 * lb / a - (1 + 3 * a) / b2
    )


def ComptonEnergy(E0: NDArray[floating], theta: NDArray[floating]) -> NDArray[floating]:
    """Photon energy after Compton scattering.

    Parameters
    ----------
    E0 : NDArray[floating]
        photon energy before scattering (keV)
    theta : NDArray[floating]
        scattering polar angle (rad)

    Returns
    -------
    NDArray[floating]
        photon energy after Compton scattering
        (keV)

    """
    xp = array_namespace(E0, theta)
    dims = (*range(E0.ndim + theta.ndim),)
    e0 = xp.expand_dims(E0, dims[E0.ndim :])
    cos_theta = xp.expand_dims(xp.cos(theta), dims[: E0.ndim])
    alpha = xp.where(e0 > 0.0, e0 / MEC2, xp.nan)
    return e0 / (1 + alpha * (1 - cos_theta))
