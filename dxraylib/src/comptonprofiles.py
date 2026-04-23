"""Compton scattering profile and subshell Compton scattering profile."""

from __future__ import annotations

__all__ = ["ComptonProfile", "ComptonProfile_Partial"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._interpolate import interpolate1d, interpolate2d
from ._load import _load

if TYPE_CHECKING:
    from numpy import floating, integer
    from numpy.typing import NDArray

COMPTONPROFILE_DATA = _load("compton_profiles")
COMPTONPROFILEPARTIAL_DATA = _load("compton_profiles_partial")


def ComptonProfile(Z: NDArray[integer], pz: NDArray[floating]) -> NDArray[floating]:
    """Compton scattering profile.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    pz : NDArray[floating]
        momentum

    Returns
    -------
    NDArray
        Compton scattering profile

    """
    xp = array_namespace(Z, pz)
    data = xp.asarray(COMPTONPROFILE_DATA)
    return xp.exp(interpolate1d(data, Z, pz, xp.log(pz + 1), xp=xp))


def ComptonProfile_Partial(
    Z: NDArray[integer],
    shell: NDArray[integer],
    pz: NDArray[floating],
) -> NDArray[floating]:
    """Subshell Compton scattering profile.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    shell : NDArray[integer]
        shell macro
    pz : NDArray[floating]
        momentum

    Returns
    -------
    NDArray[floating]
        subshell Compton scattering profile

    """
    xp = array_namespace(Z, shell, pz)
    data = xp.asarray(COMPTONPROFILEPARTIAL_DATA)
    return xp.exp(interpolate2d(data, Z, shell, pz, xp.log(pz + 1), xp=xp))
