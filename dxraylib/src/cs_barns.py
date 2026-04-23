"""Cross-sections (barns/atom)."""

from __future__ import annotations

__all__ = [
    "CSb_Compt",
    # "CSb_FluorShell",
    "CSb_Photo",
    "CSb_Rayl",
    "CSb_Total",
    "DCSPb_Compt",
    "DCSPb_Rayl",
    "DCSb_Compt",
    "DCSb_Rayl",
]

import functools
from typing import TYPE_CHECKING

from array_api_compat import array_namespace
from xraylib import AVOGNUM

from .cross_sections import CS_Compt, CS_Photo, CS_Rayl, CS_Total
from .polarized import DCSP_Compt, DCSP_Rayl
from .scattering import DCS_Compt, DCS_Rayl
from .src.atomicweight import AtomicWeight

# from .cs_line import CS_FluorShell

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import floating, integer
    from numpy.typing import NDArray


def barns(function: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Convert cross-sections from cm^2/g to barns/atom.

    Parameters
    ----------
    function : Callable
        function to be decorated

    Returns
    -------
    Callable
        decorated function

    """

    @functools.wraps(function)
    def wrapper(
        Z: NDArray[integer],
        *args: NDArray,
        **kwargs: dict[str, NDArray],
    ) -> NDArray:
        xp = array_namespace(Z, *args, *kwargs.values())
        output = function(Z, *args, **kwargs)
        a_w = xp.expand_dims(AtomicWeight(Z), (*range(Z.ndim, output.ndim),))
        return output * a_w / AVOGNUM

    return wrapper


@barns
def CSb_Total(Z: NDArray[integer], E: NDArray[floating]) -> NDArray[floating]:
    """Total cross-section: Photoelectric + Compton + Rayleigh.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Total cross-section: Photoelectric + Compton + Rayleigh
        (barn/atom)

    """
    return CS_Total(Z, E)


@barns
def CSb_Photo(Z: NDArray[integer], E: NDArray[floating]) -> NDArray[floating]:
    """Photoelectric absorption cross-section.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Photoelectric absorption cross-section
        (barn/atom)

    """
    return CS_Photo(Z, E)


@barns
def CSb_Rayl(Z: NDArray[integer], E: NDArray[floating]) -> NDArray[floating]:
    """Rayleigh scattering cross-section.

    Parameters
    ----------
    Z : NDArray[integer]
        atomic number
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Rayleigh scattering cross-section
        (barn/atom)

    """
    return CS_Rayl(Z, E)


@barns
def CSb_Compt(Z: NDArray[integer], E: NDArray[floating]) -> NDArray[floating]:
    """Compton scattering cross-section.

    Parameters
    ----------
    Z : NDArray[floating]
        atomic number
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Compton scattering cross-section
        (barn/atom)

    """
    return CS_Compt(Z, E)


# !!!
# @barns
# def CSb_FluorLine(Z, line, E):
#     return _CS_FluorLine(Z, line, E)


# !!!
# @barns
# def CSb_FluorShell(
#     Z: NDArray[integer],
#     shell: NDArray[integer],
#     E: NDArray[floating],
# ) -> NDArray[floating]:
#     # TODO(nin17): docstring
#     return CS_FluorShell(Z, shell, E)


@barns
def DCSb_Rayl(
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
        (barn/atom/sr)

    """
    return DCS_Rayl(Z, E, theta)


@barns
def DCSb_Compt(
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
        (barn/atom/sr)

    """
    return DCS_Compt(Z, E, theta)


@barns
def DCSPb_Rayl(
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
        (barn/atom/sr)

    """
    return DCSP_Rayl(Z, E, theta, phi)


@barns
def DCSPb_Compt(
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
        (barn/atom/sr)

    """
    return DCSP_Compt(Z, E, theta, phi)
