"""Cross-sections for compounds."""

from __future__ import annotations

__all__ = [
    "CS_Compt_CP",
    "CS_Energy_CP",
    "CS_Photo_CP",
    "CS_Rayl_CP",
    "CS_Total_CP",
    "CSb_Compt_CP",
    "CSb_Photo_CP",
    "CSb_Rayl_CP",
    "CSb_Total_CP",
    "DCSP_Compt_CP",
    "DCSP_Rayl_CP",
    "DCSPb_Compt_CP",
    "DCSPb_Rayl_CP",
    "DCS_Compt_CP",
    "DCS_Rayl_CP",
    "DCSb_Compt_CP",
    "DCSb_Rayl_CP",
]

import functools
from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._compounds import _compound_data
from .cross_sections import CS_Compt, CS_Energy, CS_Photo, CS_Rayl, CS_Total
from .cs_barns import (
    CSb_Compt,
    CSb_Photo,
    CSb_Rayl,
    CSb_Total,
    DCSb_Compt,
    DCSb_Rayl,
    DCSPb_Compt,
    DCSPb_Rayl,
)
from .polarized import DCSP_Compt, DCSP_Rayl
from .scattering import DCS_Compt, DCS_Rayl

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from numpy import floating
    from numpy.typing import NDArray


def cp(function: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Wrap cross-section functions to handle compound strings.

    Parameters
    ----------
    function : Callable[..., NDArray]
        function to wrap

    Returns
    -------
    Callable[..., NDArray]
        wrapped function

    """

    @functools.wraps(function)
    def wrapper(compound: str, *args: NDArray, **kwargs: dict[str, NDArray]) -> NDArray:
        xp: ModuleType = array_namespace(*args, *kwargs.values())
        compound_dict = _compound_data(compound)
        elements = xp.asarray(compound_dict["Elements"])
        mass_fractions = xp.asarray(compound_dict["massFractions"])
        output = function(elements, *args, **kwargs)
        mass_fractions = xp.expand_dims(mass_fractions, (*range(1, output.ndim),))
        return xp.sum(output * mass_fractions, axis=0)

    return wrapper


@cp
def CS_Total_CP(compound: str, E: NDArray[floating]) -> NDArray[floating]:
    """Total cross-section: Photoelctric + Compton + Rayleigh.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Total cross-section: Photelectric + Compton + Rayleigh
         (cm²/g)

    """
    return CS_Total(compound, E)


@cp
def CS_Photo_CP(compound: str, E: NDArray[floating]) -> NDArray[floating]:
    """Photoelectric absorption cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Photoelectric absorption cross-section
        (cm²/g)

    """
    return CS_Photo(compound, E)


@cp
def CS_Rayl_CP(compound: str, E: NDArray[floating]) -> NDArray[floating]:
    """Rayleigh scattering cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Rayleigh scattering cross-section
        (cm²/g)

    """
    return CS_Rayl(compound, E)


@cp
def CS_Compt_CP(compound: str, E: NDArray[floating]) -> NDArray[floating]:
    """Compton scattering cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Compton scattering cross-section
        (cm²/g)

    """
    return CS_Compt(compound, E)


@cp
def CSb_Total_CP(compound: str, E: NDArray[floating]) -> NDArray[floating]:
    """Total cross-section -> Photoelectric + Compton + Rayleigh.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Total cross-section -> Photoelectric + Compton + Rayleigh
        (barn/atom)

    """
    return CSb_Total(compound, E)


@cp
def CSb_Photo_CP(compound: str, E: NDArray[floating]) -> NDArray[floating]:
    """Photoelectric absorption cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Photoelectric absorption cross-section
        (barn/atom)

    """
    return CSb_Photo(compound, E)


@cp
def CSb_Rayl_CP(compound: str, E: NDArray[floating]) -> NDArray[floating]:
    """Rayleigh scattering cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Rayleigh scattering cross-section
        (barn/atom)

    """
    return CSb_Rayl(compound, E)


@cp
def CSb_Compt_CP(compound: str, E: NDArray[floating]) -> NDArray[floating]:
    """Compton scattering cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Compton scattering cross-section
        (barn/atom)

    """
    return CSb_Compt(compound, E)


@cp
def CS_Energy_CP(compound: str, E: NDArray[floating]) -> NDArray[floating]:
    """Mass-energy absorption cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)

    Returns
    -------
    NDArray[floating]
        Mass-energy absorption cross-section
        (cm²/g)

    """
    return CS_Energy(compound, E)


@cp
def DCS_Rayl_CP(
    compound: str,
    E: NDArray[floating],
    theta: NDArray[floating],
) -> NDArray[floating]:
    """Rayleigh differential scattering cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
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
    return DCS_Rayl(compound, E, theta)


@cp
def DCS_Compt_CP(
    compound: str,
    E: NDArray[floating],
    theta: NDArray[floating],
) -> NDArray[floating]:
    """Compton differential scattering cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
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
    return DCS_Compt(compound, E, theta)


@cp
def DCSb_Rayl_CP(
    compound: str,
    E: NDArray[floating],
    theta: NDArray[floating],
) -> NDArray[floating]:
    """Rayleigh differential scattering cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
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
    return DCSb_Rayl(compound, E, theta)


@cp
def DCSb_Compt_CP(
    compound: str,
    E: NDArray[floating],
    theta: NDArray[floating],
) -> NDArray[floating]:
    """Compton differential scattering cross-section.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
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
    return DCSb_Compt(compound, E, theta)


@cp
def DCSP_Rayl_CP(
    compound: str,
    E: NDArray[floating],
    theta: NDArray[floating],
    phi: NDArray[floating],
) -> NDArray[floating]:
    """Rayleigh differential scattering cross-section for a polarized beam.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
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
    return DCSP_Rayl(compound, E, theta, phi)


@cp
def DCSP_Compt_CP(
    compound: str,
    E: NDArray[floating],
    theta: NDArray[floating],
    phi: NDArray[floating],
) -> NDArray[floating]:
    """Compton differential scattering cross-section for a polarized beam.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
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
    return DCSP_Compt(compound, E, theta, phi)


@cp
def DCSPb_Rayl_CP(
    compound: str,
    E: NDArray[floating],
    theta: NDArray[floating],
    phi: NDArray[floating],
) -> NDArray[floating]:
    """Rayleigh differential scattering cross-section for a polarized beam.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
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
    return DCSPb_Rayl(compound, E, theta, phi)


@cp
def DCSPb_Compt_CP(
    compound: str,
    E: NDArray[floating],
    theta: NDArray[floating],
    phi: NDArray[floating],
) -> NDArray[floating]:
    """Compton differential scattering cross-section for a polarized beam.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
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
    return DCSPb_Compt(compound, E, theta, phi)


# TODO(nin17): CS_Photo_Total_CP
# TODO(nin17): CSb_Photo_Total_CP
# TODO(nin17): CS_Total_Kissel_CP
# TODO(nin17): CSb_Total_Kissel_CP
