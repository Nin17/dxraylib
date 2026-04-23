"""Refractive indices: real component, imagingary component and complex."""

from __future__ import annotations

__all__: list[str] = ["Refractive_Index", "Refractive_Index_Im", "Refractive_Index_Re"]

from typing import TYPE_CHECKING

from array_api_compat import array_namespace

from ._compounds import _compound_data
from .constants import HC_4PI, KD
from .cross_sections import CS_Total
from .fi import Fi
from .src.atomicweight import AtomicWeight

if TYPE_CHECKING:
    from numpy import complexfloating, floating
    from numpy.typing import NDArray


def Refractive_Index_Re(
    compound: str,
    E: NDArray[floating],
    density: NDArray[floating],
) -> NDArray[floating]:
    """Real component of the refractive index: 1 - 𝞭.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)
    density : NDArray[floating]
        density (g/cm³)

    Returns
    -------
    NDArray[floating]
        real component of the refractive index: 1 - 𝞭

    """
    compound_dict = _compound_data(compound)
    xp = array_namespace(E, density)
    dims = (*range(1 + E.ndim + density.ndim),)
    sd = slice(1, 1 + E.ndim)

    elements = xp.asarray(compound_dict["Elements"])
    mass_fractions = xp.asarray(compound_dict["massFractions"])
    mass_fractions = xp.expand_dims(mass_fractions, dims[sd])
    fi = Fi(elements, E)
    a_w = xp.expand_dims(AtomicWeight(elements), dims[sd])

    elements = xp.expand_dims(elements, dims[sd])

    rv = xp.sum(mass_fractions * KD * (elements + fi) / a_w / E**2, axis=0)
    rv = xp.expand_dims(rv, dims[E.ndim : -1])

    density = xp.expand_dims(density, dims[: E.ndim])
    return xp.where(density > 0, 1 - rv * density, xp.nan)


def Refractive_Index_Im(
    compound: str,
    E: NDArray[floating],
    density: NDArray[floating],
) -> NDArray[floating]:
    """Imaginary component of the refractive index: β.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)
    density : NDArray[floating]
        density (g/cm³)

    Returns
    -------
    NDArray[floating]
        imaginary component of the refractive index: β

    """
    compound_dict = _compound_data(compound)
    xp = array_namespace(E, density)
    dims = (*range(1 + E.ndim + density.ndim),)

    elements = xp.asarray(compound_dict["Elements"])
    mass_fractions = xp.asarray(compound_dict["massFractions"])
    mass_fractions = xp.expand_dims(mass_fractions, dims[1 : 1 + E.ndim])

    rv = xp.sum(CS_Total(elements, E) * mass_fractions, axis=0) / E
    rv = xp.expand_dims(rv, dims[E.ndim : -1])

    density = xp.expand_dims(density, dims[: E.ndim])
    return xp.where(density > 0, rv * density * HC_4PI, xp.nan)


def Refractive_Index(
    compound: str,
    E: NDArray[floating],
    density: NDArray[floating],
) -> NDArray[complexfloating]:
    """Complex refractive index: 1 - 𝞭 + iβ.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound name
    E : NDArray[floating]
        energy (keV)
    density : NDArray[floating]
        density (g/cm³)

    Returns
    -------
    NDArray[complexfloating]
        complex refractive index: 1 - 𝞭 + iβ

    """
    rv_real = Refractive_Index_Re(compound, E, density)
    rv_imag = Refractive_Index_Im(compound, E, density)
    # ??? if one is nan then both should be nan # ???
    return rv_real + rv_imag * 1j
