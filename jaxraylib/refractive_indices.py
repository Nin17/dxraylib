"""
Refractive indices: real component, imagingary component and complex
"""

import fuckit

from ._utilities import asarray, wrapped_partial
from .atomicweight import AtomicWeight as _AtomicWeight
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp
from .constants import HC_4PI, KD
from .cross_sections import CS_Total as _CS_Total
from .fi import Fi as _Fi
from .xraylib_nist_compounds import GetCompoundDataNISTByName
from .xraylib_parser import CompoundParser


@fuckit
def _compound_data(compound):
    compound_dict = CompoundParser(compound)
    compound_dict = GetCompoundDataNISTByName(compound)
    return compound_dict


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
@asarray(argnums=(0,), argnames=("compound"))
def Refractive_Index_Re(
    compound: str, E: ArrayLike, density: ArrayLike
) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : array_like
        _description_
    density : array_like
        _description_

    Returns
    -------
    array
        _description_
    """
    compound_dict = _compound_data(compound)

    elements = xp.atleast_1d(xp.asarray(compound_dict["Elements"]))
    mass_fractions = xp.atleast_1d(xp.asarray(compound_dict["massFractions"]))
    mass_fractions = mass_fractions.reshape((*elements.shape, *(1,) * E.ndim))
    _fi = _Fi(elements, E)
    _aw = _AtomicWeight(elements).reshape((*elements.shape, *(1,) * E.ndim))

    _rv = (
        (
            mass_fractions
            * KD
            * (elements.reshape((*elements.shape, *(1,) * E.ndim)) + _fi)
            / _aw
            / E
            / E
        )
        .sum(axis=0)
        .reshape((*E.shape, *(1,) * density.ndim))
    )
    _density = density.reshape((*(1,) * E.ndim, *density.shape))
    return xp.where(_density > 0, 1 - _rv * _density, xp.nan)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
@asarray(argnums=(0,), argnames=("compound"))
def Refractive_Index_Im(
    compound: str, E: ArrayLike, density: ArrayLike
) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : array_like
        _description_
    density : array_like
        _description_

    Returns
    -------
    array
        _description_
    """
    compound_dict = _compound_data(compound)
    elements = xp.atleast_1d(xp.asarray(compound_dict["Elements"]))
    mass_fractions = xp.atleast_1d(xp.asarray(compound_dict["massFractions"]))
    mass_fractions = mass_fractions.reshape((*elements.shape, *(1,) * E.ndim))

    _rv = ((_CS_Total(elements, E) * mass_fractions).sum(axis=0) / E).reshape(
        (*E.shape, *(1,) * density.ndim)
    )
    _density = density.reshape((*(1,) * E.ndim, *density.shape))
    return xp.where(_density > 0, _rv * _density * HC_4PI, xp.nan)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
@asarray(argnums=(0,), argnames=("compound"))
def Refractive_Index(
    compound: str,
    E: ArrayLike,
    density: ArrayLike,
) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : array_like
        _description_
    density : array_like
        _description_

    Returns
    -------
    array
        _description_
    """
    rv_real = Refractive_Index_Re(compound, E, density)
    rv_imag = Refractive_Index_Im(compound, E, density)
    # ??? if one is nan then both should be nan # ???
    return rv_real + rv_imag * 1j
