"""_summary_
"""

from ._utilities import asarray, wrapped_partial
from .atomicweight import AtomicWeight as _AtomicWeight
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp
from .constants import HC_4PI, KD
from .cross_sections import CS_Total as _CS_Total
from .fi import Fi as _Fi
from .xraylib_parser import CompoundParser


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
@asarray(argnums=(0,), argnames=("compound"))
def Refractive_Index_Re(compound, E, density):
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    density : float
        _description_

    Returns
    -------
    float
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    compound_dict = CompoundParser(compound)
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
def Refractive_Index_Im(compound, E, density):
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : float
        _description_
    density : float
        _description_

    Returns
    -------
    float
        _description_
    """
    compound_dict = CompoundParser(compound)
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
    elements : _type_
        _description_
    mass_fractions : _type_
        _description_
    E : _type_
        _description_
    density : _type_
        _description_
    """
    rv_real = Refractive_Index_Re(compound, E, density)
    rv_imag = Refractive_Index_Im(compound, E, density)
    # TODO if one is nan then both should be nan # ???
    return rv_real + rv_imag * 1j
