"""_summary_
"""
# TODO docstrings
# TODO sort out functools wraps so that signature works

import functools
from typing import Callable

import fuckit
import jax

from ._utilities import asarray, wrapped_partial
from .config import ArrayLike, jit, jit_kwargs, NDArray, xp
from .cross_sections import (
    CS_Compt as _CS_Compt,
    CS_Energy as _CS_Energy,
    CS_Photo as _CS_Photo,
    CS_Rayl as _CS_Rayl,
    CS_Total as _CS_Total,
)
from .cs_barns import (
    CSb_Compt as _CSb_Compt,
    CSb_Photo as _CSb_Photo,
    CSb_Rayl as _CSb_Rayl,
    CSb_Total as _CSb_Total,
    DCSb_Compt as _DCSb_Compt,
    DCSb_Rayl as _DCSb_Rayl,
    DCSPb_Compt as _DCSPb_Compt,
    DCSPb_Rayl as _DCSPb_Rayl,
)
from .polarized import DCSP_Compt as _DCSP_Compt, DCSP_Rayl as _DCSP_Rayl
from .scattering import DCS_Compt as _DCS_Compt, DCS_Rayl as _DCS_Rayl
from .xraylib_nist_compounds import GetCompoundDataNISTByName
from .xraylib_parser import CompoundParser


@fuckit
def _compound_data(compound):
    compound_dict = CompoundParser(compound)
    compound_dict = GetCompoundDataNISTByName(compound)
    return compound_dict


def cp(function: Callable) -> Callable:
    function = functools.update_wrapper(
        jax.tree_util.Partial(function), function
    )

    @functools.wraps(function)
    def wrapper(compound: str, *args, **kwargs) -> NDArray:
        compound_dict = _compound_data(compound)
        elements = xp.atleast_1d(xp.asarray(compound_dict["Elements"]))
        mass_fractions = xp.atleast_1d(
            xp.asarray(compound_dict["massFractions"])
        )
        output = function(elements, *args, **kwargs)
        mass_fractions = mass_fractions.reshape(
            (*elements.shape, *(1,) * (output.ndim - elements.ndim))
        )
        return (function(elements, *args, **kwargs) * mass_fractions).sum(
            axis=0
        )

    return wrapper


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def CS_Total_CP(compound: str, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """

    return _CS_Total(compound, E)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
@asarray(argnums=(0,), argnames=("compound"))
@cp
def CS_Photo_CP(compound: str, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CS_Photo(compound, E)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def CS_Rayl_CP(compound: str, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CS_Rayl(compound, E)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def CS_Compt_CP(compound: str, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CS_Compt(compound, E)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def CSb_Total_CP(compound: str, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CSb_Total(compound, E)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def CSb_Photo_CP(compound: str, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CSb_Photo(compound, E)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def CSb_Rayl_CP(compound: str, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CSb_Rayl(compound, E)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def CSb_Compt_CP(compound: str, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CSb_Compt(compound, E)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def CS_Energy_CP(compound: str, E: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _CS_Energy(compound, E)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def DCS_Rayl_CP(compound: str, E: ArrayLike, theta: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_
    theta : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _DCS_Rayl(compound, E, theta)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def DCS_Compt_CP(compound: str, E: ArrayLike, theta: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_
    theta : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _DCS_Compt(compound, E, theta)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def DCSb_Rayl_CP(compound: str, E: ArrayLike, theta: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_
    theta : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _DCSb_Rayl(compound, E, theta)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def DCSb_Compt_CP(compound: str, E: ArrayLike, theta: ArrayLike) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_
    theta : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _DCSb_Compt(compound, E, theta)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def DCSP_Rayl_CP(
    compound: str, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_
    theta : ArrayLike
        _description_
    phi : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _DCSP_Rayl(compound, E, theta, phi)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def DCSP_Compt_CP(
    compound: str, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_
    theta : ArrayLike
        _description_
    phi : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _DCSP_Compt(compound, E, theta, phi)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def DCSPb_Rayl_CP(
    compound: str, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_
    theta : ArrayLike
        _description_
    phi : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _DCSPb_Rayl(compound, E, theta, phi)


@wrapped_partial(jit, **(jit_kwargs | {"static_argnums": 0}))
# @asarray(argnums=(0,), argnames=("compound"))
@cp
def DCSPb_Compt_CP(
    compound: str, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> NDArray:
    """_summary_

    Parameters
    ----------
    compound : str
        _description_
    E : ArrayLike
        _description_
    theta : ArrayLike
        _description_
    phi : ArrayLike
        _description_

    Returns
    -------
    NDArray
        _description_
    """
    return _DCSPb_Compt(compound, E, theta, phi)
