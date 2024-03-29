"""_summary_
"""
from __future__ import annotations
import inspect
import functools
from typing import Callable

import jax

from ._utilities import asarray, wrapped_partial
from .atomicweight import AtomicWeight as _AtomicWeight
from .config import Array, ArrayLike, jit, jit_kwargs
from .constants import AVOGNUM
from .cross_sections import (
    CS_Compt as _CS_Compt,
    CS_Photo as _CS_Photo,
    CS_Rayl as _CS_Rayl,
    CS_Total as _CS_Total,
)
from .polarized import DCSP_Compt as _DCSP_Compt, DCSP_Rayl as _DCSP_Rayl
from .scattering import DCS_Compt as _DCS_Compt, DCS_Rayl as _DCS_Rayl


def barns(function: Callable) -> Callable:
    """_summary_

    Parameters
    ----------
    function : Callable
        _description_

    Returns
    -------
    Callable
        _description_
    """
    # TODO jax.tree_util.partial
    # function = jax.tree_util.Partial(functools.wraps(function))
    function = functools.update_wrapper(jax.tree_util.Partial(function), function)
    # ??? do i need unwrap here
    @functools.wraps(function)
    def wrapper(Z, *args, **kwargs) -> Array:
        output = function(Z, *args, **kwargs)
        a_w = inspect.unwrap(_AtomicWeight)(Z).reshape(
            (*Z.shape, *(1,) * (output.ndim - Z.ndim))
        )
        return output * a_w / AVOGNUM

    return wrapper


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def CSb_Total(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Total cross-section (barn/atom): Photoelectric + Compton + Rayleigh.

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        Total cross-section (barn/atom): Photoelectric + Compton + Rayleigh
    """
    return _CS_Total(Z, E)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def CSb_Photo(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Photoelectric absorption cross-section (barn/atom).

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        Photoelectric absorption cross-section (barn/atom)
    """
    return _CS_Photo(Z, E)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def CSb_Rayl(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Rayleigh scattering cross-section (barn/atom).

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        Rayleigh scattering cross-section (barn/atom)
    """
    return _CS_Rayl(Z, E)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def CSb_Compt(Z: ArrayLike, E: ArrayLike) -> Array:
    """
    Compton scattering cross-section (barn/atom).

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)

    Returns
    -------
    array
        Compton scattering cross-section (barn/atom)
    """
    return _CS_Compt(Z, E)


# @barns(CS_FluorLine)
def CSb_FluorLine(Z, line, E):
    # TODO CS_FluorLine
    ...


# @barns(CS_FluorShell)
def CSb_FluorShell(Z, shell, E):
    # TODO CS_FluorShell
    ...


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def DCSb_Rayl(Z: ArrayLike, E: ArrayLike, theta: ArrayLike) -> Array:
    """
    Rayleigh differential scattering cross-section (barn/atom/sr).

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)
    theta : array_like
        scattering polar angle (rad)

    Returns
    -------
    array
        Rayleigh differential scattering cross-section (barn/atom/sr)
    """
    return _DCS_Rayl(Z, E, theta)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def DCSb_Compt(Z: ArrayLike, E: ArrayLike, theta: ArrayLike) -> Array:
    """
    Compton differential scattering cross-section (barn/atom/sr).

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        energy (keV)
    theta : array_like
        scattering polar angle (rad)

    Returns
    -------
    array
        Compton differential scattering cross-section (barn/atom/sr)
    """
    return _DCS_Compt(Z, E, theta)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def DCSPb_Rayl(Z: ArrayLike, E: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> Array:
    """
    Rayleigh differential scattering cross-section for a polarized beam
    (barn/atom/sr).

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
        Rayleigh differential scattering cross-section for a polarized beam
        (barn/atom/sr)
    """
    return _DCSP_Rayl(Z, E, theta, phi)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def DCSPb_Compt(Z: ArrayLike, E: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> Array:
    """
    Compton differential scattering cross-section for a polarized beam
    (barn/atom/sr).

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
        (barn/atom/sr)
    """
    return _DCSP_Compt(Z, E, theta, phi)
