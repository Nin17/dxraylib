"""_summary_
"""
from __future__ import annotations
import inspect
import functools
from typing import Callable

import jax

from ._utilities import asarray, wrapped_partial
from .atomicweight import AtomicWeight as _AtomicWeight
from .config import ArrayLike, jit, jit_kwargs, NDArray
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
    def wrapper(Z, *args, **kwargs) -> NDArray:
        output = function(Z, *args, **kwargs)
        a_w = inspect.unwrap(_AtomicWeight)(Z).reshape(
            (*Z.shape, *(1,) * (output.ndim - Z.ndim))
        )
        return output * a_w / AVOGNUM

    return wrapper


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def CSb_Total(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Total cross section  (barns)
    (Photoelectric + Compton + Rayleigh)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        energy (keV)

    Returns
    -------
    Array
        Total cross section  (barns)
        (Photoelectric + Compton + Rayleigh)
    """
    return _CS_Total(Z, E)


# @wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def CSb_Photo(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Photoelectric absorption cross section  (barns)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        energy (keV)

    Returns
    -------
    Array
        Photoelectric absorption cross section  (barns)
    """
    return _CS_Photo(Z, E)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def CSb_Rayl(Z: ArrayLike, E: ArrayLike) -> NDArray:
    """
    Rayleigh scattering cross section  (barns)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        energy (keV)

    Returns
    -------
    Array
        Rayleigh scattering cross section  (barns)
    """
    return _CS_Rayl(Z, E)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def CSb_Compt(Z, E):
    """
    Compton scattering cross section  (barns)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        energy (keV)

    Returns
    -------
    Array
        Compton scattering cross section  (barns)
    """
    return _CS_Compt(Z, E)


# @barns(CS_FluorLine)
def CSb_FluorLine(Z, line, E):
    ...


# @barns(CS_FluorShell)
def CSb_FluorShell(Z, shell, E):
    ...


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def DCSb_Rayl(Z: ArrayLike, E: ArrayLike, theta: ArrayLike) -> NDArray:
    """
    Differential Rayleigh scattering cross section (barns/sterad)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        Energy (keV)
    theta : Array
        scattering polar angle (rad)

    Returns
    -------
    Array
        Differential Rayleigh scattering cross section (barns/sterad)
    """
    return _DCS_Rayl(Z, E, theta)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def DCSb_Compt(Z, E, theta):
    """
    Differential Compton scattering cross section (cm2/g/sterad)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        Energy (keV)
    theta : Array
        scattering polar angle (rad)

    Returns
    -------
    Array
        Differential Compton scattering cross section (cm2/g/sterad)
    """
    return _DCS_Compt(Z, E, theta)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def DCSPb_Rayl(Z, E, theta, phi):
    """
    Differential Rayleigh scattering cross section
    for polarized beam (barns/sterad)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        Energy (keV)
    theta : Array
        scattering polar angle (rad)
    phi : Array
        scattering azimuthal angle (rad)

    Returns
    -------
    Array
        Differential Rayleigh scattering cross section
        for polarized beam (barns/sterad)
    """
    return _DCSP_Rayl(Z, E, theta, phi)


@wrapped_partial(jit, **jit_kwargs)
@asarray()
@barns
def DCSPb_Compt(Z, E, theta, phi):
    """
    Differential Compton scattering cross section
    for polarized beam (barns/sterad)

    Parameters
    ----------
    Z : Array
        atomic number
    E : Array
        Energy (keV)
    theta : Array
        scattering polar angle (rad)
    phi : Array
        scattering azimuthal angle (rad)

    Returns
    -------
    Array
        Differential Compton scattering cross section
        for polarized beam (barns/sterad)
    """
    return _DCSP_Compt(Z, E, theta, phi)
