"""_summary_
"""
from __future__ import annotations
import functools
from typing import Callable

from jax._src.typing import Array

from . import atomicweight
from . import cross_sections
from . import polarized
from . import scattering

from .config import jit, jit_kwargs, xp
from .constants import AVOGNUM
from .atomicweight import _AtomicWeight
from .cross_sections import _CS_Total, CS_Photo, CS_Rayl, CS_Compt
from .polarized import DCSP_Compt, _DCSP_Rayl, jit as hola
from .scattering import DCS_Rayl, DCS_Compt
from ._utilities import value_error, wrapped_partial, output_type, xrl_xrlnp


# TODO update to use _func
# FIXME doesn't work

# TODO maybe can jit this? or all individual functions?
# TODO can i do it with one decorator?
# def convert_to_barns(func: Callable) -> Callable:

# TODO make this work with wrapped_partial
# TODO include nan in function output
# @wrapped_partial(jit, **jit_kwargs)
def convert_to_barns(function: Callable) -> Callable:
    # @jit
    # @functools.wraps(function)
    # @wrapped_partial(jit, **jit_kwargs)
    @functools.wraps(function)
    def wrapper(Z, *args, **kwargs) -> Array:
        aw = _AtomicWeight(Z)[0]
        cs, nan = function(Z, *args, **kwargs)
        return cs * aw / AVOGNUM, nan

    return wrapper

    # return decorator


def add_doc(func: Callable) -> Callable:
    def _doc(function: Callable) -> Callable:
        function.__doc__ = (
            func.__doc__.replace("cm2/g", "barns")
            if func.__doc__ is not None
            else None
        )
        function.__annotations__ = func.__annotations__
        return function

    return _doc


def barns(func):
    def decorator(function):
        @add_doc(func)
        @convert_to_barns(func)
        @functools.wraps(function)
        def f(*args, **kwargs):
            return function(*args, **kwargs)

        return f

    return decorator


@wrapped_partial(jit, **jit_kwargs)
@convert_to_barns
def _CSb_Total(Z: Array, E: Array) -> Array:
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


@output_type
@value_error(" # TODO value Error")
def CSb_Total(Z: Array, E: Array) -> Array:
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
    # TODO redo implementation of cross-sections for this
    return _CSb_Total(Z, E)


@convert_to_barns
def CSb_Photo(Z: Array, E: Array) -> Array:
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
    return CS_Photo(Z, E)


@convert_to_barns
def CSb_Rayl(Z: Array, E: Array) -> Array:
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
    return CS_Rayl(Z, E)


@convert_to_barns
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
    return CS_Compt(Z, E)


# @barns(CS_FluorLine)
def CSb_FluorLine(Z, line, E):
    ...


# @barns(CS_FluorShell)
def CSb_FluorShell(Z, shell, E):
    ...


@convert_to_barns
def DCSb_Rayl(Z: Array, E: Array, theta: Array) -> Array:
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
    return DCS_Rayl(Z, E, theta)


@convert_to_barns
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
    return DCS_Compt(Z, E, theta)


@convert_to_barns
@xrl_xrlnp()
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


@convert_to_barns
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
    return DCSP_Compt(Z, E, theta, phi)
