"""_summary_
"""
# TODO docstrings

import functools
from typing import Any, Callable

from jax._src.typing import Array

from . import cross_sections as cross_sections
from . import cs_barns as cs_barns
from . import scattering as scattering
from . import polarized as polarized

from .config import jit
from .constants import AVOGNUM
from .cross_sections import CS_Total, CS_Photo, CS_Rayl, CS_Compt, CS_Energy
from .cs_barns import (
    CSb_Total,
    CSb_Photo,
    CSb_Rayl,
    CSb_Compt,
    DCSb_Rayl,
    DCSb_Compt,
    DCSPb_Rayl,
    DCSPb_Compt,
)
from .scattering import DCS_Rayl, DCS_Compt
from .polarized import DCSP_Rayl, DCSP_Compt
from .xraylib_parser import CompoundParser

# TODO update to use 
# FIXME doesn't work

# CS_Photo_Total, CSb_Photo_Total, CS_Total_Kissel, CSb_Total_Kissel

# TODO decorator that propagates __doc__ and __annotations__ and does
# conversion to taking a compound as an argument
# def convert_to_barns(func: Callable) -> Callable:
#     def decorator(function: Callable) -> Callable:
#         @jit
#         @functools.wraps(function)
#         def wrapper(Z, *args, **kwargs) -> Array:
#             aw = _AtomicWeight(Z)[0]
#             cs = func(Z, *args, **kwargs)
#             return cs * aw / AVOGNUM

#         return wrapper

#     return decorator

# def convert_to_barns(function: Callable) -> Callable:
#     @jit
#     @functools.wraps(function)
#     def wrapper(Z, *args, **kwargs) -> Array:
#         aw = _AtomicWeight(Z)[0]
#         cs = function(Z, *args, **kwargs)
#         return cs * aw / AVOGNUM

#     return wrapper

# TODO jit & functools wraps
# def compound(func: Callable) -> Callable:


# TODO use _function and account for nan
def compound(function: Callable) -> Callable:
    @functools.wraps(function)
    def wrapper(compound: str, *args, **kwargs) -> Array:
        # TODO nist compound database
        try:
            compound_dict = CompoundParser(compound)
        except ValueError:
            # TODO logging or something
            pass
        elements = compound_dict["Elements"]
        mass_fractions = compound_dict["massFractions"]
        return sum(
            function(i, *args, **kwargs) * j
            for i, j in zip(elements, mass_fractions)
        )

    return wrapper


# TODO docstrings


@compound
def CS_Total_CP(compound: str, E: Array) -> Array:
    return CS_Total(compound, E)


@compound
def CS_Photo_CP(compound: str, E: Array) -> Array:
    return CS_Photo(compound, E)


@compound
def CS_Rayl_CP(compound: str, E: Array) -> Array:
    return CS_Rayl(compound, E)


@compound
def CS_Compt_CP(compound: str, E: Array) -> Array:
    return CS_Compt(compound, E)


@compound
def CSb_Total_CP(compound: str, E: Array) -> Array:
    return CSb_Total(compound, E)


@compound
def CSb_Photo_CP(compound: str, E: Array) -> Array:
    return CSb_Photo(compound, E)


@compound
def CSb_Rayl_CP(compound: str, E: Array) -> Array:
    return CSb_Rayl(compound, E)


@compound
def CSb_Compt_CP(compound: str, E: Array) -> Array:
    return CSb_Compt(compound, E)


@compound
def CS_Energy_CP(compound: str, E: Array) -> Array:
    return CS_Energy(compound, E)


@compound
def DCS_Rayl_CP(compound: str, E: Array, theta: Array) -> Array:
    return DCS_Rayl(compound, E, theta)


@compound
def DCS_Compt_CP(compound: str, E: Array, theta: Array) -> Array:
    return DCS_Compt(compound, E, theta)


@compound
def DCSb_Rayl_CP(compound: str, E: Array, theta: Array) -> Array:
    return DCSb_Rayl(compound, E, theta)


@compound
def DCSb_Compt_CP(compound: str, E: Array, theta: Array) -> Array:
    return DCSb_Compt(compound, E, theta)


@compound
def DCSP_Rayl_CP(compound: str, E: Array, theta: Array, phi: Array) -> Array:
    return DCSP_Rayl(compound, E, theta, phi)


@compound
def DCSP_Compt_CP(compound: str, E: Array, theta: Array, phi: Array) -> Array:
    return DCSP_Compt(compound, E, theta, phi)


@compound
def DCSPb_Rayl_CP(compound: str, E: Array, theta: Array, phi: Array) -> Array:
    return DCSPb_Rayl(compound, E, theta, phi)


@compound
def DCSPb_Compt_CP(compound: str, E: Array, theta: Array, phi: Array) -> Array:
    return DCSPb_Compt(compound, E, theta, phi)
