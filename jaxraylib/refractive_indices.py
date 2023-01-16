"""_summary_
"""

from .config import xp, ArrayLike, NDArray
from .constants import HC_4PI, KD
from .atomicweight import _AtomicWeight
from .config import jit
from .fi import _Fi
from .cross_sections import _CS_Total
from .xraylib_parser import CompoundParser
from ._utilities import xrl_xrlnp

# TODO decorator that does this stuff


@jit
def _Refractive_Index_Re(elements, mass_fractions, E, density):
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

    Returns
    -------
    _type_
        _description_
    """
    # TODO numpy implementation of this
    rv = 0.0
    for i, j in zip(elements, mass_fractions):
        fi = _Fi(i, E)[0]
        aw = _AtomicWeight(i)[0]
        rv += j * KD * (i + fi) / aw / E / E
    output = xp.where(density > 0, 1 - rv * density, xp.nan)
    return output, xp.isnan(output).any()


@xrl_xrlnp("# TODO error message")
def Refractive_Index_Re(compound: str, E: float, density: float) -> float:
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
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    if any(i > 103 for i in elements) or any(i < 1 for i in elements):
        raise ValueError("Z out of range: 1 to 103.")
    return _Refractive_Index_Re(elements, mass_fractions, E, density)


@jit
def _Refractive_Index_Im(elements, mass_fractions, E, density):
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

    Returns
    -------
    _type_
        _description_
    """
    rv = 0.0
    for i, j in zip(elements, mass_fractions):
        xs = _CS_Total(i, E)[0]
        rv += xs * j
    output = xp.where(density > 0, rv * density * HC_4PI / E, xp.nan)
    return output, xp.isnan(output).any()


@xrl_xrlnp("# TODO error message")
def Refractive_Index_Im(compound: str, E: float, density: float) -> float:
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
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    # TODO check what this should be
    if any(i > 103 for i in elements) or any(i < 1 for i in elements):
        raise ValueError("Z out of range: 1 to 103.")
    return _Refractive_Index_Im(elements, mass_fractions, E, density)


@jit
def _Refractive_Index(
    elements: ArrayLike,
    mass_fractions: ArrayLike,
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
    rv_real = 0.0
    rv_imag = 0.0
    for i, j in zip(elements, mass_fractions):
        fi = _Fi(i, E)[0]
        aw = _AtomicWeight(i)[0]
        rv_real += j * KD * (i + fi) / aw / E / E
        xs = _CS_Total(i, E)[0]
        rv_imag += xs * j
    output = xp.where(
        density > 0,
        1 - rv_real * density + (rv_imag * density * HC_4PI / E) * 1j,
        xp.nan,
    )
    return output, xp.isnan(output).any()


@xrl_xrlnp("# TODO error message")
def Refractive_Index(compound: str, E: float, density: float) -> complex:
    compound_dict = CompoundParser(compound)
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    if any(i > 103 for i in elements) or any(i < 1 for i in elements):
        raise ValueError("Z out of range: 1 to 103.")
    return _Refractive_Index(elements, mass_fractions, E, density)
