"""_summary_
"""

from .anomalous_scattering import Fi
from .atomic_weight import AtomicWeight
from .config import jit
from .compound_parser import CompoundParser
from .cross_sections import CS_Total


KD = 4.15179082788e-4
HC_4PI = 9.8663479e-9  # ??? h*c/4pi apparently ???


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
    rv = 0.0
    for i, j in zip(elements, mass_fractions):
        fi = Fi(i, E)
        aw = AtomicWeight(i)
        rv += j * KD * (i + fi) / aw / E / E
    return 1 - rv * density


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
        xs = CS_Total(i, E)
        rv += xs * j
    return rv * density * HC_4PI / E


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
def _Refractive_Index(elements, mass_fractions, E, density):
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
        fi = Fi(i, E)
        aw = AtomicWeight(i)
        rv_real += j * KD * (i + fi) / aw / E / E
        xs = CS_Total(i, E)
        rv_imag += xs * j
    return 1 - rv_real * density + (rv_imag * density * HC_4PI / E) * 1j


def Refractive_Index(compound: str, E: float, density: float) -> complex:
    compound_dict = CompoundParser(compound)
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    if any(i > 103 for i in elements) or any(i < 1 for i in elements):
        raise ValueError("Z out of range: 1 to 103.")
    return _Refractive_Index(elements, mass_fractions, E, density)
