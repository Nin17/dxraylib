"""_summary_
"""

from .anomalous_scattering import Fi
from .atomic_weight import AtomicWeight
from .compound_parser import CompoundParser
from .cross_sections import CS_Total


KD = 4.15179082788e-4
HC_4PI = 9.8663479e-9  # h*c/4pi apparently ???


def Refractive_Index_Re(compound: str, E: float, density: float) -> float:
    compound_dict = CompoundParser(compound)
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    assert len(mass_fractions) == compound_dict["nElements"]
    assert len(elements) == compound_dict["nElements"]
    rv = 0.0
    for i, j in zip(elements, mass_fractions):
        fi = Fi(i, E)
        aw = AtomicWeight(i)
        rv += j * KD * (i + fi) / aw / E / E
    return 1 - rv * density


def Refractive_Index_Im(compound: str, E: float, density: float) -> float:
    compound_dict = CompoundParser(compound)
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    assert len(mass_fractions) == compound_dict["nElements"]
    assert len(elements) == compound_dict["nElements"]
    rv = 0.0
    for i, j in zip(elements, mass_fractions):
        xs = CS_Total(i, E)
        rv += xs * j
    return rv * density * HC_4PI / E


def Refractive_Index(compound: str, E: float, density: float) -> complex:
    compound_dict = CompoundParser(compound)
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    assert len(mass_fractions) == compound_dict["nElements"]
    assert len(elements) == compound_dict["nElements"]
    rv_real = 0.0
    rv_imag = 0.0
    for i, j in zip(elements, mass_fractions):
        fi = Fi(i, E)
        aw = AtomicWeight(i)
        rv_real += j * KD * (i + fi) / aw / E / E

        xs = CS_Total(i, E)
        rv_imag += xs * j

    return 1 - rv_real * density + (rv_imag * density * HC_4PI / E) * 1j
