"""_summary_
"""


from .anomalous_scattering import fi
from .atomic_weight import atomic_weight
from .compound_parser import compound_parser
from .cross_sections import cs_total


KD = 4.15179082788e-4
HC_4PI = 9.8663479e-9  # h*c/4pi apparently ???


def refractive_index_re(compound: str, energy: float, density: float) -> float:
    compound_dict = compound_parser(compound)
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    assert len(mass_fractions) == compound_dict["nElements"]
    assert len(elements) == compound_dict["nElements"]
    rv = 0.0
    for i, j in zip(elements, mass_fractions):
        _fi = fi(i, energy)
        aw = atomic_weight(i)
        rv += j * KD * (i + _fi) / aw / energy / energy
    return 1 - rv * density


def refractive_index_im(compound: str, energy: float, density: float) -> float:
    compound_dict = compound_parser(compound)
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    assert len(mass_fractions) == compound_dict["nElements"]
    assert len(elements) == compound_dict["nElements"]
    rv = 0.0
    for i, j in zip(elements, mass_fractions):
        xs = cs_total(i, energy)
        rv += xs * j
    return rv * density * HC_4PI / energy


def refractive_index(compound: str, energy: float, density: float) -> complex:
    compound_dict = compound_parser(compound)
    mass_fractions = compound_dict["massFractions"]
    elements = compound_dict["Elements"]
    assert len(mass_fractions) == compound_dict["nElements"]
    assert len(elements) == compound_dict["nElements"]
    rv_real = 0.0
    rv_imag = 0.0
    for i, j in zip(elements, mass_fractions):
        _fi = fi(i, energy)
        aw = atomic_weight(i)
        rv_real += j * KD * (i + _fi) / aw / energy / energy

        xs = cs_total(i, energy)
        rv_imag += xs * j

    return 1 - rv_real * density + (rv_imag * density * HC_4PI / energy) * 1j
