"""_summary_
"""

from __future__ import annotations
import typing

import xraylib


# TODO docstings, type hinting


class radioNuclideData(typing.TypedDict):
    """
    A radionuclide dataset.
    """

    #: a string containing the mass number (A), followed by the chemical
    #: element (e.g. 55Fe)
    name: str

    #: atomic number of the radionuclide
    Z: int

    #: mass number of the radionuclide
    A: int

    #: number of neutrons of the radionuclide
    N: int

    #: atomic number of the nuclide after decay, which should be used in
    #: calculating the energy of the emitted X-ray lines using LineEnergy
    Z_xray: int

    #: number of emitted characteristic X-rays
    nXrays: int

    #: a tuple (length = nXrays) of line-type macros, identifying the emitted
    #: X-rays
    XrayLines: tuple[int, ...]

    #: a tuple (length = nXrays) of photons per disintegration, one value per
    #: emitted X-ray
    XrayIntensities: tuple[float, ...]

    #: number of emitted gamma-rays
    nGammas: int

    #: a tuple (length = nGammas) of emitted gamma-ray energies
    GammaEnergies: tuple[float, ...]

    #: a tuple (length = nGammas) of emitted gamma-ray photons per
    #: disintegration
    GammaIntensities: tuple[float, ...]


def GetRadioNuclideDataByName(radioNuclideString: str) -> radioNuclideData:
    """_summary_

    Parameters
    ----------
    radioNuclideString : str
        _description_

    Returns
    -------
    radioNuclideData
        _description_
    """
    return xraylib.GetRadioNuclideDataByName(radioNuclideString)


def GetRadioNuclideDataByIndex(radioNuclideIndex: int) -> radioNuclideData:
    """_summary_

    Parameters
    ----------
    radioNuclideIndex : int
        _description_

    Returns
    -------
    radioNuclideData
        _description_
    """
    return xraylib.GetRadioNuclideDataByIndex(radioNuclideIndex)


def GetRadioNuclideDataList() -> tuple[str, ...]:
    """_summary_

    Returns
    -------
    tuple[str, ...]
        _description_
    """
    return xraylib.GetRadioNuclideDataList()
