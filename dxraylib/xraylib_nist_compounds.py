"""_summary_
"""
# TODO docstring
# TODO type hint dict

from __future__ import annotations
import typing

import xraylib


class compoundDataNIST(typing.TypedDict):
    """
    A NIST compound dataset.
    """

    #: a string containing the full name of the compound, as retrieved from the
    #: NIST database
    name: str

    #: number of different elements in the compound
    nElements: int

    #: a tuple (length = nElements) containing the elements, in ascending order
    Elements: tuple[int, ...]

    #: a tuple (length = nElements) containg the mass fractions of the elements
    #: in Elements
    massFractions: tuple[float, ...]

    #: the density of the compound, expressed in g/cm³
    density: float


def GetCompoundDataNISTByName(compoundString: str) -> compoundDataNIST:
    """_summary_

    Parameters
    ----------
    compoundString : str
        _description_

    Returns
    -------
    compoundDataNIST
        _description_
    """
    return xraylib.GetCompoundDataNISTByName(compoundString)
