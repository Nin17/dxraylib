"""_summary_
"""

# TODO docstring

from __future__ import annotations
import typing

import xraylib


class compoundData(typing.TypedDict):
    """
    A compound dataset.
    """

    #: number of different elements in the compound
    nElements: int

    #: number of atoms in the formula. Since indices may be real numbers, this
    #: attribute is of type float
    nAtomsAll: float

    #: a tuple (length = nElements) containing the elements in ascending order
    Elements: tuple[int, ...]

    #: a tuple (length = nElements) containg the mass fractions of the elements
    #: in Elements
    massFractions: tuple[float, ...]

    #: a tuple (length = nElements) containing the number of atoms each element
    #: has in the compound
    nAtoms: tuple[float, ...]

    #: the molar mass of the compound, in g/mol
    molarMass: float


def CompoundParser(compoundString: str) -> compoundData:
    """
    Wrapper around xraylib.CompoundParser

    Parse a chemical formula compoundString to a compoundData dictionary.
    Chemical formulas may contain (nested) brackets, followed by an
    integer or real number (with a dot) subscript. Examples of accepted
    formulas are: H20, Ca5(PO4)3F, Ca5(PO4)F0.33Cl0.33(OH)0.33

    Parameters
    ----------
    compoundString : str
       chemical formula to parse

    Returns
    -------
    compoundData
        _description_

    Raises
    ------
    ValueError
        If compoundString is an invalid chemical formula
    """
    return xraylib.CompoundParser(compoundString)


def AtomicNumberToSymbol(Z: int) -> str:
    """
    Wrapper around xraylib.AtomicNumberToSymbol

    Return the symbol for the element of atomic number Z.

    Parameters
    ----------
    Z : int
        atomic number

    Returns
    -------
    str
        element symbol

    Raises
    ------
    ValueError
        if Z is an invalid atomic number: Z<1 ^ Z>107
    """
    return xraylib.AtomicNumberToSymbol(Z)


def SymbolToAtomicNumber(symbol: str) -> int:
    """
    Wrapper around xraylib.SymbolToAtomicNumber

    Return the atomic number for the elemtent with symbol: symbol.

    Parameters
    ----------
    symbol : str
        element symbol

    Returns
    -------
    int
        atomic number

    Raises
    ------
    ValueError
        if symbol is an invalid element symbol or corresponds to an element
        with Z>107
    """
    xraylib.SymbolToAtomicNumber(symbol)
