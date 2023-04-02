"""
helper function to parse a compound as a chemical formula and then as NIST
compounds if that failed
"""
from __future__ import annotations

from .xraylib_nist_compounds import compoundDataNIST, GetCompoundDataNISTByName
from .xraylib_parser import compoundData, CompoundParser


def _compound_data(compound: str) -> compoundData | compoundDataNIST:
    """
    Helper function to try parsing a compound as a chemical formula and then as
    a NIST compound if that fails.

    Parameters
    ----------
    compound : str
        chemical formula or NIST compound

    Returns
    -------
    compoundData | compoundDataNIST
        dataset corresponding to the given compound

    Raises
    ------
    ValueError
        if compound is neither a valid chemical formula nor a valid NIST
        compound
    """
    try:
        compound_dict = CompoundParser(compound)
    except ValueError:
        try:
            compound_dict = GetCompoundDataNISTByName(compound)
        except ValueError as error:
            msg = """
            Compound is not a valid chemical formula and is not
             present in the NIST compound database
            """
            raise ValueError(
                msg.replace("\n", "").replace("  ", "")
            ) from error

    return compound_dict
