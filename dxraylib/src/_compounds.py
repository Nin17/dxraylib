"""Parse a compound as a chemical formula or NIST compound if that failed."""

from __future__ import annotations

from .xraylib_nist_compounds import GetCompoundDataNISTByName, compoundDataNIST
from .xraylib_parser import CompoundParser, compoundData


def _compound_data(compound: str) -> compoundData | compoundDataNIST:
    """Parse compound as a chemical formula and then as a NIST compound if that fails.

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
                msg.replace("\n", "").replace("  ", ""),
            ) from error

    return compound_dict
