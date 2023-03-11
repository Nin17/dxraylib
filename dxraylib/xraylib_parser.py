"""_summary_
"""

# TODO docstring
# TODO type hint dict

import xraylib


def CompoundParser(compoundString: str) -> dict:
    """
    Wrapper around xraylib.CompoundParser

    Parameters
    ----------
    compoundString : str
       The compound to parse.

    Returns
    -------
    dict
        _description_
    """
    return xraylib.CompoundParser(compoundString)


def AtomicNumberToSymbol(Z: int) -> dict:
    """_summary_

    Parameters
    ----------
    Z : int
        _description_

    Returns
    -------
    dict
        _description_
    """
    return xraylib.AtomicNumberToSymbol(Z)
