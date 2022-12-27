"""_summary_
"""
# ???
# TODO python implementation of this

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
