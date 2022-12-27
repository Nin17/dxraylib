"""_summary_
"""
# ???
# TODO python implementation of this

import xraylib


def compound_parser(compound: str) -> dict:
    """
    Wrapper around xraylib.CompoundParser

    Parameters
    ----------
    compound : str
       The compound to parse.

    Returns
    -------
    dict
        _description_
    """
    return xraylib.CompoundParser(compound)
