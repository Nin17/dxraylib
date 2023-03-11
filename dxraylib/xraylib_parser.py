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
