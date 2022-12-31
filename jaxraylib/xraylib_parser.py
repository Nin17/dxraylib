"""_summary_
"""

import xraylib


# TODO docstring
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
