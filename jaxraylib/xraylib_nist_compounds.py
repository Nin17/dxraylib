"""_summary_
"""

import xraylib

# TODO docstring
# TODO type hint dict
def GetCompoundDataNISTByName(compoundString: str) -> dict:
    """_summary_

    Parameters
    ----------
    compoundString : str
        _description_

    Returns
    -------
    dict
        _description_
    """
    return xraylib.GetCompoundDataNISTByName(compoundString)
