"""
Tests for dxraylib.CS_Compt, dxraylib.CS_Energy, dxraylib.CS_Photo,
dxraylib.CS_Rayl and dxraylib.CS_Total
"""

import numpy as np

from dxraylib.cross_sections import _CS_COMPT, _CS_PHOTO, _CS_RAYL, _CS_ENERGY

from tests.utilities import CubicInterpolators


class CS_Compt(CubicInterpolators):
    """
    Base test class for dxraylib.CS_Compt
    """

    data: tuple = (_CS_COMPT,)
    scale = ((np.exp, 1000),)


class TestCS_Compt(CS_Compt):
    """
    Test class for dxraylib.CS_Compt
    """


class CS_Energy(CubicInterpolators):
    """
    Base test class for dxraylib.CS_Energy
    """

    data: tuple = (_CS_ENERGY,)
    scale = ((np.exp, 1),)


class TestCS_Energy(CS_Energy):
    """
    Test class for dxraylib.CS_Energy
    """


class CS_Photo(CubicInterpolators):
    """
    Base test class for dxraylib.CS_Photo
    """

    data: tuple = (_CS_PHOTO,)
    scale = ((np.exp, 1000),)


class TestCS_Photo(CS_Photo):
    """
    Test class for dxraylib.CS_Photo
    """


class CS_Rayl(CubicInterpolators):
    """
    Base test class for dxraylib.CS_Rayl
    """

    data: tuple = (_CS_RAYL,)
    scale = ((np.exp, 1000),)


class TestCS_Rayl(CS_Rayl):
    """
    Test class for dxraylib.CS_Rayl
    """


class CS_Total(CubicInterpolators):
    """
    Base test class for dxraylib.CS_Total
    """

    data: tuple = _CS_COMPT, _CS_PHOTO, _CS_RAYL
    scale = ((np.exp, 1000),)


class TestCS_Total(CS_Total):
    """
    Test class for dxraylib.CS_Total
    """
