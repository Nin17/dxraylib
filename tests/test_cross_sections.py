"""
Tests for jaxraylib.CS_Compt, jaxraylib.CS_Energy, jaxraylib.CS_Photo,
jaxraylib.CS_Rayl and jaxraylib.CS_Total
"""

import numpy as np

from dxraylib.cross_sections import _CS_COMPT, _CS_PHOTO, _CS_RAYL, _CS_ENERGY

from tests.utilities import CubicInterpolators


class CS_Compt(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Compt
    """

    data: tuple = (_CS_COMPT,)
    scale = ((np.exp, 1000),)


class TestCS_Compt(CS_Compt):
    """
    Test class for jaxraylib.CS_Compt
    """


class CS_Energy(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Energy
    """

    data: tuple = (_CS_ENERGY,)
    scale = ((np.exp, 1),)


class TestCS_Energy(CS_Energy):
    """
    Test class for jaxraylib.CS_Energy
    """


class CS_Photo(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Photo
    """

    data: tuple = (_CS_PHOTO,)
    scale = ((np.exp, 1000),)


class TestCS_Photo(CS_Photo):
    """
    Test class for jaxraylib.CS_Photo
    """


class CS_Rayl(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Rayl
    """

    data: tuple = (_CS_RAYL,)
    scale = ((np.exp, 1000),)


class TestCS_Rayl(CS_Rayl):
    """
    Test class for jaxraylib.CS_Rayl
    """


class CS_Total(CubicInterpolators):
    """
    Base test class for jaxraylib.CS_Total
    """

    data: tuple = _CS_COMPT, _CS_PHOTO, _CS_RAYL
    scale = ((np.exp, 1000),)


class TestCS_Total(CS_Total):
    """
    Test class for jaxraylib.CS_Total
    """
