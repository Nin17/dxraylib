"""
Tests for jaxraylib.Fii
"""

from jaxraylib.fii import FII

from tests.utilities import CubicInterpolators


class TestFii(CubicInterpolators):
    """
    Test class for jaxraylib.Fii

    Parameters
    ----------
    CubicInterpolators : type
        Base test class for functions that interpolate an array of data
    """

    data: tuple = (FII,)
