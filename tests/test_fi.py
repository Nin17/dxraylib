"""
Tests for jaxraylib.Fi
"""

from jaxraylib.fi import FI

from tests.utilities import CubicInterpolators


class Fi(CubicInterpolators):
    """
    Base test class for jaxraylib.Fi

    Parameters
    ----------
    CubicInterpolators : type
        Base test class for functions that interpolate an array of data
    """

    data: tuple = (FI,)


class TestFi(Fi):
    """
    Test class for jaxraylib.Fi

    Parameters
    ----------
    Fi : type
        Base test class for jaxraylib.Fi
    """
