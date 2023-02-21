"""
Tests for jaxraylib.Fii
"""

from jaxraylib.fii import FII

from tests.utilities import CubicInterpolators


class TestFii(CubicInterpolators):
    """
    Test class for jaxraylib.Fii
    """

    data = (FII,)
