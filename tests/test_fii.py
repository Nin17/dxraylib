"""
Tests for jaxraylib.Fii
"""

from dxraylib.fii import _FII

from tests.utilities import CubicInterpolators


class TestFii(CubicInterpolators):
    """
    Test class for jaxraylib.Fii
    """

    data = (_FII,)
    scale = ((lambda x: x, 1),)
