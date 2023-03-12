"""
Tests for dxraylib.Fii
"""

from dxraylib.fii import _FII

from tests.utilities import CubicInterpolators


class TestFii(CubicInterpolators):
    """
    Test class for dxraylib.Fii
    """

    data = (_FII,)
    scale = ((lambda x: x, 1),)
