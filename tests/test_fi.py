"""
Tests for dxraylib.Fi
"""

from dxraylib.fi import _FI

from tests.utilities import CubicInterpolators


class Fi(CubicInterpolators):
    """
    Base test class for dxraylib.Fi
    """

    data = (_FI,)
    scale = ((lambda x: x, 1),)


class TestFi(Fi):
    """
    Test class for dxraylib.Fi
    """
