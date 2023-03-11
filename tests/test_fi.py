"""
Tests for jaxraylib.Fi
"""

from dxraylib.fi import _FI

from tests.utilities import CubicInterpolators


class Fi(CubicInterpolators):
    """
    Base test class for jaxraylib.Fi
    """

    data = (_FI,)
    scale = ((lambda x: x, 1),)


class TestFi(Fi):
    """
    Test class for jaxraylib.Fi
    """
