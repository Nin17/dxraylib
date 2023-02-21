"""
Tests for jaxraylib.Fi
"""

from jaxraylib.fi import FI

from tests.utilities import CubicInterpolators


class Fi(CubicInterpolators):
    """
    Base test class for jaxraylib.Fi
    """

    data = (FI,)


class TestFi(Fi):
    """
    Test class for jaxraylib.Fi
    """
