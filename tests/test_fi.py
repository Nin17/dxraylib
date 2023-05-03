"""
Tests for dxraylib.Fi
"""

from dxraylib._load import load
from tests.utilities import CubicInterpolators


class Fi(CubicInterpolators):
    """
    Base test class for dxraylib.Fi
    """

    data = (load("fi"),)
    scale = ((lambda x: x, 1),)


class TestFi(Fi):
    """
    Test class for dxraylib.Fi
    """
