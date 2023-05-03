"""
Tests for dxraylib.Fii
"""

from dxraylib._load import load
from tests.utilities import CubicInterpolators


class TestFii(CubicInterpolators):
    """
    Test class for dxraylib.Fii
    """

    data = (load("fii"),)
    scale = ((lambda x: x, 1),)
