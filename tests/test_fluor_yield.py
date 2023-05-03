"""
Tests for dxraylib.FluorYield
"""

from dxraylib._load import load
from tests.utilities import Indexors2D


class TestFluorYield(Indexors2D):
    """
    Test class for dxraylib.FluorYield
    """

    shape = load("fluor_yield").shape
