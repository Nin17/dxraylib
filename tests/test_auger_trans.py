"""
Tests for dxraylib.AugerRate and dxraylib.AugerYield
"""

from dxraylib._load import load
from tests.utilities import Indexors2D


class TestAugerRate(Indexors2D):
    """
    Test class for dxraylib.AugerRate
    """

    shape = load("auger_rates").shape


class TestAugerYield(Indexors2D):
    """
    Test class for dxraylib.AugerYield
    """

    shape = load("auger_yields").shape
