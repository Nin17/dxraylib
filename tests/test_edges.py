"""
Tests for dxraylib.EdgeEnergy
"""

from dxraylib._load import load
from tests.utilities import Indexors2D


class TestEdgeEnergy(Indexors2D):
    """
    Test class for dxraylib.EdgeEnergy
    """

    shape = load("edges").shape
