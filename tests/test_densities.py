"""
Tests for dxraylib.ElementDensity
"""

from dxraylib._load import load
from tests.utilities import Indexors1D


class TestElementDensity(Indexors1D):
    """
    Test class for dxraylib.ElementDensity
    """

    size = load("densities").size
