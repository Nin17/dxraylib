"""
Tests for dxraylib.ElementDensity
"""

from dxraylib.densities import _DEN

from tests.utilities import Indexors1D


class TestElementDensity(Indexors1D):
    """
    Test class for dxraylib.ElementDensity
    """

    size = _DEN.size
