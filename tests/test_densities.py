"""
Tests for jaxraylib.ElementDensity
"""

from dxraylib.densities import _DEN

from tests.utilities import Indexors1D


class TestElementDensity(Indexors1D):
    """
    Test class for jaxraylib.ElementDensity
    """

    size = _DEN.size
