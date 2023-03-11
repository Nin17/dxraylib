"""
Tests for jaxraylib.ElementDensity
"""

from dxraylib.densities import _DEN

from tests.utilities import Indexors


class TestElementDensity(Indexors):
    """
    Test class for jaxraylib.ElementDensity
    """

    size = _DEN.size
