"""
Tests for jaxraylib.ElementDensity
"""

from jaxraylib.densities import DEN

from tests.utilities import Indexors


class TestElementDensity(Indexors):
    """
    Test class for jaxraylib.ElementDensity

    Parameters
    ----------
    Indexors : type
        Base test class for functions that index an array of data
    """

    size = DEN.size
