"""
Tests for jaxraylib.AtomicWeight
"""

from jaxraylib.atomicweight import AW

from tests.utilities import Indexors


class AtomicWeight(Indexors):
    """
    Base test class for jaxraylib.AtomicWeight

    Parameters
    ----------
    Indexors : type
        Base test class for functions that index an array of data
    """

    size: int = AW.size


class TestAtomicWeight(AtomicWeight):
    """
    Test class for jaxraylib.AtomicWeight

    Parameters
    ----------
    AtomicWeight : type
        Base test class for jaxraylib.AtomicWeight
    """
