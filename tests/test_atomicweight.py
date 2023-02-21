"""
Tests for jaxraylib.AtomicWeight
"""

from jaxraylib.atomicweight import AW

from tests.utilities import Indexors


class AtomicWeight(Indexors):
    """
    Base test class for jaxraylib.AtomicWeight
    """

    size = AW.size


class TestAtomicWeight(AtomicWeight):
    """
    Test class for jaxraylib.AtomicWeight
    """
