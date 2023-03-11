"""
Tests for jaxraylib.AtomicWeight
"""

from dxraylib.atomicweight import _AW

from tests.utilities import Indexors


class AtomicWeight(Indexors):
    """
    Base test class for jaxraylib.AtomicWeight
    """

    size = _AW.size


class TestAtomicWeight(AtomicWeight):
    """
    Test class for jaxraylib.AtomicWeight
    """
