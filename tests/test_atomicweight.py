"""
Tests for dxraylib.AtomicWeight
"""

from dxraylib.atomicweight import _AW

from tests.utilities import Indexors1D


class AtomicWeight(Indexors1D):
    """
    Base test class for dxraylib.AtomicWeight
    """

    size = _AW.size


class TestAtomicWeight(AtomicWeight):
    """
    Test class for dxraylib.AtomicWeight
    """
