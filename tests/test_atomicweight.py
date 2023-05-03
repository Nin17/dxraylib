"""
Tests for dxraylib.AtomicWeight
"""

from dxraylib._load import load
from tests.utilities import Indexors1D


class AtomicWeight(Indexors1D):
    """
    Base test class for dxraylib.AtomicWeight
    """

    size = load("atomic_weight").size


class TestAtomicWeight(AtomicWeight):
    """
    Test class for dxraylib.AtomicWeight
    """
