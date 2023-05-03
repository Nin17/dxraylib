"""
Tests for dxraylib.AtomicLevelWidth
"""

from dxraylib._load import load
from tests.utilities import Indexors2D


class TestAtomicLevelWidth(Indexors2D):
    """
    Test class for dxraylib.AtomicLevelWidth
    """

    shape = load("atomic_level_width").shape
