"""
Tests for dxraylib.CosKronTransProb
"""

from dxraylib._load import load
from tests.utilities import Indexors2D


class TestCosKronTransProb(Indexors2D):
    """
    Test class for dxraylib.CosKronTransProb
    """

    shape = load("coskron").shape
