"""
Tests for dxraylib.JumpFactor
"""

from dxraylib._load import load
from tests.utilities import Indexors2D


class TestJumpFactor(Indexors2D):
    """
    Test class for dxraylib.JumpFactor
    """

    shape = load("jump").shape
