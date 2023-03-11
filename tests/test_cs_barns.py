"""
Tests for cross sections in barns
"""

from tests.test_atomicweight import AtomicWeight
from tests.test_cross_sections import (
    CS_Compt,
    CS_Photo,
    CS_Rayl,
    CS_Total,
)
from tests.test_polarized import DCSP_Compt, DCSP_Rayl
from tests.test_scattering import DCS_Compt, DCS_Rayl


class TestCSb_Total(CS_Total, AtomicWeight):
    """
    Test class for jaxraylib.CSb_Total
    """


class TestCSb_Photo(CS_Photo, AtomicWeight):
    """
    Test class for jaxraylib.CSb_Photo
    """


class TestCSb_Rayl(CS_Rayl, AtomicWeight):
    """
    Test class for jaxraylib.CSb_Rayl
    """


class TestCSb_Compt(CS_Compt, AtomicWeight):
    """
    Test class for jaxraylib.CSb_Compt
    """


class TestDCSb_Rayl(DCS_Rayl, AtomicWeight):
    """
    Test class for jaxraylib.DCSb_Rayl
    """


class TestDCSb_Compt(DCS_Compt, AtomicWeight):
    """
    Test class for jaxraylib.DCSb_Compt
    """


class TestDCSPb_Rayl(DCSP_Rayl, AtomicWeight):
    """
    Test class for jaxraylib.DCSPb_Rayl
    """


class TestDCSPb_Compt(DCSP_Compt, AtomicWeight):
    """
    Test class for jaxraylib.DCSPb_Compt
    """
