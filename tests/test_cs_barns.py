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

    Parameters
    ----------
    CS_Total : type
        Base test class for jaxraylib.CS_Total
    AtomicWeight : type
        Base test class for jaxraylib.AtomicWeight
    """


class TestCSb_Photo(CS_Photo, AtomicWeight):
    """
    Test class for jaxraylib.CSb_Photo

    Parameters
    ----------
    CS_Photo : type
        Base test class for jaxraylib.CS_Photo
    AtomicWeight : type
        Base test class for jaxraylib.AtomicWeight
    """


class TestCSb_Rayl(CS_Rayl, AtomicWeight):
    """
    Test class for jaxraylib.CSb_Rayl

    Parameters
    ----------
    CS_Rayl : type
        Base test class for jaxraylib.CS_Rayl
    AtomicWeight : type
        Base test class for jaxraylib.AtomicWeight
    """


class TestCSb_Compt(CS_Compt, AtomicWeight):
    """
    Test class for jaxraylib.CSb_Compt

    Parameters
    ----------
    CS_Compt : type
        Base test class for jaxraylib.CS_Compt
    AtomicWeight : type
        Base test class for jaxraylib.AtomicWeight
    """


class TestDCSb_Rayl(DCS_Rayl, AtomicWeight):
    """
    Test class for jaxraylib.DCSb_Rayl

    Parameters
    ----------
    DCS_Rayl : type
        Base test class for jaxraylib.DCS_Rayl
    AtomicWeight : type
        Base test class for jaxraylib.AtomicWeight
    """


class TestDCSb_Compt(DCS_Compt, AtomicWeight):
    """
    Test class for jaxraylib.DCSb_Compt

    Parameters
    ----------
    DCS_Compt : type
        Base test class for jaxraylib.DCS_Compt
    AtomicWeight : type
        Base test class for jaxraylib.AtomicWeight
    """


class TestDCSPb_Rayl(DCSP_Rayl, AtomicWeight):
    """
    Test class for jaxraylib.DCSPb_Rayl

    Parameters
    ----------
    DCSP_Rayl : type
        Base test class for jaxraylib.DCSP_Rayl
    AtomicWeight : type
        Base test class for jaxraylib.AtomicWeight
    """


class TestDCSPb_Compt(DCSP_Compt, AtomicWeight):
    """
    Test class for jaxraylib.DCSPb_Compt

    Parameters
    ----------
    DCSP_Compt : type
        Base test class for jaxraylib.DCSP_Compt
    AtomicWeight : type
        Base test class for jaxraylib.AtomicWeight
    """
