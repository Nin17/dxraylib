"""_summary_
"""

import functools

import numpy as np
import pytest

from tests.utilities import random_formula, ELEMENTS


from tests.test_cross_sections import (
    CS_Total,
    CS_Photo,
    CS_Rayl,
    CS_Compt,
    CS_Energy,
)
from tests.test_polarized import DCSP_Compt, DCSP_Rayl
from tests.test_scattering import DCS_Compt, DCS_Rayl

rng = np.random.default_rng()

# TODO this is basically the same as in test_refractive_index
def compound(cls):
    """_summary_

    Parameters
    ----------
    og_class : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    @property
    def args0(self):
        formula = random_formula(self.shape0)
        args = list(super(cls, self).args0)
        args[0] = formula
        return tuple(args)

    @pytest.mark.skip("..._CP not implemented in xrl_np")
    def test_xrlnp(_):
        ...

    @pytest.mark.skip("")
    def test_data_points(_):
        ...

    @pytest.mark.skip("")
    def test_extrapolate_2small(_):
        ...

    @pytest.mark.skip("")
    def test_extrapolate_2big(_):
        ...

    @pytest.mark.skip("")
    def test_z_out_of_range(_):
        ...

    @pytest.mark.skip("")
    def test_negative(_):
        ...

    @pytest.mark.skip("")
    def test_z_1_2big(_):
        ...

    @pytest.mark.skip("")
    def test_z_1_2small(_):
        ...

    @pytest.mark.skip("")
    def test_0(_):
        ...

    cls.args0 = args0
    cls.test_xrlnp = test_xrlnp
    cls.test_data_points = test_data_points
    cls.test_extrapolate_2big = test_extrapolate_2big
    cls.test_extrapolate_2small = test_extrapolate_2small
    cls.test_z_out_of_range = test_z_out_of_range
    cls.test_negative = test_negative
    cls.test_z_1_2big = test_z_1_2big
    cls.test_z_1_2small = test_z_1_2small
    cls.test_0 = test_0
    return cls


@compound
class TestCS_Total_CP(CS_Total):
    """
    Test class for jaxraylib.CS_Total_CP
    """


@compound
class TestCS_Photo_CP(CS_Photo):
    """
    Test class for jaxraylib.CS_Photo_CP
    """


@compound
class TestCS_Rayl_CP(CS_Rayl):
    """
    Test class for jaxraylib.CS_Rayl_CP
    """


@compound
class TestCS_Compt_CP(CS_Compt):
    """
    Test class for jaxraylib.CS_Compt_CP
    """


@compound
class TestCSb_Total_CP(CS_Total):
    """
    Test class for jaxraylib.CSb_Total_CP
    """


@compound
class TestCSb_Photo_CP(CS_Photo):
    """
    Test class for jaxraylib.CSb_Photo_CP
    """


@compound
class TestCSb_Rayl_CP(CS_Rayl):
    """
    Test class for jaxraylib.CSb_Rayl_CP
    """


@compound
class TestCSb_Compt_CP(CS_Compt):
    """
    Test class for jaxraylib.CSb_Compt_CP
    """


@compound
class TestCS_Energy_CP(CS_Energy):
    """
    Test class for jaxraylib.CS_Energy_CP
    """


@compound
class TestDCS_Rayl_CP(DCS_Rayl):
    """
    Test class for jaxraylib.DCS_Rayl_CP
    """


@compound
class TestDCS_Compt_CP(DCS_Compt):
    """
    Test class for jaxraylib.DCS_Compt_CP
    """


@compound
class TestDCSb_Rayl_CP(DCS_Rayl):
    """
    Test class for jaxraylib.DCSb_Rayl_CP
    """


@compound
class TestDCSb_Compt_CP(DCS_Compt):
    """
    Test class for jaxraylib.DCSb_Compt_CP
    """


@compound
class TestDCSP_Rayl_CP(DCSP_Rayl):
    """
    Test class for jaxraylib.DCSP_Rayl_CP
    """


@compound
class TestDCSP_Compt_CP(DCSP_Compt):
    """
    Test class for jaxraylib.DCSP_Compt_CP
    """


@compound
class TestDCSPb_Rayl_CP(DCSP_Rayl):
    """
    Test class for jaxraylib.DCSPb_Rayl_CP
    """


@compound
class TestDCSPb_Compt_CP(DCSP_Compt):
    """
    Test class for jaxraylib.DCSPb_Compt_CP
    """
