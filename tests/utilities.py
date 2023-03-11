"""_summary_
"""

# TODO type hint functions
import functools
import itertools
import typing


import pytest

import numpy as np
from numpy.typing import NDArray

import xraylib as xrl
import xraylib_np as xrl_np

import dxraylib as dxrl

from tests.config import ATOL, RTOL, N

rng = np.random.default_rng()


ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
]


class TestBaseXrlXrlnp:
    """_summary_"""

    @property
    def args(self) -> tuple[typing.Any, ...]:
        """_summary_

        Returns
        -------
        tuple[typing.Any, ...]
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError("Must overwrite args property!")

    @functools.cached_property
    def args0(self) -> tuple[typing.Any, ...]:
        """_summary_

        Returns
        -------
        tuple[typing.Any, ...]
            _description_
        """
        return tuple(
            i if isinstance(i, str) else rng.choice(i) for i in self.args
        )

    @functools.cached_property
    def func(self) -> str:
        """_summary_

        Returns
        -------
        str
            _description_
        """
        return self.__class__.__name__.removeprefix("Test")

    def test_xrl(self, *args):
        """_summary_"""
        args = args or self.args0
        xrl_output = getattr(xrl, self.func)(*args)
        dxrl_output = getattr(dxrl, self.func)(*args)
        assert dxrl_output.shape == ()
        np.testing.assert_allclose(
            xrl_output, dxrl_output, atol=ATOL, rtol=RTOL
        )

    def test_xrlnp(self, *args):
        """_summary_

        Parameters
        ----------
        jxrl_args : _type_, optional
            _description_, by default None
        """
        args = args or self.args
        xrlnp_output = getattr(xrl_np, self.func)(*args)
        dxrl_output = np.nan_to_num(
            getattr(dxrl, self.func)(*args)
        )
        assert xrlnp_output.shape == dxrl_output.shape
        np.testing.assert_allclose(
            xrlnp_output, dxrl_output, atol=ATOL, rtol=RTOL
        )


class CubicInterpolators(TestBaseXrlXrlnp):
    """_summary_"""

    data: tuple[NDArray, ...]
    scale: tuple[tuple[typing.Callable, float], ...]

    @functools.cached_property
    def x_data(self) -> tuple[NDArray, ...]:
        return tuple(i[:, 0] for i in self.data)

    @functools.cached_property
    def shape0(self):
        return min(i.shape[0] for i in self.data)

    @functools.cached_property
    def nanmin(self):
        return np.max(
            [
                func(np.nanmin(i, axis=1)[: self.shape0]) / factor
                for i, (func, factor) in itertools.zip_longest(
                    self.x_data, self.scale, fillvalue=(lambda x: x, 1)
                )
            ],
            axis=0,
        )

    @functools.cached_property
    def nanmax(self):
        return np.min(
            [
                func(np.nanmax(i, axis=1)[: self.shape0]) / factor
                for i, (func, factor) in itertools.zip_longest(
                    self.x_data, self.scale, fillvalue=(lambda x: x, 1)
                )
            ],
            axis=0,
        )

    @functools.cached_property
    def args(self):
        z = np.arange(1, self.shape0)
        e = (self.nanmax - self.nanmin) * rng.random(N) + self.nanmin
        return z, e

    @functools.cached_property
    def args0(self):
        j = rng.integers(self.shape0 - 1)
        z = self.args[0][j]
        e = self.args[1][j]
        return z, e

    def test_extrapolate_2big(self):
        z = np.arange(1, self.shape0)
        super().test_xrlnp(z, self.nanmax + 1 * np.finfo(np.float16).eps)

    def test_extrapolate_2small(self):
        z = np.arange(1, self.shape0)
        super().test_xrlnp(z, self.nanmin - 1 * np.finfo(np.float16).eps)

    def test_negative(self):
        z = np.arange(1, self.shape0)
        e = np.array([-np.finfo(np.float64).eps])
        super().test_xrlnp(z, e)

    def test_z_1_2big(self):
        z = np.array([self.shape0 + 1])
        e = self.nanmin + (self.nanmax - self.nanmin) / 2
        super().test_xrlnp(z, e)

    def test_z_1_2small(self):
        z = np.array([0])
        e = self.nanmin + (self.nanmax - self.nanmin) / 2
        super().test_xrlnp(z, e)
    
    def test_0(self):
        z = np.arange(1, self.shape0)
        e = np.array([0.0])
        super().test_xrlnp(z, e)

    # TODO change this to test if @ h==0 it still works
    # @pytest.mark.skip("Doesn't work yet")
    # @pytest.mark.xfail(reason="Final data point fails")
    def test_data_points(self):
        z = np.arange(1, self.shape0 + 1)
        unique = np.unique(
            np.concatenate([i.ravel() for i in self.x_data])
        ).ravel()
        unique = unique[~np.isnan(unique)]
        super().test_xrlnp(z, unique)


class Indexors(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # Must overwrite
    size: int

    @functools.cached_property
    def args(self) -> NDArray[np.int64]:
        """_summary_

        Returns
        -------
        NDArray[np.int64]
            _description_
        """
        # ??? test out of bounds in separate test
        return (np.arange(1, self.size + 1),)


class Analytic(TestBaseXrlXrlnp):
    """_summary_

    Parameters
    ----------
    TestBaseXrlXrlnp : _type_
        _description_
    """

    ab: tuple[tuple[float, float], ...]

    @functools.cached_property
    def args(self):
        return tuple((i - j) * rng.random(N) + j for i, j in self.ab)


def random_formula(size) -> str:
    # TODO fix this - making bad formulae
    """_summary_

    Returns
    -------
    str
        _description_
    """
    while True:
        n = min(size, N)
        symbols = rng.choice(ELEMENTS[:size], n, replace=False)
        numbers = rng.random(n)
        formula = "".join(i + str(j)[:n] for i, j in zip(symbols, numbers))
        try:
            xrl.CompoundParser(formula)
            return formula
        except ValueError:
            continue
