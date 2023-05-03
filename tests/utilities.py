"""_summary_
"""

# TODO type hint functions
# TODO do it all with kwargs instead
import functools
import inspect
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
    
    
    @functools.cached_property
    def xrl_func(self) -> callable:
        return getattr(xrl, self.func)
    
    @functools.cached_property
    def xrl_np_func(self) -> callable:
        return getattr(xrl_np, self.func)
    
    @functools.cached_property
    def dxrl_func(self) -> callable:
        return getattr(dxrl, self.func)

    # TODO test signature
    def test_signature(self):
        xrl_sig = inspect.signature(self.xrl_func).parameters.keys()
        dxrl_sig = inspect.signature(self.dxrl_func).parameters.keys()
        assert xrl_sig == dxrl_sig
        # print(inspect.signature(self.xrl_func).parameters.keys())
        # print(inspect.signature(self.dxrl_func).parameters.keys())
    
    def test__name__(self):
        assert self.dxrl_func.__name__ == self.func

    def test_xrl(self, *args):
        """_summary_"""
        args = args or self.args0
        dxrl_output = self.dxrl_func(*args)
        assert dxrl_output.shape == ()
        try:
            xrl_output = self.xrl_func(*args)
            np.testing.assert_allclose(
                xrl_output, dxrl_output, atol=ATOL, rtol=RTOL
            )
        except ValueError:
            assert np.isnan(dxrl_output).all()

    def test_xrlnp(self, *args):
        """_summary_

        Parameters
        ----------
        jxrl_args : _type_, optional
            _description_, by default None
        """
        # if hasattr(xrl_np, self.func):
        args = args or self.args
        xrlnp_output = self.xrl_np_func(*args)
        dxrl_output = np.nan_to_num(self.dxrl_func(*args))
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
    
    def test_xrl(self, *args):
        return super().test_xrl(*args)
    
    def test_xrlnp(self, *args):
        return super().test_xrlnp(*args)

    def test_extrapolate_2big(self):
        z = np.arange(1, self.shape0)
        super().test_xrlnp(z, self.nanmax + 1 * np.finfo(np.float16).eps, *self.args[2:])

    def test_extrapolate_2small(self):
        z = np.arange(1, self.shape0)
        super().test_xrlnp(z, self.nanmin - 1 * np.finfo(np.float16).eps, *self.args[2:])

    def test_negative(self):
        z = np.arange(1, self.shape0)
        e = np.array([-np.finfo(np.float64).eps])
        super().test_xrlnp(z, e, *self.args[2:])

    def test_z_1_2big(self):
        z = np.array([self.shape0 + 1])
        e = self.nanmin + (self.nanmax - self.nanmin) / 2
        super().test_xrlnp(z, e, *self.args[2:])

    def test_z_1_2small(self):
        z = np.array([0])
        e = self.nanmin + (self.nanmax - self.nanmin) / 2
        super().test_xrlnp(z, e, *self.args[2:])

    def test_0(self):
        z = np.arange(1, self.shape0)
        e = np.array([0.0])
        super().test_xrlnp(z, e, *self.args[2:])

    # TODO change this to test if @ h==0 it still works
    # @pytest.mark.skip("Doesn't work yet")
    # @pytest.mark.xfail(reason="Final data point fails")
    def test_data_points(self):
        z = np.arange(1, self.shape0 + 1)
        unique = np.unique(
            np.concatenate([i.ravel() for i in self.x_data])
        ).ravel()
        unique = unique[~np.isnan(unique)]
        super().test_xrlnp(z, unique, *self.args[2:])


class Indexors1D(TestBaseXrlXrlnp):
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

    # TODO out of bounds test

    size: int
    
    def test_xrl(self, *args):
        return super().test_xrl(*args)
    
    def test_xrlnp(self, *args):
        return super().test_xrlnp(*args)

    @functools.cached_property
    def args(self) -> NDArray[np.int64]:
        """_summary_

        Returns
        -------
        NDArray[np.int64]
            _description_
        """
        return (np.arange(1, self.size),)
    
    def test_out_of_range(self):
        super().test_xrlnp(np.arange(-2*self.size, 2*self.size), *self.args[1:])


class Indexors2D(TestBaseXrlXrlnp):
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

    # TODO out of bounds tests

    shape: tuple[int, int]
    offset: tuple[int, int]
    
    def test_xrl(self, *args):
        return super().test_xrl(*args)
    
    def test_xrlnp(self, *args):
        return super().test_xrlnp(*args)

    @functools.cached_property
    def args(self) -> NDArray[np.int64]:
        """

        Returns
        -------
        NDArray[np.int64]
            _description_
        """
        return np.arange(1, self.shape[0]), np.arange(1, self.shape[1])
    
    def test_out_of_range(self):
        a = np.arange(-2*self.shape[0], 2*self.shape[0])
        b = np.arange(-2*self.shape[1], 2*self.shape[1])
        super().test_xrlnp(a, b)


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
