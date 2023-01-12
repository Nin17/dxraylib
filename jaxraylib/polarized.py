"""
Differential scattering cross sections for polarized beams
"""

from __future__ import annotations

from .config import jit, jit_kwargs, xp, NDArray, ArrayLike
from .constants import AVOGNUM, MEC2, RE2
from .scattering import _MomentTransf, _FF_Rayl, _SF_Compt
from .atomicweight import _AtomicWeight
from ._utilities import wrapped_partial, xrl_xrlnp


@wrapped_partial(jit, **jit_kwargs)
def _DCSP_Rayl(
    Z: ArrayLike, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> tuple[NDArray, bool]:
    Z = xp.atleast_1d(xp.asarray(Z))
    E = xp.atleast_1d(xp.asarray(E))
    theta = xp.atleast_1d(xp.asarray(theta))
    phi = xp.atleast_1d(xp.asarray(phi))
    q = _MomentTransf(E, theta)[0]
    F = _FF_Rayl(Z, q)[0]
    output = AVOGNUM / _AtomicWeight(Z)[0] * F * F * _DCSP_Thoms(theta, phi)
    return output, xp.isnan(output).any()


@xrl_xrlnp("# TODO error message")
def DCSP_Rayl(
    Z: ArrayLike, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> NDArray:
    """
    Differential Rayleigh scattering cross section
    for polarized beam (cm2/g/sterad)

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        Energy (keV)
    theta : array_like
        scattering polar angle (rad)
    phi : Array
        scattering azimuthal angle (rad)

    Returns
    -------
    Array
        Differential Rayleigh scattering cross section
        for polarized beam (cm2/g/sterad)
    """
    # TODO broadcasting of arguments # ???
    return _DCSP_Rayl(Z, E, theta, phi)


@wrapped_partial(jit, **jit_kwargs)
def _DCSP_Compt(
    Z: ArrayLike, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> tuple[NDArray, bool]:
    Z = xp.asarray(Z)
    E = xp.asarray(E)
    theta = xp.asarray(theta)
    phi = xp.asarray(phi)
    q = _MomentTransf(E, theta)[0]
    S = _SF_Compt(Z, q)[0]
    output = AVOGNUM / _AtomicWeight(Z)[0] * S * _DCSP_KN(E, theta, phi)[0]
    # TODO broadcasting of arguments # ???
    return output, xp.isnan(output).any()


@xrl_xrlnp(" # TODO error message")
def DCSP_Compt(
    Z: ArrayLike, E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> NDArray:
    """
    Differential Compton scattering cross section
    for polarized beam (cm2/g/sterad)

    Parameters
    ----------
    Z : array_like
        atomic number
    E : array_like
        Energy (keV)
    theta : array_like
        scattering polar angle (rad)
    phi : array_like
        scattering azimuthal angle (rad)

    Returns
    -------
    Array
        Differential Compton scattering cross section
        for polarized beam (cm2/g/sterad)
    """
    # TODO broadcasting of arguments # ???
    return _DCSP_Compt(Z, E, theta, phi)


@wrapped_partial(jit, **jit_kwargs)
def _DCSP_KN(
    E: ArrayLike, theta: ArrayLike, phi: ArrayLike
) -> tuple[NDArray, bool]:
    E = xp.atleast_1d(xp.asarray(E))
    theta = xp.atleast_1d(xp.asarray(theta))
    phi = xp.atleast_1d(xp.asarray(phi))
    cos_th = xp.cos(theta)
    sin_th = xp.sin(theta)
    cos_phi = xp.cos(phi)
    k0_k = xp.where(E > 0, 1.0 + (1.0 - cos_th) * E / MEC2, xp.nan)
    k_k0 = 1.0 / k0_k
    k_k0_2 = k_k0 * k_k0

    output = (
        (RE2 / 2.0)
        * k_k0_2
        * (k_k0 + k0_k - 2 * sin_th * sin_th * cos_phi * cos_phi)
    )
    return output, xp.isnan(output).any()


# TODO work out why slow with expand_dims in jit
# TODO some parameter on how to broadcast the arguments
@xrl_xrlnp("Energy must be strictly positive")
def DCSP_KN(E: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> NDArray:
    """
    Klein-Nishina differential scattering cross section
    for polarized beam (barn)

    Parameters
    ----------
    E : array_like
        Energy (keV)
    theta : array_like
        scattering polar angle (rad)
    phi : array_like
        scattering azimuthal angle (rad)

    Returns
    -------
    Array
        Klein-Nishina differential scattering cross section
        for polarized beam (barn)
    """
    # TODO broadcasting of arguments # ???
    return _DCSP_KN(E, theta, phi)
    # if xrl_np:
    #     # TODO some stuff with ndim
    #     E = xp.expand_dims(xp.atleast_1d(xp.asarray(E)), (1, 2))
    #     theta = xp.expand_dims(xp.atleast_1d(xp.asarray(theta)), (0, 2))
    #     phi = xp.expand_dims(xp.atleast_1d(xp.asarray(phi)), (0, 1))
    # return _DCSP_KN(E, theta, phi)


@wrapped_partial(jit, **jit_kwargs)
def _DCSP_Thoms(theta: ArrayLike, phi: ArrayLike) -> NDArray:
    theta = xp.atleast_1d(xp.asarray(theta))
    phi = xp.atleast_1d(xp.asarray(phi))
    sin_th = xp.sin(theta)
    cos_phi = xp.cos(phi)
    return RE2 * (1 - sin_th * sin_th * cos_phi * cos_phi), None


@xrl_xrlnp()
def DCSP_Thoms(theta: ArrayLike, phi: ArrayLike) -> NDArray:
    """
    Thomson differential scattering cross section
    for polarized beam (barn)

    Parameters
    ----------
    theta : array_like
        scattering polar angle (rad)
    phi : array_like
        scattering azimuthal angle (rad)

    Returns
    -------
    Array
        Thomson differential scattering cross section
        for polarized beam (barn)
    """
    # TODO broadcasting of arguments # ???
    return _DCSP_Thoms(theta, phi)
    # if xrl_np:
    #     theta = xp.expand_dims(xp.atleast_1d(xp.asarray(theta)), 1)
    #     phi = xp.expand_dims(xp.atleast_1d(xp.asarray(phi)), 0)
    # return xp.squeeze(_DCSP_Thoms(theta, phi))
