"""_summary_
"""
# TODO summary
# TODO docstrings

from . import _interpolate, _load, _utilities, config as cfg


@_utilities.asarray()
def ComptonProfile(Z: cfg.ArrayLike, pz: cfg.ArrayLike) -> cfg.Array:
    """_summary_

    Parameters
    ----------
    Z : array_like
        _description_
    pz : array_like
        _description_

    Returns
    -------
    array
        _description_
    """
    return cfg.xp.exp(
        _interpolate.interpolate1d(
            _load.load("compton_profiles"), Z, pz, cfg.xp.log(pz + 1)
        )
    )


@_utilities.asarray()
def ComptonProfile_Partial(
    Z: cfg.ArrayLike, shell: cfg.ArrayLike, pz: cfg.ArrayLike
) -> cfg.Array:
    """_summary_

    Parameters
    ----------
    Z : ArrayLike
        _description_
    shell : ArrayLike
        _description_
    pz : ArrayLike
        _description_

    Returns
    -------
    Array
        _description_
    """
    return cfg.xp.exp(
        _interpolate.interpolate2d(
            _load.load("compton_profiles_partial"),
            Z,
            shell,
            pz,
            cfg.xp.log(pz + 1),
        )
    )
