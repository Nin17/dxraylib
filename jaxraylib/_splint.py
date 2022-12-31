"""_summary_
"""

from __future__ import annotations

from .config import jit, xp


# TODO docstring
@jit
def _splint(data, x: float) -> float:
    """
    Perform cubic spline interpolation to calculate an interpolated y value at
    a given x value.


    Parameters
    ----------
    data : xp.ndarray
        Array with the known x, y, and second derivative values stacked along
        the first axis.
    x : float
        Target x value at which to calculate the interpolated y value.

    Returns
    -------
    float
        Interpolated y value at the target x value
    """
    klo = xp.searchsorted(data[0], x) - 1

    h = data[0][klo + 1] - data[0][klo]

    # TODO check if this is actually necessary
    # if h == 0.0:
    #     return (data[1][klo] + data[1][klo + 1]) / 2.0

    a = (data[0][klo + 1] - x) / h
    b = (x - data[0][klo]) / h
    return (
        a * data[1][klo]
        + b * data[1][klo + 1]
        + ((a * a * a - a) * data[2][klo] + (b * b * b - b) * data[2][klo + 1])
        * (h * h)
        / 6.0
    )
