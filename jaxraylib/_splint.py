"""_summary_
"""

from .config import jit, xp


# TODO docstring
@jit
def _splint(data, x: float) -> float:
    """
    Perform spline interpolation to calculate an interpolated y value at a given x value.

    Parameters:
        xa (List[float]): Known x values.
        ya (List[float]): Known y values.
        y2a (List[float]): Second derivatives of the y values.
        x (float): Target x value at which to calculate the interpolated y value.

    Returns:
        float: Interpolated y value at the target x value, or 0.0 if the target x value is outside the range of the known x values.
    """
    # xa, ya, y2a = data
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
