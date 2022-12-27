"""_summary_
"""


def _splint(
    xa: list[float], ya: list[float], y2a: list[float], n: int, x: float
) -> float:
    """
    Perform spline interpolation to calculate an interpolated y value at a given x value.

    Parameters:
        xa (List[float]): Known x values.
        ya (List[float]): Known y values.
        y2a (List[float]): Second derivatives of the y values.
        n (int): Number of values in the input arrays.
        x (float): Target x value at which to calculate the interpolated y value.

    Returns:
        float: Interpolated y value at the target x value, or 0.0 if the target x value is outside the range of the known x values.
    """
    if x - xa[n - 1] > 1e-7:
        return 0.0
    if x < xa[0]:
        return 0.0

    klo = 0
    khi = n - 1
    while khi - klo > 1:
        k = (khi + klo) // 2
        if xa[k] > x:
            khi = k
        else:
            klo = k

    h = xa[khi] - xa[klo]
    if h == 0.0:
        return (ya[klo] + ya[khi]) / 2.0

    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h
    return (
        a * ya[klo]
        + b * ya[khi]
        + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi])
        * (h * h)
        / 6.0
    )
