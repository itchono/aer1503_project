MU_EARTH = 3.986004418e14  # in m^3/s^2
R_EARTH = 6378137.0  # in meters


def get_tu(lu: float) -> float:
    """Get time unit (TU) in seconds given length unit (LU) in meters."""
    return (lu**3 / MU_EARTH) ** 0.5
