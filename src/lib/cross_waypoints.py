from src.debug.log import get_logger
from src.model.cross import Cross
from numpy import sin, cos, radians, array
from src.lib.constants import CROSS_WAYPOINT_OFFSET


def get_cross_waypoints(cross: Cross) -> list[tuple[float, float]]:
    """
    Calculate and return waypoints to get around the cross
    :param cross: cross
    :return: list[tuple[float, float]] of waypoints
    """
    if cross is None:
        return []

    p0 = array([cross.position[0], cross.position[1]])
    heading = radians(cross.orientation - 90)
    h = array([cos(heading), sin(heading)])
    h_hat = array([-h[1], h[0]])
    d = CROSS_WAYPOINT_OFFSET

    wp1 = p0 + d * h + d * h_hat
    wp2 = p0 + d * h - d * h_hat
    wp3 = p0 - d * h - d * h_hat
    wp4 = p0 - d * h + d * h_hat

    return  [(float(wp[0]), float(wp[1])) for wp in [wp1, wp2, wp3, wp4]]
