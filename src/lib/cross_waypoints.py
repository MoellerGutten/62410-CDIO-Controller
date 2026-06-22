from math import sin, cos, radians
from src.model.cross import Cross
from pygame.math import Vector2
from src.state.arena_config import ArenaConfig
from src.lib.constants import CROSS_WAYPOINT_OFFSET
import numpy as np

# look here to understand this math https://github.com/MoellerGutten/62410-CDIO-Controller/wiki/cross-approach-points

def get_cross_waypoints(cross: Cross) -> list[tuple[float, float]]:
    """
    Calculate and return waypoints to get around the cross
    :param cross: cross
    :return: list[tuple[float, float]] of waypoints
    """
    if cross is None:
        return []

    p0 = np.array([cross.position[0], cross.position[1]])
    heading = np.radians(cross.orientation)
    h = np.array([np.cos(heading), np.sin(heading)])
    h_hat = np.array([-h[1], h[0]])
    d = 20  # fix denne til at være en konstant og juster så den passer med ('kryds arm længde') + (robot bredde + buffer).

    wp1 = p0 + d * h + d * h_hat
    wp2 = p0 + d * h - d * h_hat
    wp3 = p0 - d * h - d * h_hat
    wp4 = p0 - d * h + d * h_hat

    wp1 = tuple(wp1)
    wp2 = tuple(wp2)
    wp3 = tuple(wp3)
    wp4 = tuple(wp4)

    return  [(float(wp[0]), float(wp[1])) for wp in [wp1, wp2, wp3, wp4]]
