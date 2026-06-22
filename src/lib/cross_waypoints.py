from math import sin, cos, radians
from src.model.cross import Cross
from pygame.math import Vector2
from src.state.arena_config import ArenaConfig
from src.lib.constants import CROSS_WAYPOINT_OFFSET
import numpy as np

# look here to understand this math https://github.com/MoellerGutten/62410-CDIO-Controller/wiki/cross-approach-points

def get_cross_waypoints(cross: Cross) -> list[tuple[float, float]]:
    """Get a list of approach points for the cross."""
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


    # return [_get_tl_cross_approach_point(cross), _get_tr_cross_approach_point(cross), _get_br_cross_approach_point(cross), _get_bl_cross_approach_point(cross)]
#
# def _get_tl_cross_approach_point(cross: Cross) -> tuple[float, float]:
#     """Get the first quadrant's approach point (corresponds to point 'b' in the diagram)"""
#     hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a = _calculate_intermediate_points(cross)
#     pos_vector = Vector2(hori_c_to_i + hori_i_to_a, vert_c_to_i - vert_i_to_a)
#     correction_vector = Vector2(cross.position[0], cross.position[1])
#     corrected_vector = pos_vector + correction_vector # add cross' position to approach point position vector (which is relative to cross' center)
#     return (corrected_vector.x, corrected_vector.y) # convert to tuple[float, float]
#
# def _get_tr_cross_approach_point(cross: Cross) -> tuple[float, float]:
#     """Get the second quadrant's approach point (corresponds to point 'a' in the diagram)"""
#     hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a = _calculate_intermediate_points(cross)
#     pos_vector = Vector2(hori_c_to_i - hori_i_to_a, vert_c_to_i + vert_i_to_a)
#     correction_vector = Vector2(cross.position[0], cross.position[1])
#     corrected_vector = pos_vector + correction_vector # add cross' position to approach point position vector (which is relative to cross' center)
#     return (corrected_vector.x, corrected_vector.y) # convert to tuple[float, float]
#
# def _get_br_cross_approach_point(cross: Cross) -> tuple[float, float]:
#     """Get the third quadrant's approach point (corresponds to point 'c' in the diagram)"""
#     hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a = _calculate_intermediate_points(cross)
#     pos_vector = Vector2(-hori_c_to_i - hori_i_to_a, -vert_c_to_i + vert_i_to_a)
#     correction_vector = Vector2(cross.position[0], cross.position[1])
#     corrected_vector = pos_vector + correction_vector # add cross' position to approach point position vector (which is relative to cross' center)
#     return (corrected_vector.x, corrected_vector.y) # convert to tuple[float, float]
#
# def _get_bl_cross_approach_point(cross: Cross) -> tuple[float, float]:
#     """Get the fourth quadrant's approach point (corresponds to point 'd' in the diagram)"""
#     hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a = _calculate_intermediate_points(cross)
#     pos_vector = Vector2(-hori_c_to_i + hori_i_to_a, -vert_c_to_i - vert_i_to_a)
#     correction_vector = Vector2(cross.position[0], cross.position[1])
#     corrected_vector = pos_vector + correction_vector # add cross' position to approach point position vector (which is relative to cross' center)
#     return (corrected_vector.x, corrected_vector.y) # convert to tuple[float, float]
#
# def _calculate_intermediate_points(cross: Cross) -> tuple[float, float, float, float]:
#     """Calculate the distances from the center to the intermediate points and from the intermediate points to the approach points"""
#     # distance from cross' center to intermediate points
#     hori_c_to_i = CROSS_WAYPOINT_OFFSET * cos(radians(90 - cross.orientation)) # horizontal distance between cross' center and AB or CD
#     vert_c_to_i = CROSS_WAYPOINT_OFFSET * sin(radians(90 - cross.orientation)) # vertical distance between cross' center and AB or CD
#
#     # distance from intermediate points to approach points
#     hori_i_to_a = CROSS_WAYPOINT_OFFSET * cos(radians(cross.orientation))    # horizontal distance from AB or CD to approach point
#     vert_i_to_a = CROSS_WAYPOINT_OFFSET * sin(radians(cross.orientation))    # vertical distance from AB or CD to approach point
#     return (hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a)
