from math import sin, cos, radians
from src.model.cross import Cross
from pygame.math import Vector2
from src.state.arena_config import ArenaConfig
from src.lib.constants import CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET, CROSS_APPROACH_POINTS_VERTICAL_OFFSET

# look here to understand this math https://github.com/MoellerGutten/62410-CDIO-Controller/wiki/cross-approach-points

def get_cross_approach_points(cross: Cross) -> list[tuple[float, float]]:
    """Get a list of approach points for the cross."""
    cross = Cross(position=(ArenaConfig.width_cm / 2, ArenaConfig.height_cm / 2), orientation=0.0, bounding_box=[(0,0),(0,0),(0,0),(0,0)]) # for testing
    return [_get_q1_cross_approach_point(cross), _get_q2_cross_approach_point(cross), _get_q3_cross_approach_point(cross), _get_q4_cross_approach_point(cross)]

def _get_q1_cross_approach_point(cross: Cross) -> tuple[float, float]:
    """Get the first quadrant's approach point (corresponds to point 'b' in the diagram)"""
    hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a = _calculate_intermediate_points(cross)
    pos_vector = Vector2(hori_c_to_i + hori_i_to_a, vert_c_to_i - vert_i_to_a)
    correction_vector = Vector2(cross.position[0], cross.position[1])
    corrected_vector = pos_vector + correction_vector # add cross' position to approach point position vector (which is relative to cross' center)
    return (corrected_vector.x, corrected_vector.y) # convert to tuple[float, float]

def _get_q2_cross_approach_point(cross: Cross) -> tuple[float, float]:
    """Get the second quadrant's approach point (corresponds to point 'a' in the diagram)"""
    hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a = _calculate_intermediate_points(cross)
    pos_vector = Vector2(hori_c_to_i - hori_i_to_a, vert_c_to_i + vert_i_to_a)
    correction_vector = Vector2(cross.position[0], cross.position[1])
    corrected_vector = pos_vector + correction_vector # add cross' position to approach point position vector (which is relative to cross' center)
    return (corrected_vector.x, corrected_vector.y) # convert to tuple[float, float]

def _get_q3_cross_approach_point(cross: Cross) -> tuple[float, float]:
    """Get the third quadrant's approach point (corresponds to point 'c' in the diagram)"""
    hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a = _calculate_intermediate_points(cross)
    pos_vector = Vector2(-hori_c_to_i - hori_i_to_a, -vert_c_to_i + vert_i_to_a)
    correction_vector = Vector2(cross.position[0], cross.position[1])
    corrected_vector = pos_vector + correction_vector # add cross' position to approach point position vector (which is relative to cross' center)
    return (corrected_vector.x, corrected_vector.y) # convert to tuple[float, float]

def _get_q4_cross_approach_point(cross: Cross) -> tuple[float, float]:
    """Get the fourth quadrant's approach point (corresponds to point 'd' in the diagram)"""
    hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a = _calculate_intermediate_points(cross)
    pos_vector = Vector2(-hori_c_to_i + hori_i_to_a, -vert_c_to_i - vert_i_to_a)
    correction_vector = Vector2(cross.position[0], cross.position[1])
    corrected_vector = pos_vector + correction_vector # add cross' position to approach point position vector (which is relative to cross' center)
    return (corrected_vector.x, corrected_vector.y) # convert to tuple[float, float]

def _calculate_intermediate_points(cross: Cross) -> tuple[float, float, float, float]:
    """Calculate the distances from the center to the intermediate points and from the intermediate points to the approach points"""
    # distance from cross' center to intermediate points
    hori_c_to_i = CROSS_APPROACH_POINTS_VERTICAL_OFFSET * cos(radians(90 - cross.orientation)) # horizontal distance between cross' center and AB or CD
    vert_c_to_i = CROSS_APPROACH_POINTS_VERTICAL_OFFSET * sin(radians(90 - cross.orientation)) # vertical distance between cross' center and AB or CD

    # distance from intermediate points to approach points
    hori_i_to_a = CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET * cos(radians(cross.orientation))    # horizontal distance from AB or CD to approach point
    vert_i_to_a = CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET * sin(radians(cross.orientation))    # vertical distance from AB or CD to approach point
    return (hori_c_to_i, vert_c_to_i, hori_i_to_a, vert_i_to_a)
