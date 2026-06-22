from src.autonomous_mode.movement_helpers import go_to, turn_to_point, drive_backward, creep_forward_step, \
    escape_cross_zone, drive_forward
from src.autonomous_mode.state_helpers import update_ball_count_estimate, await_robot

from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.lib.connection import RobotConnection
from src.lib.constants import (CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET, CROSS_APPROACH_POINTS_VERTICAL_OFFSET, CROSS_FINAL_APPROACH_HORIZONTAL_OFFSET, CROSS_FINAL_APPROACH_VERTICAL_OFFSET, CROSS_ZONE_VERIFY_RADIUS,CROSS_ZONE_CREEP_STEP_SPEED, CROSS_ZONE_CREEP_STEP_MS,
                               CROSS_ZONE_MAX_CREEP_STEPS, CROSS_ZONE_WAYPOINT_TOLERANCE,
                               CROSS_WAYPOINT_DIAGONAL_EXTENSION)
from src.lib.cross_approach_points import get_cross_approach_points
from src.lib.cross_waypoints import get_cross_waypoints
from src.debug.log import get_logger
from math import hypot


def nearest_ball_within(state: ArenaState, point: tuple[float, float], radius: float) -> Ball | None:
    candidates = [b for b in state.balls if b.distance_to_point(point) <= radius]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b.distance_to_point(point))

def _push_outward(center: tuple[float, float], point: tuple[float, float], distance: float) -> tuple[float, float]:
    """Move `point` `distance` cm further from `center`, along the center->point direction."""
    dx, dy = point[0] - center[0], point[1] - center[1]
    length = hypot(dx, dy)
    if length == 0:
        return point
    scale = (length + distance) / length
    return (center[0] + dx * scale, center[1] + dy * scale)

def collect_cross_zone_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    logger = get_logger("collect_cross_zone_ball")
    approach_points = get_cross_approach_points(state.cross, CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET, CROSS_APPROACH_POINTS_VERTICAL_OFFSET)
    staging_point = min(approach_points, key=ball.distance_to_point)

    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point, approach_radius=CROSS_ZONE_WAYPOINT_TOLERANCE)
    final_points = get_cross_approach_points(state.cross, CROSS_FINAL_APPROACH_HORIZONTAL_OFFSET, CROSS_FINAL_APPROACH_VERTICAL_OFFSET)
    target = min(final_points, key=ball.distance_to_point)
    with state.lock:
        state.target_point = target

    for step in range(CROSS_ZONE_MAX_CREEP_STEPS):
        if state.robot.distance_to_point(target) > 10:
            logger.debug("Inching towards ball, iteration: " + str(step))

            turn_to_point(state, connection, target, precise_mode=True)
            drive_forward(state, connection, target)

            #creep_forward_step(state, connection, ms=CROSS_ZONE_CREEP_STEP_MS, speed=CROSS_ZONE_CREEP_STEP_SPEED)

            await_robot(state, connection)
            remaining = nearest_ball_within(state, target, CROSS_ZONE_VERIFY_RADIUS)
            if remaining is None:
                logger.debug("Ball no longer detected near target — collected")
                break
        else:
            logger.debug("Reached target — stopping")
            break
    else:
        logger.warning("Reached max creep steps without confirming collection")

    escape_cross_zone(state, connection)
    update_ball_count_estimate(state)
