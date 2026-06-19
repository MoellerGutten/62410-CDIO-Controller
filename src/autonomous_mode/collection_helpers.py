from src.autonomous_mode.movement_helpers import go_to, turn_to_point, drive_backward, creep_forward_step
from src.autonomous_mode.state_helpers import update_ball_count_estimate, await_robot
from src.state.state_manager import update_state
from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.lib.connection import RobotConnection
from src.lib.constants import (CROSS_ZONE_BACKWARD_SPEED, CROSS_ZONE_BACKWARD_MS, CROSS_ZONE_VERIFY_RADIUS,
                               CROSS_ZONE_CREEP_STEP_SPEED, CROSS_ZONE_CREEP_STEP_MS, CROSS_ZONE_MAX_CREEP_STEPS,
                               CROSS_ZONE_WAYPOINT_TOLERANCE)
from src.lib.cross_approach_points import get_cross_approach_points
from src.lib.cross_waypoints import get_cross_waypoints
from src.debug.log import get_logger
from math import hypot


def nearest_ball_within(state: ArenaState, point: tuple[float, float], radius: float) -> Ball | None:
    candidates = [b for b in state.balls if b.distance_to_point(point) <= radius]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b.distance_to_point(point))


def escape_cross_zone(state: ArenaState, connection: RobotConnection):
    logger = get_logger("escape_cross_zone")
    waypoints = get_cross_waypoints(state.cross)
    if not waypoints:
        return
    logger.debug("Reversing out of the cross zone")
    drive_backward(state, connection, speed=CROSS_ZONE_BACKWARD_SPEED, ms=CROSS_ZONE_BACKWARD_MS)

    robot = await_robot(state, connection)
    nearest = min(waypoints, key=robot.distance_to_point)
    logger.debug(f"Going to nearest waypoint {nearest}")
    go_to(state, connection, nearest)


def collect_cross_zone_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    logger = get_logger("collect_cross_zone_ball")
    approach_points = get_cross_approach_points(state.cross)
    staging_point = min(approach_points, key=ball.distance_to_point)

    waypoints = get_cross_waypoints(state.cross)
    if waypoints:
        nearest_waypoint = min(waypoints, key=lambda w: hypot(w[0] - staging_point[0], w[1] - staging_point[1]))
        logger.debug(f"Going to nearest waypoint {nearest_waypoint} before staging")
        go_to(state, connection, nearest_waypoint, distance_tolerance=CROSS_ZONE_WAYPOINT_TOLERANCE)

    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point, distance_tolerance=CROSS_ZONE_WAYPOINT_TOLERANCE)

    target = ball.position
    turn_to_point(state, connection, target, True)
    for step in range(CROSS_ZONE_MAX_CREEP_STEPS):
        creep_forward_step(state, connection, ms=CROSS_ZONE_CREEP_STEP_MS, speed=CROSS_ZONE_CREEP_STEP_SPEED)

        update_state(state)
        remaining = nearest_ball_within(state, target, CROSS_ZONE_VERIFY_RADIUS)
        if remaining is None:
            logger.debug("Ball no longer detected near target — collected")
            break
        target = remaining.position
        turn_to_point(state, connection, target, True)
    else:
        logger.warning("Reached max creep steps without confirming collection")

    escape_cross_zone(state, connection)
    update_ball_count_estimate(state)
