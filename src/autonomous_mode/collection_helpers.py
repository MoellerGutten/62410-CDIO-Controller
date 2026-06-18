from src.autonomous_mode.movement_helpers import go_to, drive_forward, turn_to_point, drive_backward, move_slowly_towards_point
from src.autonomous_mode.state_helpers import update_ball_count_estimate, await_robot
from src.state.state_manager import update_state
from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.lib.connection import RobotConnection
from src.lib.constants import (CROSS_ZONE_BACKWARD_SPEED, CROSS_ZONE_BACKWARD_MS,
                               CROSS_ZONE_VERIFY_RADIUS, CROSS_ZONE_MAX_ATTEMPTS)
from src.lib.cross_approach_points import get_cross_approach_points
from src.lib.cross_waypoints import get_cross_waypoints
from src.debug.log import get_logger


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
    robot = await_robot(state, connection)
    nearest = min(waypoints, key=robot.distance_to_point)
    logger.debug(f"Escaping cross zone towards nearest waypoint {nearest}")
    turn_to_point(state, connection, nearest, True)
    # move_slowly_towards_point(state, connection, nearest)
    # go_to(state, connection, nearest)


def collect_cross_zone_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    logger = get_logger("collect_cross_zone_ball")
    approach_points = get_cross_approach_points(state.cross)
    staging_point = min(approach_points, key=ball.distance_to_point)
    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point)

    target = ball.position
    for attempt in range(CROSS_ZONE_MAX_ATTEMPTS):
        logger.debug(f"Attempt {attempt + 1}/{CROSS_ZONE_MAX_ATTEMPTS}: turning to {target}")
        turn_to_point(state, connection, target, True)
        if attempt == 0:
            logger.debug("Driving to ball")
            drive_forward(state, connection, target)
        else:
            logger.debug("Creeping slowly to ball")
            move_slowly_towards_point(state, connection, target)
        logger.debug("Backing away slowly")
        drive_backward(state, connection, speed=CROSS_ZONE_BACKWARD_SPEED, ms=CROSS_ZONE_BACKWARD_MS)

        update_state(state)
        remaining = nearest_ball_within(state, target, CROSS_ZONE_VERIFY_RADIUS)
        if remaining is None:
            logger.debug("Ball collected")
            break
        logger.debug(f"Ball still present at {remaining.position}, retrying")
        target = remaining.position
    else:
        logger.warning("Gave up collecting cross-zone ball after max attempts")

    escape_cross_zone(state, connection)
    update_ball_count_estimate(state)
