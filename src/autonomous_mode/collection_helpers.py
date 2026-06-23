from math import hypot, cos, sin, radians

from src.lib.cross_approach_points import get_cross_approach_points
from src.autonomous_mode.movement_helpers import burst_into_ball_slightly_smaller, _start_ball_intake, drive_backward_speed_ms, drive_forward, escape_cross_zone, go_to, burst_into_ball, move_slowly_towards_point, turn_to_heading, turn_to_point, burst_backward, handle_balls_in_radius
from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.model.cross import Cross
from src.debug.log import get_logger
from src.lib.connection import RobotConnection
from src.lib.constants import BACKWARD_MS, BACKWARD_SPEED, CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET, CROSS_APPROACH_POINTS_VERTICAL_OFFSET, CROSS_AVOIDANCE_WAYPOINTS_OFFSET, CROSS_FINAL_APPROACH_HORIZONTAL_OFFSET, CROSS_FINAL_APPROACH_VERTICAL_OFFSET, CROSS_WAYPOINT_OFFSET, CROSS_ZONE_MAX_CREEP_STEPS, CROSS_ZONE_VERIFY_RADIUS, GO_TO_BALL_EDGE_APPROACH_RADIUS, GO_TO_BALL_APPROACH_RADIUS, EDGE_BALL_GENTLE_BURST_TARGET_RANGE, WAYPOINT_ZONE_COLLECTION_RANGE, WAYPOINT_ZONE_COLLECTION_STAGING_POINT_OFFSET
from src.autonomous_mode.corners import advance_to_corner_ball, get_staging_data, back_towards_wall_and_turn, gentle_burst
from src.autonomous_mode.state_helpers import await_robot, update_ball_count_estimate

def collect_edge_ball(state: ArenaState, ball: Ball, connection: RobotConnection, staging_point: tuple[float, float]):
    go_to(state, connection, staging_point)
    turn_to_point(state, connection, ball.position)
    go_to(state, connection, ball.position, GO_TO_BALL_EDGE_APPROACH_RADIUS)
    _start_ball_intake(connection)
    gentle_burst(state, connection, ball.position, EDGE_BALL_GENTLE_BURST_TARGET_RANGE, 20)
    burst_backward(state, connection)

def collect_normal_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    handle_balls_in_radius(state, connection, ball)

    go_to(state, connection, ball.position, approach_radius=GO_TO_BALL_APPROACH_RADIUS)
    robot = await_robot(state, connection)
    if robot.distance_to_point(ball.position) > 28.0: # TODO: adjust and make constant
        # return early if ball is in a galaxy far far away.
        update_ball_count_estimate(state)
        return
    turn_to_point(state, connection, ball.position, precise_mode=True)
    _start_ball_intake(connection)
    burst_into_ball(state, connection, ball.position)

    if ball.is_within_waypoint_zone(state):
        escape_cross_zone(state, connection)

def collect_corner_ball(state: ArenaState, connection: RobotConnection, ball: Ball) -> None:
    logger = get_logger("collect_corner_ball")

    staging_point, staging_heading, along_edge, collection_heading, true_corner_target_point = get_staging_data(ball)

    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point)

    is_edge_ball, edge_ball_staging_point = ball.is_edge_ball()
    if ball.distance_to_nearest_corner() > 12 and is_edge_ball and await_robot(state, connection).is_facing_edge(ball.nearest_edge()):
        logger.debug("Wall hugging not needed, proceeding with edge ball collection")
        _start_ball_intake(connection)
        collect_edge_ball(state, ball, connection, edge_ball_staging_point)
        return
    
    logger.debug(f"Turning to staging heading {staging_heading}")
    turn_to_heading(state, connection, staging_heading)

    logger.debug("Backing to wall and adjusting heading")
    try:
        # backing towards wall can throw if heading is incorrect and max attempts is reached
        back_towards_wall_and_turn(state, connection, along_edge, collection_heading)
    except:
        logger.error("Failed to collect corner ball, unable to drive backwards to edge. Giving up:(")
        return

    logger.debug("Going to collect ball")
    _start_ball_intake(connection)
    advance_to_corner_ball(state, connection, ball, true_corner_target_point)

def collect_waypoint_zone_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    logger = get_logger("collect_waypoint_zone_ball")
    inflated_waypoint_points = state.cross.inflate_bounding_box(inflation_cm=CROSS_AVOIDANCE_WAYPOINTS_OFFSET+10)
    target = min(inflated_waypoint_points, key=ball.distance_to_point)

    handle_balls_in_radius(state, connection, ball)

    if state.cross is not None and _is_on_same_cross_side(state.robot.position, ball.position, state.cross):
        logger.debug("Robot and ball are on the same side of the cross; skipping waypoint staging.")
    else:
        logger.debug(f"Going to staging point at {target}")
        go_to(state, connection, target)

    turn_to_point(state, connection, ball.position)
    go_to(state, connection, ball.position, approach_radius=GO_TO_BALL_APPROACH_RADIUS)
    robot = await_robot(state, connection)
    if robot.distance_to_point(ball.position) > 28.0: # TODO: adjust and make constant
        # return early if ball is in a galaxy far far away.
        update_ball_count_estimate(state)
        return
    turn_to_point(state, connection, ball.position, precise_mode=True)
    _start_ball_intake(connection)
    burst_into_ball_slightly_smaller(state, connection, ball.position)
    escape_cross_zone(state, connection)


def _is_on_same_cross_side(robot_position: tuple[float, float], ball_position: tuple[float, float], cross: Cross) -> bool:
    if cross is None:
        return False

    cx, cy = cross.position
    heading = radians(cross.orientation - 90)
    axis_x = (cos(heading), sin(heading))
    axis_y = (-axis_x[1], axis_x[0])

    def quadrant_signs(point: tuple[float, float]) -> tuple[int, int] | None:
        dx = point[0] - cx
        dy = point[1] - cy
        proj_x = dx * axis_x[0] + dy * axis_x[1]
        proj_y = dx * axis_y[0] + dy * axis_y[1]
        epsilon = 1e-6
        if abs(proj_x) < epsilon or abs(proj_y) < epsilon:
            return None
        return (1 if proj_x > 0 else -1, 1 if proj_y > 0 else -1)

    robot_quad = quadrant_signs(robot_position)
    ball_quad = quadrant_signs(ball_position)
    if robot_quad is None or ball_quad is None:
        return False

    return robot_quad == ball_quad


def _get_waypoint_zone_staging_point(
    corners: list[tuple[float, float]],
    ball: Ball,
) -> tuple[float, float]:
    """
    Find the bbox edge closest to `ball`'s position, then return a staging point at
    the midpoint of that edge, pushed outward along the edge's
    normal (away from the box center).
    """
    n = len(corners)
    cx = sum(c[0] for c in corners) / n
    cy = sum(c[1] for c in corners) / n

    best_dist_sq = None
    best_mid = None
    best_normal = None

    for i in range(n):
        a = corners[i]
        b = corners[(i + 1) % n]

        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

        # distance from point to this edge's line (closest point on segment)
        ax, ay = a
        bx, by = b
        abx, aby = bx - ax, by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq == 0:
            continue  # degenerate edge, skip

        t = ((ball.position[0] - ax) * abx + (ball.position[1] - ay) * aby) / ab_len_sq
        t = max(0.0, min(1.0, t))
        closest = (ax + t * abx, ay + t * aby)
        dist_sq = (closest[0] - ball.position[0]) ** 2 + (closest[1] - ball.position[1]) ** 2

        if best_dist_sq is None or dist_sq < best_dist_sq:
            # outward normal: perpendicular to edge, pointing away from centroid
            nx, ny = -aby, abx
            norm_len = hypot(nx, ny)
            nx, ny = nx / norm_len, ny / norm_len

            if (mid[0] - cx) * nx + (mid[1] - cy) * ny < 0:
                nx, ny = -nx, -ny

            best_dist_sq = dist_sq
            best_mid = mid
            best_normal = (nx, ny)

    return (
        best_mid[0] + best_normal[0] * WAYPOINT_ZONE_COLLECTION_STAGING_POINT_OFFSET,
        best_mid[1] + best_normal[1] * WAYPOINT_ZONE_COLLECTION_STAGING_POINT_OFFSET,
    )


def nearest_ball_within(state: ArenaState, point: tuple[float, float], radius: float) -> Ball | None:
    candidates = [b for b in state.balls if b.distance_to_point(point) <= radius]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b.distance_to_point(point))

def distance_between_points(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

def collect_cross_zone_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    logger = get_logger("collect_cross_zone_ball")
    approach_points = get_cross_approach_points(state.cross, CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET, CROSS_APPROACH_POINTS_VERTICAL_OFFSET)
    staging_point = min(approach_points, key=ball.distance_to_point)

    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point)
    _start_ball_intake(connection)
    final_points = get_cross_approach_points(state.cross, CROSS_FINAL_APPROACH_HORIZONTAL_OFFSET, CROSS_FINAL_APPROACH_VERTICAL_OFFSET)
    target = min(final_points, key=ball.distance_to_point)
    with state.lock:
        state.target_point = target

    for step in range(CROSS_ZONE_MAX_CREEP_STEPS):
        if state.robot.distance_to_point(target) > 8:
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
