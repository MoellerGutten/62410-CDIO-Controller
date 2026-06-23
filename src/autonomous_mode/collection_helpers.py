from src.autonomous_mode.movement_helpers import drive_backward_speed_ms, go_to, burst_into_ball, move_slowly_towards_point, turn_to_heading, turn_to_point, burst_backward, handle_balls_in_radius
from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.debug.log import get_logger
from src.lib.connection import RobotConnection
from src.lib.constants import BACKWARD_MS, BACKWARD_SPEED, GO_TO_BALL_EDGE_APPROACH_RADIUS, GO_TO_BALL_APPROACH_RADIUS, EDGE_BALL_GENTLE_BURST_TARGET_RANGE
from src.autonomous_mode.corners import advance_to_corner_ball, get_staging_data, back_towards_wall_and_turn, gentle_burst
from src.autonomous_mode.state_helpers import await_robot, update_ball_count_estimate

def collect_edge_ball(state: ArenaState, ball: Ball, connection: RobotConnection, staging_point: tuple[float, float]):
    go_to(state, connection, staging_point)
    turn_to_point(state, connection, ball.position)
    go_to(state, connection, ball.position, GO_TO_BALL_EDGE_APPROACH_RADIUS)
    gentle_burst(state, connection, ball.position, EDGE_BALL_GENTLE_BURST_TARGET_RANGE, 15)
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
    burst_into_ball(state, connection, ball.position)

def collect_corner_ball(state: ArenaState, connection: RobotConnection, ball: Ball) -> None:
    logger = get_logger("collect_corner_ball")

    staging_point, staging_heading, along_edge, collection_heading, true_corner_target_point = get_staging_data(ball)

    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point)

    is_edge_ball, edge_ball_staging_point = ball.is_edge_ball()
    if ball.distance_to_nearest_corner() > 12 and is_edge_ball and await_robot(state, connection).is_facing_edge(ball.nearest_edge()):
        logger.debug("Wall hugging not needed, proceeding with edge ball collection")
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
    advance_to_corner_ball(state, connection, ball, true_corner_target_point)

def collect_waypoint_zone_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    logger = get_logger("collect_waypoint_zone_ball")
    waypoints = get_cross_waypoints(state.cross)
    staging_point = _get_waypoint_zone_staging_point(waypoints, ball)
    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point)
    logger.debug(f"Turning to ball position {ball.position}")
    turn_to_point(state, connection, ball.position)
    logger.debug("Bursting")
    move_slowly_towards_point(state, connection, ball.position)
    logger.debug("Backing away")
    drive_backward_speed_ms(state, connection, BACKWARD_SPEED, BACKWARD_MS)
    logger.debug("Ball collected")


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