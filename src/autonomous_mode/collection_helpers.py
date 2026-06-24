from math import hypot, cos, sin, radians
from src.lib.cross_approach_points import get_cross_approach_points
from src.autonomous_mode.movement_helpers import burst_into_ball_slightly_smaller, _start_ball_intake, escape_cross_zone, gentle_burst, go_to, burst_into_ball, \
turn_to_heading, turn_to_point, burst_backward, handle_balls_in_radius,gentle_burst_edge_ball
from src.model.arena_state import ArenaState
from src.model.arena_edge import get_other_corner_edge
from src.model.ball import Ball
from src.model.cross import Cross
from src.debug.log import get_logger
from src.lib.connection import RobotConnection
from src.lib.constants import CROSS_APPROACH_POINTS_HORIZONTAL_OFFSET, CROSS_APPROACH_POINTS_VERTICAL_OFFSET, \
    CROSS_AVOIDANCE_WAYPOINTS_OFFSET, \
    CROSS_FINAL_APPROACH_HORIZONTAL_OFFSET, CROSS_FINAL_APPROACH_VERTICAL_OFFSET, CROSS_ZONE_MAX_CREEP_STEPS, \
    CROSS_ZONE_VERIFY_RADIUS, \
    GO_TO_BALL_EDGE_APPROACH_RADIUS, GO_TO_BALL_APPROACH_RADIUS, EDGE_BALL_GENTLE_BURST_HORIZONTAL, \
    EDGE_BALL_GENTLE_BURST_VERTICAL, CROSS_ZONE_FINAL_POINT_DIST_TOLERANCE
from src.autonomous_mode.corners import advance_to_corner_ball, get_staging_data, back_towards_wall_and_turn
from src.autonomous_mode.state_helpers import await_robot
from src.model.arena_edge import ArenaEdge


def collect_edge_ball(state: ArenaState, ball: Ball, connection: RobotConnection, staging_point: tuple[float, float]):
    logger = get_logger("collect_edge_ball")

    logger.debug("Collecting edge ball, going to staging point")
    go_to(state, connection, staging_point)

    logger.debug("Turning to ball position")
    turn_to_point(state, connection, ball.position)

    logger.debug("Going to ball position")
    go_to(state, connection, ball.position, GO_TO_BALL_EDGE_APPROACH_RADIUS)

    logger.debug("Turning to ball position")
    turn_to_point(state, connection, ball.position, precise_mode=True)

    _start_ball_intake(connection)

    if ball.nearest_edge() in [ArenaEdge.NORTH, ArenaEdge.SOUTH]:
        logger.debug("Gentle burst towards horizontal edge")
        with state.lock:
            state.target_point = ball.position # set the target point in the GUI
        gentle_burst_edge_ball(state, connection, ball.position, EDGE_BALL_GENTLE_BURST_HORIZONTAL)
    else:
        logger.debug("Gentle burst towards vertical edge")
        with state.lock:
            state.target_point = ball.position # set the target point in the GUI
        gentle_burst_edge_ball(state, connection, ball.position, EDGE_BALL_GENTLE_BURST_VERTICAL)

    logger.debug("Backing off from edge")
    burst_backward(state, connection)


def collect_normal_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    logger = get_logger("collect_normal_ball")

    logger.debug("Collecting normal ball, first checking if ball is in radius")
    robot = await_robot(state, connection)
    handle_balls_in_radius(state, robot, connection, ball) # reposition robot before collection if the ball is directly behind or to the side of the robot

    logger.debug("Going to ball position")
    go_to(state, connection, ball.position, approach_radius=GO_TO_BALL_APPROACH_RADIUS)
    if robot.distance_to_point(ball.position) > 28.0: # TODO: adjust and make constant
        # return early if ball is in a galaxy far far away.
        logger.debug("Ball is in a galaxy far, far away")
        return
    logger.debug("Turning to ball position")
    turn_to_point(state, connection, ball.position, precise_mode=True)
    _start_ball_intake(connection)
    logger.debug("Bursting into ball")
    burst_into_ball(state, connection, ball.position)
    burst_backward(state, connection)

    if ball.is_within_waypoint_zone(state):
        logger.debug("Ball is within waypoint zone, escaping zone")
        escape_cross_zone(state, connection)

def collect_corner_ball(state: ArenaState, connection: RobotConnection, ball: Ball) -> None:
    logger = get_logger("collect_corner_ball")

    staging_point, staging_heading, along_edge, collection_heading, true_corner_target_point = get_staging_data(ball)

    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point)

    is_edge_ball, edge_ball_staging_point = ball.is_edge_ball()
    if ball.distance_to_nearest_corner() > 12 and is_edge_ball:
        robot = await_robot(state, connection)
        ball_nearest_edge = ball.nearest_edge()
        ball_nearest_corner = ball.nearest_corner()

        # this corner ball is a candidate for being treated as an edge ball, its far from the corner along its adjacent edge
        if robot.is_facing_edge(ball_nearest_edge):
            logger.debug("Wall hugging not needed, robot already facing ball and ball far from corner along ball's adjacent edge. Treating ball as edge ball")
            _start_ball_intake(connection)
            collect_edge_ball(state, ball, connection, edge_ball_staging_point)
            return

        if robot.is_facing_edge(get_other_corner_edge(ball_nearest_corner, ball_nearest_edge)):
            logger.debug("Wall hugging not needed, turning and treating ball as edge ball")
            burst_backward(state, connection)
            turn_to_point(state, connection, edge_ball_staging_point, precise_mode=False)
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
    logger.debug("Collecting waypoint zone ball")

    robot = await_robot(state, connection)

    inflated_waypoint_points = state.cross.inflate_bounding_box(inflation_cm=CROSS_AVOIDANCE_WAYPOINTS_OFFSET+10)
    target = min(inflated_waypoint_points, key=ball.distance_to_point)

    handle_balls_in_radius(state, robot, connection, ball)

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
        logger.debug("Robot is in a galaxy far, far away")
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


def nearest_ball_within(state: ArenaState, point: tuple[float, float], radius: float) -> Ball | None:
    candidates = [b for b in state.balls if b.distance_to_point(point) <= radius]
    if not candidates:
        return None
    return min(candidates, key=lambda b: b.distance_to_point(point))

def distance_between_points(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

def collect_cross_zone_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    logger = get_logger("collect_cross_zone_ball")
    logger.debug("Collecting cross zone ball")

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
        if state.robot.distance_to_point(target) > 8 and ball.is_within_cross_zone(state.cross):
            logger.debug("Inching towards ball, iteration: " + str(step))

            turn_to_point(state, connection, target, precise_mode=True)
            gentle_burst(state, connection, target, CROSS_ZONE_FINAL_POINT_DIST_TOLERANCE)

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
