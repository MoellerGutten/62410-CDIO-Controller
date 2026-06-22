from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.model.robot import Robot
from src.lib.connection import RobotConnection
from src.lib.constants import EAST_HEADING, NORTH_EAST_HEADING, NORTH_HEADING, NORTH_WEST_HEADING, WEST_HEADING, \
SOUTH_WEST_HEADING, SOUTH_HEADING, SOUTH_EAST_HEADING, ARENA_WIDTH_CM, ARENA_HEIGHT_CM, SLEEP_BUFFER_SECONDS, \
BACK_TOWARDS_EDGE_MAX_ITERATIONS, WALL_THRESHOLD, CORNER_BALL_COLLECTION_APPROACH_RADIUS, CORNER_COLLECTION_STAGING_POINT
from src.model.arena_corner import ArenaCorner
from src.model.arena_edge import ArenaEdge
from src.autonomous_mode.state_helpers import await_robot
from src.debug.log import get_logger
from src.lib.algorithms import backward_speed, backward_ms
from time import sleep
from src.lib.time import ms_to_seconds
from src.autonomous_mode.movement_helpers import go_to, turn_to_heading, burst_backward, drive_backward_speed_ms, burst_into_ball

def get_staging_data(ball: Ball) -> tuple[tuple[float, float], float, ArenaEdge, float, tuple[float, float]]:
    """Get staging point, staging heading, edge against which to drive to collection, collection heading, and target point for go_to in true corner cases"""
    nearest_corner = ball.nearest_corner()
    nearest_edge = ball.nearest_edge()

    a, b = CORNER_COLLECTION_STAGING_POINT
    w = ARENA_WIDTH_CM
    h = ARENA_HEIGHT_CM

    assbitch = 7

    match (nearest_corner, nearest_edge):

        # north east corner
        case (ArenaCorner.NORTH_EAST, ArenaEdge.NORTH):
            return ((w - b, h - a), NORTH_WEST_HEADING, ArenaEdge.EAST, NORTH_HEADING, (w - assbitch, h))
        case (ArenaCorner.NORTH_EAST, ArenaEdge.EAST):
            return ((w - a, h - b), SOUTH_EAST_HEADING, ArenaEdge.NORTH, EAST_HEADING, (w, h - assbitch))

        # north west corner
        case (ArenaCorner.NORTH_WEST, ArenaEdge.NORTH):
            return ((b, h - a), NORTH_EAST_HEADING, ArenaEdge.WEST, NORTH_HEADING, (0, h - assbitch))
        case (ArenaCorner.NORTH_WEST, ArenaEdge.WEST):
            return ((a, h - b), SOUTH_WEST_HEADING, ArenaEdge.NORTH, WEST_HEADING, (assbitch, h))

        # south east corner
        case (ArenaCorner.SOUTH_EAST, ArenaEdge.SOUTH):
            return ((w - b, a), SOUTH_WEST_HEADING, ArenaEdge.EAST, SOUTH_HEADING, (w - assbitch, 0))
        case (ArenaCorner.SOUTH_EAST, ArenaEdge.EAST):
            return ((w - a, b), NORTH_EAST_HEADING, ArenaEdge.SOUTH, EAST_HEADING, (w, assbitch))

        # south west corner
        case (ArenaCorner.SOUTH_WEST, ArenaEdge.SOUTH):
            return ((b, a), SOUTH_EAST_HEADING, ArenaEdge.WEST, SOUTH_HEADING, (assbitch, 0))
        case (ArenaCorner.SOUTH_WEST, ArenaEdge.WEST):
            return ((a, b), NORTH_WEST_HEADING, ArenaEdge.SOUTH, WEST_HEADING, (0, assbitch))

        case _:
            raise ValueError(
                f"No staging rule for corner={nearest_corner}, edge={nearest_edge}"
            )

def back_towards_wall_and_turn(state: ArenaState, connection: RobotConnection, along_edge: ArenaEdge, collection_heading: float):
    logger = get_logger("back_towards_wall_and_turn")

    logger.debug(f"Backing towards edge {along_edge}")

    _iter = 0
    while True:
        robot = await_robot(state, connection)

        distance_to_wall = robot.distance_to_point(_get_wall_staging_point(robot, along_edge))

        if _iter >= BACK_TOWARDS_EDGE_MAX_ITERATIONS:
            msg = "Reached max attempts for backing towards wall"
            logger.error(msg)
            raise RuntimeError(msg)

        if distance_to_wall < WALL_THRESHOLD:
            logger.debug(f"Reached edge {along_edge}")
            break

        # back to wall
        ms = backward_ms(distance_to_wall)
        speed = backward_speed(distance_to_wall)
        drive_backward_speed_ms(state, connection, speed, ms)
        sleep(ms_to_seconds(ms) + SLEEP_BUFFER_SECONDS)
        _iter += 1
    
    logger.debug(f"Turning towards collection heading {collection_heading}")
    turn_to_heading(state, connection, collection_heading)

def _get_wall_staging_point(robot: Robot, along_edge: ArenaEdge):
    match along_edge:
        case ArenaEdge.NORTH:
            return (robot.position[0], ARENA_HEIGHT_CM)
        case ArenaEdge.SOUTH:
            return (robot.position[0], 0)
        case ArenaEdge.EAST:
            return (ARENA_WIDTH_CM, robot.position[1])
        case ArenaEdge.WEST:
            return (0, robot.position[1])

def advance_to_corner_ball(state: ArenaState, connection: RobotConnection, ball: Ball, true_corner_target_point: tuple[float, float]) -> None:
    if ball.distance_to_nearest_corner() < 3: # magic value
        # true corner ball
        go_to(state, connection, true_corner_target_point, approach_radius=CORNER_BALL_COLLECTION_APPROACH_RADIUS)
        burst_into_ball(state, connection, true_corner_target_point)
        burst_backward(state, connection)
        burst_backward(state, connection)
    else:
        # not close to both edges
        go_to(state, connection, ball.position, approach_radius=CORNER_BALL_COLLECTION_APPROACH_RADIUS)
        burst_into_ball(state, connection, ball.position)
        burst_backward(state, connection)
        burst_backward(state, connection)
