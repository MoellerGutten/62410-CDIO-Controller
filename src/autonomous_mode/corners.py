from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.model.robot import Robot
from src.lib.connection import RobotConnection
from src.lib.constants import EAST_HEADING, NORTH_EAST_HEADING, NORTH_HEADING, NORTH_WEST_HEADING, WEST_HEADING, SOUTH_WEST_HEADING, SOUTH_HEADING, SOUTH_EAST_HEADING
from src.model.arena_corner import ArenaCorner
from src.model.arena_edge import ArenaEdge
from src.autonomous_mode.state_helpers import await_robot
from src.debug.log import get_logger
from src.lib.algorithms import backward_speed, backward_ms
from src.autonomous_mode.movement_helpers import go_to, turn_to_heading, burst_backward, drive_backward_speed_ms

def get_staging_data(ball: Ball) -> tuple[tuple[float, float], float, ArenaEdge, float]:
    """Get staging point, staging heading, edge against which to drive to collection, and collection heading"""
    from src.state.arena_config import ArenaConfig
    nearest_corner = ball.nearest_corner()
    nearest_edge = ball.nearest_edge()

    a = 51
    b = 18
    w = ArenaConfig.width_cm
    h = ArenaConfig.height_cm

    match (nearest_corner, nearest_edge):

        # north east corner
        case (ArenaCorner.NORTH_EAST, ArenaEdge.NORTH):
            return ((w - b, h - a), NORTH_WEST_HEADING, ArenaEdge.EAST, NORTH_HEADING)
        case (ArenaCorner.NORTH_EAST, ArenaEdge.EAST):
            return ((w - a, h - b), SOUTH_EAST_HEADING, ArenaEdge.NORTH, EAST_HEADING)

        # north west corner
        case (ArenaCorner.NORTH_WEST, ArenaEdge.NORTH):
            return ((b, h - a), NORTH_EAST_HEADING, ArenaEdge.WEST, NORTH_HEADING)
        case (ArenaCorner.NORTH_WEST, ArenaEdge.WEST):
            return ((a, h - b), SOUTH_WEST_HEADING, ArenaEdge.NORTH, WEST_HEADING)

        # south east corner
        case (ArenaCorner.SOUTH_EAST, ArenaEdge.SOUTH):
            return ((w - b, a), SOUTH_WEST_HEADING, ArenaEdge.EAST, SOUTH_HEADING)
        case (ArenaCorner.SOUTH_EAST, ArenaEdge.EAST):
            return ((w - a, b), NORTH_EAST_HEADING, ArenaEdge.SOUTH, EAST_HEADING)

        # south west corner
        case (ArenaCorner.SOUTH_WEST, ArenaEdge.SOUTH):
            return ((b, a), SOUTH_EAST_HEADING, ArenaEdge.WEST, SOUTH_HEADING)
        case (ArenaCorner.SOUTH_WEST, ArenaEdge.WEST):
            return ((a, b), NORTH_WEST_HEADING, ArenaEdge.SOUTH, WEST_HEADING)

        case _:
            raise ValueError(
                f"No staging rule for corner={nearest_corner}, edge={nearest_edge}"
            )

def back_towards_wall_and_turn(state: ArenaState, connection: RobotConnection, along_edge: ArenaEdge, collection_heading: float):
    logger = get_logger("back_towards_wall_and_turn")

    logger.debug(f"Backing towards edge {along_edge}")
    while True:
        robot = await_robot(state, connection)

        distance_to_wall = robot.distance_to_point(_get_wall_staging_point(robot, along_edge))

        if distance_to_wall < 12:
            logger.debug(f"Reached edge {along_edge}")
            break

        # back to wall
        drive_backward_speed_ms(state, connection, backward_speed(distance_to_wall), backward_ms(distance_to_wall))
    
    logger.debug(f"Turning towards collection heading {collection_heading}")
    turn_to_heading(state, connection, collection_heading)

def _get_wall_staging_point(robot: Robot, along_edge: ArenaEdge):
    from src.state.arena_config import ArenaConfig
    match along_edge:
        case ArenaEdge.NORTH:
            return (robot.position[0], ArenaConfig.height_cm)
        case ArenaEdge.SOUTH:
            return (robot.position[0], 0)
        case ArenaEdge.EAST:
            return (ArenaConfig.width_cm, robot.position[1])
        case ArenaEdge.WEST:
            return (0, robot.position[1])

def advance_to_corner_ball(state: ArenaState, connection: RobotConnection, ball: Ball) -> None:
    go_to(state, connection, ball.position)
    burst_backward(state, connection)
    burst_backward(state, connection)
