from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.lib.connection import RobotConnection
from src.lib.constants import SOUTH_HEADING
from src.model.arena_corner import ArenaCorner
from src.model.arena_edge import ArenaEdge

def get_staging_point_and_heading(ball: Ball) -> tuple[tuple[float, float], float]:
    nearest_corner = ball.nearest_corner()
    nearest_edge = ball.nearest_edge()

    match (nearest_corner, nearest_edge):

        # north east corner
        case (ArenaCorner.NORTH_EAST, ArenaEdge.NORTH):
            return ((0.0, 0.0), 0.0)
        case (ArenaCorner.NORTH_EAST, ArenaEdge.EAST):
            return ((0.0, 0.0), 0.0)

        # north west corner
        case (ArenaCorner.NORTH_WEST, ArenaEdge.NORTH):
            return ((0.0, 0.0), 0.0)
        case (ArenaCorner.NORTH_WEST, ArenaEdge.WEST):
            return ((0.0, 0.0), 0.0)

        # south east corner
        case (ArenaCorner.SOUTH_EAST, ArenaEdge.SOUTH):
            return ((0.0, 0.0), 0.0)
        case (ArenaCorner.SOUTH_EAST, ArenaEdge.EAST):
            return ((0.0, 0.0), 0.0)

        # south west corner
        case (ArenaCorner.SOUTH_WEST, ArenaEdge.SOUTH):
            return ((0.0, 0.0), 0.0)
        case (ArenaCorner.SOUTH_WEST, ArenaEdge.WEST):
            return ((51, 21), SOUTH_HEADING - 45)

        case _:
            raise ValueError(
                f"No staging rule for corner={nearest_corner}, edge={nearest_edge}"
            )

def approach_ball_while_turning(state: ArenaState, connection: RobotConnection, ballBall) -> None:
    pass

def advance_to_corner_ball(state: ArenaState, connection: RobotConnection, ball: Ball) -> None:
    pass
