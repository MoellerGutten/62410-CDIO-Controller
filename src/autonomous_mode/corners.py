from time import sleep
from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.lib.connection import RobotConnection
from src.lib.constants import SOUTH_HEADING, SLEEP_BUFFER_SECONDS
from src.model.arena_corner import ArenaCorner
from src.model.arena_edge import ArenaEdge
from src.autonomous_mode.state_helpers import await_robot
from protocol import Instruction, Message, Arguments, CommandName, InstructionType
from src.debug.log import get_logger
from src.state.arena_config import ArenaConfig
from src.autonomous_mode.movement_helpers import burst_into_ball, drive_backward

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
            return ((18, 51), SOUTH_HEADING - 45)
        case (ArenaCorner.SOUTH_WEST, ArenaEdge.WEST):
            return ((51, 18), SOUTH_HEADING - 45)

        case _:
            raise ValueError(
                f"No staging rule for corner={nearest_corner}, edge={nearest_edge}"
            )

def approach_ball_while_turning(state: ArenaState, connection: RobotConnection, ball: Ball):
    logger = get_logger("approach_ball_while_turning")
    bx, by = ball.position
    while True:
        robot = await_robot(state, connection)

        distance = robot.distance_to_point(ball.position)

        if distance < 20: # approach stop distance
            logger.debug("reached approach stop distance, stopping")
            break

        edge = ball.nearest_edge()
        corner = ball.nearest_corner()

        # --------------------------------------------------
        # 1. Build constrained target (slide along wall axis)
        # --------------------------------------------------

        m = 8

        if edge in (ArenaEdge.WEST, ArenaEdge.EAST):
            # west/east edge
            target = (bx, m if by <= 12 else by) if corner == ArenaCorner.SOUTH_WEST or corner == ArenaCorner.SOUTH_EAST else (bx, ArenaConfig.height_cm - m if by >= ArenaConfig.height_cm - 12 else by)
            logger.debug(f"west/east {target}")
        else:
            # north/south edge
            target = (m if bx <= 12 else bx, by) if corner == ArenaCorner.SOUTH_WEST or corner == ArenaCorner.NORTH_WEST else (ArenaConfig.width_cm - m if bx >= ArenaConfig.width_cm - 12 else bx, by)
            logger.debug(f"north/south {target}")

        # south west corner
        if edge == ArenaEdge.WEST and corner == ArenaCorner.SOUTH_WEST:
            target = (bx, m if by <= 12 else by)
        elif edge == ArenaEdge.SOUTH and corner == ArenaCorner.SOUTH_WEST:
            target = (m if bx <= 12 else bx, by)

        with state.lock:
            state.target_point = target

        # --------------------------------------------------
        # 2. Steering toward constrained target
        # --------------------------------------------------

        angle_error = robot.angle_to_point(target)

        base_speed = 20 # approach base speed

        turn = clamp(angle_error * 3, -10, 10) # some random numbers for now

        left_speed = base_speed + turn
        right_speed = base_speed - turn

        # --------------------------------------------------
        # 3. Drive small controlled step
        # --------------------------------------------------

        logger.debug(f"turning ls {left_speed} rs {right_speed}")
        inst = Instruction(
            name=CommandName.TANK_LEFT,
            type=InstructionType.COMMAND,
            args=Arguments(
                seconds=0.25,
                lspeed=left_speed,
                rspeed=right_speed,
            ),
        )

        connection.send_message(Message(instruction=inst))

        sleep(0.15 + SLEEP_BUFFER_SECONDS)

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))

def advance_to_corner_ball(state: ArenaState, connection: RobotConnection, ball: Ball) -> None:
    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=0.5, speed=40),
    )
    connection.send_message(Message(instruction=inst))
    sleep(0.3 + SLEEP_BUFFER_SECONDS)
    drive_backward(state, connection)
    drive_backward(state, connection)
