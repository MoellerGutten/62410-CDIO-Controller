from src.autonomous_mode.movement_helpers import burst_backward, go_to, turn_to_point, _start_ejaculation, _stop_ball_intake
from src.autonomous_mode.state_helpers import await_robot
from src.model.arena_state import ArenaState
from src.lib.connection import RobotConnection
from src.debug.log import get_logger
from src.lib.constants import DRIVE_TO_CENTER_DISTANCE_TOLERANCE, GOAL_DELIVERY_POINT, ARENA_WIDTH_CM, ARENA_HEIGHT_CM

def deliver_balls(state: ArenaState, connection: RobotConnection) -> None:
    logger = get_logger("deliver_balls")    
    logger.debug("Commence delivery")

    _stop_ball_intake(connection)
    drive_to_center(state, connection)
    drive_to_goal(state, connection)
    _start_ejaculation(connection)
    burst_backward(state, connection)

    logger.debug("Delivery done\n")


def drive_to_center(state: ArenaState, connection: RobotConnection):
    logger = get_logger("drive_to_center")

    robot = await_robot(state, connection)
    center_line_point: tuple[float, float] = (ARENA_WIDTH_CM * 0.70, ARENA_HEIGHT_CM / 2)

    logger.debug("Commence drive to center")

    go_to(state, connection, center_line_point)

    logger.debug("At center\n")

        
def drive_to_goal(state: ArenaState, connection: RobotConnection):
    logger = get_logger("drive_to_goal")

    goal: tuple[float, float] = (ARENA_WIDTH_CM, ARENA_HEIGHT_CM / 2)
    with state.lock:
        state.target_point = goal

    logger.debug("Commence drive to goal")

    turn_to_point(state, connection, GOAL_DELIVERY_POINT)
    go_to(state, connection, GOAL_DELIVERY_POINT)
    turn_to_point(state, connection, goal, precise_mode=True)
        

    logger.debug("At goal\n")

