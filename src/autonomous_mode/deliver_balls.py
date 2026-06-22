from src.autonomous_mode.movement_helpers import drive_backward, drive_forward, go_to, turn_to_point, _start_ejaculation, _stop_ball_intake
from src.autonomous_mode.state_helpers import await_robot
from src.model.arena_state import ArenaState
from src.lib.connection import RobotConnection
from src.debug.log import get_logger
from src.state.arena_config import ArenaConfig
from src.lib.constants import DRIVE_TO_CENTER_DISTANCE_TOLERANCE, DRIVE_TO_GOAL_CLOSE_TO_GOAL_RANGE, DRIVE_TO_GOAL_AT_GOAL_RANGE, GOAL_DELIVERY_POINT, ARUCO_OFFSET_X

def deliver_balls(state: ArenaState, connection: RobotConnection) -> None:
    logger = get_logger("deliver_balls")    
    logger.debug("Commence delivery")

    _stop_ball_intake(connection)
    drive_to_center(state, connection)
    drive_to_goal(state, connection)
    _start_ejaculation(connection)
    drive_backward(state, connection)

    logger.debug("Delivery done\n")


def drive_to_center(state: ArenaState, connection: RobotConnection):
    logger = get_logger("drive_to_center")

    robot = await_robot(state, connection)
    center_line_point: tuple[float, float] = (ArenaConfig.width_cm * 3/4, ArenaConfig.height_cm / 2)

    logger.debug("Commence drive to center")

    while True:
        if robot.distance_to_point(center_line_point) < DRIVE_TO_CENTER_DISTANCE_TOLERANCE:
            break
        go_to(state, connection, center_line_point)
        robot = await_robot(state, connection)

    logger.debug("At center\n")

        
def drive_to_goal(state: ArenaState, connection: RobotConnection):
    logger = get_logger("drive_to_goal")

    goal: tuple[float, float] = (ArenaConfig.width_cm, ArenaConfig.height_cm / 2)
    with state.lock:
        state.target_point = goal

    logger.debug("Commence drive to goal")

    go_to(state, connection, GOAL_DELIVERY_POINT, approach_radius=ARUCO_OFFSET_X)
    turn_to_point(state, connection, goal, precise_mode=True)
        

    logger.debug("At goal\n")

