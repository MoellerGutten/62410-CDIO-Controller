from src.autonomous_mode.movement_helpers import drive_forward, turn_to_point, _start_ejaculation, _stop_ball_intake
from src.autonomous_mode.state_helpers import await_robot
from src.model.arena_state import ArenaState
from src.lib.connection import RobotConnection
from src.debug.log import get_logger
from src.state.arena_config import ArenaConfig
from src.lib.constants import DRIVE_TO_CENTER_DISTANCE_TOLERANCE, DRIVE_TO_GOAL_CLOSE_TO_GOAL_RANGE, DRIVE_TO_GOAL_AT_GOAL_RANGE

def deliver_balls(state: ArenaState, connection: RobotConnection) -> None:
    logger = get_logger("deliver_balls")    
    logger.debug("Commence delivery")

    _stop_ball_intake(connection)
    drive_to_center(state, connection)
    drive_to_goal(state, connection)
    _start_ejaculation(connection)

    logger.debug("Delivery done\n")


def drive_to_center(state: ArenaState, connection: RobotConnection):
    logger = get_logger("drive_to_center")

    if (state.robot == None): return
    center_line_point = [state.robot.position[0], ArenaConfig.height_cm / 2]

    logger.debug("Commence drive to center")

    while True:
        robot = await_robot(state, connection)

        if (robot.distance_to_point(center_line_point) < DRIVE_TO_CENTER_DISTANCE_TOLERANCE): break

        turn_to_point(state, connection, center_line_point)
        drive_forward(state, connection, center_line_point)

    logger.debug("At center\n")

        
def drive_to_goal(state: ArenaState, connection: RobotConnection):
    logger = get_logger("drive_to_goal")

    goal = [ArenaConfig.width_cm, ArenaConfig.height_cm / 2]

    logger.debug("Commence drive to goal")

    while True:
        robot = await_robot(state, connection)

        if (robot.distance_to_point(goal) < DRIVE_TO_GOAL_CLOSE_TO_GOAL_RANGE):
            # when close, use precise mode for turning
            turn_to_point(state, connection, goal, precise_mode=True)

            if (robot.distance_to_point(goal) < DRIVE_TO_GOAL_AT_GOAL_RANGE):
                # when within range, we consider the robot to be at the goal
                break

        turn_to_point(state, connection, goal)
        drive_forward(state, connection, goal)

    logger.debug("At goal\n")
