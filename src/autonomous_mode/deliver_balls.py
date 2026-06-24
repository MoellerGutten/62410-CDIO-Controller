from src.autonomous_mode.movement_helpers import burst_backward, go_to, turn_to_point, _start_ejaculation, _stop_ball_intake, creep_forward_step
from src.autonomous_mode.state_helpers import await_robot
from src.model.arena_state import ArenaState
from src.lib.connection import RobotConnection
from src.debug.log import get_logger
from src.lib.constants import GOAL_DELIVERY_POINT, ARENA_WIDTH_CM, ARENA_HEIGHT_CM, FINAL_GOAL_DELIVERY_POINT, GET_REAL_CLOSE_MAX_ITER
from time import time


def deliver_balls(state: ArenaState, connection: RobotConnection) -> None:
    logger = get_logger("deliver_balls")    
    logger.debug("Commence delivery")

    _stop_ball_intake(connection)
    _drive_to_center(state, connection)
    _drive_to_goal(state, connection)
    _get_real_close(state, connection)
    _start_ejaculation(connection)

    if state.is_final_delivery:
        with state.lock:
            if state.finish_time is None:
                state.finish_time = time()


    burst_backward(state, connection)

    logger.debug("Delivery done\n")


def _drive_to_center(state: ArenaState, connection: RobotConnection):
    logger = get_logger("drive_to_center")

    await_robot(state, connection)
    center_line_point: tuple[float, float] = (ARENA_WIDTH_CM * 0.70, ARENA_HEIGHT_CM / 2)

    logger.debug("Commence drive to center")

    go_to(state, connection, center_line_point)
    goal: tuple[float, float] = (ARENA_WIDTH_CM, ARENA_HEIGHT_CM / 2)

    logger.debug("At center")
    turn_to_point(state, connection, goal)
    logger.debug("Turned to goal")

        
def _drive_to_goal(state: ArenaState, connection: RobotConnection):
    logger = get_logger("drive_to_goal")

    goal: tuple[float, float] = (ARENA_WIDTH_CM, ARENA_HEIGHT_CM / 2)
    with state.lock:
        state.target_point = goal

    logger.debug("Commence drive to goal")

    turn_to_point(state, connection, GOAL_DELIVERY_POINT)
    go_to(state, connection, GOAL_DELIVERY_POINT)
    turn_to_point(state, connection, goal, precise_mode=True)
        

    logger.debug("At goal\n")

def _get_real_close(state: ArenaState, connection: RobotConnection):
    logger = get_logger("getting_real_close")
    robot = await_robot(state, connection)

    goal: tuple[float, float] = (ARENA_WIDTH_CM, ARENA_HEIGHT_CM / 2)

    with state.lock:
        state.target_point = goal

    logger.debug("Commence get real close")

    turn_to_point(state, connection, goal, precise_mode=True)
    await_robot(state, connection)
    _iter = 0 # for debug only
    while robot.position[0] < FINAL_GOAL_DELIVERY_POINT[0] and _iter < GET_REAL_CLOSE_MAX_ITER:
        logger.debug(f"creeping - iter: {_iter}")
        creep_forward_step(state, connection)
        robot = await_robot(state, connection)
        _iter += 1
    logger.debug("At final goal position")
    turn_to_point(state, connection, goal, precise_mode=True)
    logger.debug("At goal\n")

