from src.autonomous_mode.deliver_balls import deliver_balls
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.model.robot import Robot
from src.debug.log import get_logger
from src.autonomous_mode.movement_helpers import _start_ball_intake, _turn_toward_point, _stop_ball_intake, drive_and_collect_ball
from src.autonomous_mode.state_helpers import await_robot, has_vip_balls, update_ball_count_estimate
from time import time
from protocol import Instruction, InstructionType, CommandName, Arguments, Message

_last_ball_count_update_time = 0

COLLECT_BALLS_PER_DELIVERY = 4


def _select_next_ball(robot: Robot, balls_in_robot: int, state: ArenaState):
    """
    Return the next ball to collect, or None if nothing is reachable.
    """
    if has_vip_balls(state) and balls_in_robot == 3:
        # if vip ball is on the field and the robot contains 3 balls
        return robot.get_nearest_vip_ball(state.balls)
    return robot.get_nearest_ball(state.balls)


def _should_deliver(balls_in_robot: int, state: ArenaState) -> bool:
    """
    Return True when the robot should head to the goal and deliver.
    """
    return balls_in_robot >= COLLECT_BALLS_PER_DELIVERY or state.estimated_ball_count == 0

def _all_balls_delivered(balls_in_robot: int, state: ArenaState):
    """
    Return True if all balls have been delivered, False otherwise.
    """
    return state.estimated_ball_count == 0 and balls_in_robot == 0


def _collect_ball(robot: Robot, ball_point: tuple[float, float], connection: RobotConnection, state: ArenaState) -> None:
    """Navigate to and collect a single ball."""
    _turn_toward_point(state, connection, ball_point)
    drive_and_collect_ball(robot, ball_point, connection, state)


def _deliver_and_recount( state: ArenaState, connection: RobotConnection, total_balls: int, balls_delivered_so_far: int) -> tuple[int, int]:
    """
    Deliver balls, recount estimated ball count, and return updated (total_balls, balls_delivered_so_far).
    """
    balls_in_arena_before = total_balls - balls_delivered_so_far
    deliver_balls(state, connection)
    balls_in_arena_after = update_ball_count_estimate(state)
    newly_delivered = balls_in_arena_before - balls_in_arena_after
    return total_balls, balls_delivered_so_far + newly_delivered


def start_autonomous_session(state: ArenaState) -> None:
    logger = get_logger("start_autonomous_session")
    global _last_ball_count_update_time

    connection = RobotConnection()
    _start_ball_intake(connection)

    total_balls = update_ball_count_estimate(state)
    _last_ball_count_update_time = time()
    balls_delivered = 0
    balls_in_robot = 0

    while True:
        _tick(state)

        balls_in_robot = total_balls - state.estimated_ball_count - balls_delivered

        if _all_balls_delivered(balls_in_robot, state):
            logger.debug("All balls delivered, stopping.")
            _stop_ball_intake(connection)
            _send_win_message(connection)
            break

        robot = await_robot(state, connection)

        if _should_deliver(balls_in_robot, state):
            logger.debug(f"Delivering — balls_in_robot={balls_in_robot}, estimated={state.estimated_ball_count}")
            total_balls, balls_delivered = _deliver_and_recount(state, connection, total_balls, balls_delivered)
            balls_in_robot = 0
            continue

        ball = _select_next_ball(robot, balls_in_robot, state)
        if ball is None:
            continue

        logger.debug(f"Collecting ball at {ball.position}")
        _collect_ball(robot, ball.position, connection, state)
        logger.debug("End of loop\n")


def _tick(state: ArenaState) -> None:
    global _last_ball_count_update_time

    if time() - _last_ball_count_update_time >= 10:
        update_ball_count_estimate(state)
        _last_ball_count_update_time = time()


def _send_win_message(connection: RobotConnection) -> None:
    inst = Instruction(
        name=CommandName.TALK,
        type=InstructionType.COMMAND,
        args=Arguments(talk="I did it"),
    )
    connection.send_message(Message(instruction=inst))