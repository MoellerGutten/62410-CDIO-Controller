from src.autonomous_mode.deliver_balls import deliver_balls
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.model.robot import Robot
from src.model.ball import Ball
from src.debug.log import get_logger
from src.autonomous_mode.movement_helpers import drive_forward, _start_ball_intake, turn_to_point, _stop_ball_intake, burst_into_ball, drive_backward, go_to, handle_balls_in_radius
from src.autonomous_mode.state_helpers import await_robot, has_vip_balls, update_ball_count_estimate
from time import time
from protocol import Instruction, InstructionType, CommandName, Arguments, Message
from src.lib.constants import BALLS_PER_DELIVERY, ROBOT_TO_POINT_DISTANCE_BEFORE_BURST, BALL_COUNT_ESTIMATE_INVALIDATION_SECONDS, WIN_MESSAGE, GO_TO_BALL_APPROACH_RADIUS, GO_TO_BALL_EDGE_APPROACH_RADIUS

_last_ball_count_update_time = 0


def _select_next_ball(robot: Robot, balls_in_robot: int, state: ArenaState):
    """
    Return the next ball to collect, or None if nothing is reachable.
    """
    vips_on_field = has_vip_balls(state)
    # if vip is on field and 3 balls (or none other left on field) in robot, select nearest vip ball, otherwise go for nearest ball

    if not vips_on_field:
        next_ball = robot.get_nearest_non_vip_ball(state.balls)
    else:
        balls_left_before_delivery = BALLS_PER_DELIVERY - balls_in_robot if (state.estimated_ball_count + balls_in_robot) >= BALLS_PER_DELIVERY else state.estimated_ball_count
        next_ball = robot.get_nearest_vip_ball(state.balls) if balls_left_before_delivery == 1 else robot.get_nearest_non_vip_ball(state.balls)

    get_logger("_select_next_ball").debug(f"Next ball: {next_ball!r}, balls_in_robot: {balls_in_robot}, has_vip_balls: {vips_on_field}")
    return next_ball


def _should_deliver(balls_in_robot: int, state: ArenaState) -> bool:
    """
    Return True when the robot should head to the goal and deliver.
    """
    return balls_in_robot >= BALLS_PER_DELIVERY or state.estimated_ball_count == 0

def _all_balls_delivered(balls_in_robot: int, state: ArenaState):
    """
    Return True if all balls have been delivered, False otherwise.
    """
    return state.estimated_ball_count == 0 and balls_in_robot == 0


def _collect_ball(ball: Ball, connection: RobotConnection, state: ArenaState) -> None:
    """Navigate to and collect a single ball."""
    is_edge_ball, new_ball_point = ball.is_edge_ball()
    if is_edge_ball:
        go_to(state, connection, new_ball_point)

        turn_to_point(state, connection, ball.position)
        go_to(state, connection, ball.position, GO_TO_BALL_EDGE_APPROACH_RADIUS)
        burst_into_ball(state, connection, ball.position)
        drive_backward(state, connection)
    else: 
        handle_balls_in_radius(state, connection, ball)

        go_to(state, connection, ball.position, approach_radius=GO_TO_BALL_APPROACH_RADIUS)
        robot = await_robot(state, connection)
        if robot.distance_to_point(ball.position) > 20.0: # TODO: adjust and make constant
            # return early if ball is in a galaxy far far away.
            update_ball_count_estimate(state)
            return
        turn_to_point(state, connection, ball.position, precise_mode=True)
        burst_into_ball(state, connection, ball.position)

    update_ball_count_estimate(state)


def _deliver_and_recount(state: ArenaState, connection: RobotConnection, total_balls: int, balls_delivered_so_far: int) -> int:
    """
    Deliver balls, recount estimated ball count, and return updated balls_delivered_so_far.
    """
    balls_in_arena_before = total_balls - balls_delivered_so_far
    deliver_balls(state, connection)
    balls_in_arena_after = update_ball_count_estimate(state)
    newly_delivered = balls_in_arena_before - balls_in_arena_after
    return balls_delivered_so_far + newly_delivered


def start_autonomous_session(state: ArenaState) -> None:
    logger = get_logger("start_autonomous_session")
    global _last_ball_count_update_time

    connection = RobotConnection()
    _start_ball_intake(connection)

    total_balls = update_ball_count_estimate(state)
    _last_ball_count_update_time = time()

    while True:
        _tick(state)

        with state.lock:
            state.estimated_balls_in_robot = total_balls - state.estimated_ball_count - state.estimated_balls_delivered

        if _all_balls_delivered(state.estimated_balls_in_robot, state):
            logger.debug("All balls delivered, stopping.")
            _stop_ball_intake(connection)
            _send_win_message(connection)
            with state.lock:
                state.all_balls_delivered = True
            break

        robot = await_robot(state, connection)

        if _should_deliver(state.estimated_balls_in_robot, state):
            logger.debug(f"Delivering — estimated_balls_in_robot={state.estimated_balls_in_robot}, estimated_ball_count={state.estimated_ball_count}, estimated_balls_delivered={state.estimated_balls_delivered}")
            balls_delivered_so_far = _deliver_and_recount(state, connection, total_balls, state.estimated_balls_delivered)
            with state.lock:
                state.estimated_balls_delivered = balls_delivered_so_far
                state.estimated_balls_in_robot = 0
            continue

        ball = _select_next_ball(robot, state.estimated_balls_in_robot, state)
        if ball is None:
            continue

        logger.debug(f"Collecting ball at {ball.position}")
        _collect_ball(ball, connection, state)
        logger.debug("End of loop\n")


def _tick(state: ArenaState) -> None:
    global _last_ball_count_update_time

    if time() - _last_ball_count_update_time >= BALL_COUNT_ESTIMATE_INVALIDATION_SECONDS:
        update_ball_count_estimate(state)
        _last_ball_count_update_time = time()


def _send_win_message(connection: RobotConnection) -> None:
    inst = Instruction(
        name=CommandName.TALK,
        type=InstructionType.COMMAND,
        args=Arguments(talk=WIN_MESSAGE),
    )
    connection.send_message(Message(instruction=inst))
