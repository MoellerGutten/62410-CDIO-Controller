from src.autonomous_mode.deliver_balls import deliver_balls
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.model.robot import Robot
from src.model.ball import Ball
from src.debug.log import get_logger
from src.autonomous_mode.movement_helpers import _start_ball_intake, _stop_ball_intake
from src.autonomous_mode.state_helpers import await_robot, has_vip_balls, update_ball_count_estimate
from time import time
from src.autonomous_mode.collection_helpers import collect_cross_zone_ball, collect_edge_ball, collect_normal_ball, collect_corner_ball, collect_waypoint_zone_ball
from protocol import Instruction, InstructionType, CommandName, Arguments, Message
from src.lib.constants import BALLS_PER_DELIVERY, BALL_COUNT_ESTIMATE_INVALIDATION_SECONDS, WIN_MESSAGE, MATCH_DURATION_SECONDS, HAIL_MARY_TIME_LEFT_SECONDS

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
    is_edge_ball, edge_ball_staging_point = ball.is_edge_ball()

    if ball.is_corner_ball():
        collect_corner_ball(state, connection, ball)
    elif is_edge_ball:
        collect_edge_ball(state, ball, connection, edge_ball_staging_point)
    elif not ball.is_within_cross_zone(state.cross) and ball.is_within_waypoint_zone(state):
        collect_waypoint_zone_ball(state, ball, connection)
    elif ball.is_within_cross_zone(state.cross):
        collect_cross_zone_ball(state, ball, connection)
    else: 
        collect_normal_ball(state, ball, connection)
        
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

        if _is_hail_mary_time(state):
            logger.debug("Hail mary time")
            deliver_balls(state, connection)
            _send_win_message(connection)
            with state.lock:
                state.all_balls_delivered = True
            break

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

def _is_hail_mary_time(state: ArenaState) -> bool:
    if state.start_time is None:
        return False
    time_left = MATCH_DURATION_SECONDS - (time() - state.start_time)
    return time_left <= HAIL_MARY_TIME_LEFT_SECONDS
