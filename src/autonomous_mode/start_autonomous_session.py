from src.autonomous_mode.deliver_balls import deliver_balls
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from logging import Logger
from src.autonomous_mode.movement_helpers import _collect_ball, _start_ball_intake, _stop_ball_intake, _turn_toward_point, _drive_toward_point
from src.autonomous_mode.state_helpers import _await_robot, has_vip_balls, update_ball_count_estimate


def start_autonomous_session(state: ArenaState, logger: Logger) -> None:
    connection = RobotConnection()
    _start_ball_intake(connection)
    update_ball_count_estimate(state, logger)

    while True:
        robot = _await_robot(state, connection, logger)
        if robot is None:
            print("Robot not detected after nudge — retrying main loop")
            continue

        start_collecting_vip_balls(state, connection, logger)

        start_collecting_normal_balls(state, connection, logger)


def start_collecting_vip_balls(state: ArenaState, connection: RobotConnection, logger: Logger) -> None:
    while True:
        robot = _await_robot(state, connection, logger)
        if robot is None:
            print("Robot not detected after nudge — retrying main loop")
            continue

        if not has_vip_balls(state):
            if update_ball_count_estimate(state, logger) > 3: continue
            deliver_balls(state, connection, logger)
            break

        vip = robot.get_nearest_vip_ball(state.balls)
        if vip is None: continue
        ball_point = vip.position

        _turn_toward_point(state, connection, logger, ball_point)

        drive_into_ball(robot, ball_point, connection, state, logger)


def start_collecting_normal_balls(state: ArenaState, connection: RobotConnection, logger: Logger) -> None:
    while True:
        robot = _await_robot(state, connection, logger)
        if robot is None:
            print("Robot not detected after nudge — retrying main loop")
            continue

        if not state.balls:
            if update_ball_count_estimate(state, logger) > 0: continue
            deliver_balls(state, connection, logger)
            break

        nearest = robot.get_nearest_ball(state.balls)
        if nearest is None:continue
        ball_point = nearest.position

        _turn_toward_point(state, connection, logger, ball_point)

        drive_into_ball(robot, ball_point, connection, state, logger)


def drive_into_ball(robot, ball_point, connection, state, logger):
    if robot.distance_to_point(ball_point) < 10:
        _collect_ball(state, connection, logger, ball_point)
        update_ball_count_estimate(state, logger)
    else:
        _drive_toward_point(state, connection, logger, ball_point)