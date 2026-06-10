from src.autonomous_mode.deliver_balls import deliver_balls
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState

from src.debug.log import get_logger
from src.autonomous_mode.movement_helpers import _collect_ball, _start_ball_intake, _stop_ball_intake, _turn_toward_point, _drive_toward_point
from src.autonomous_mode.state_helpers import _await_robot, has_vip_balls, update_ball_count_estimate


def start_autonomous_session(state: ArenaState) -> None:
    connection = RobotConnection()
    _start_ball_intake(connection)
    update_ball_count_estimate(state)

    while True:
        robot = _await_robot(state, connection)
        if robot is None:
            get_logger().warning("Robot not detected after nudge — retrying main loop")
            continue

        start_collecting_vip_balls(state, connection, logger)

        start_collecting_normal_balls(state, connection, logger)
        
        break


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

        get_logger().debug("Finding VIP ball to collect")
        vip = robot.get_nearest_vip_ball(state.balls)
        if vip is None: continue
        ball_point = vip.position

        get_logger().debug("Turning towards ball")
        _turn_toward_point(state, connection, logger, ball_point)
        get_logger().debug("Turned towards ball")

        drive_into_ball(robot, ball_point, connection, state, logger)


def start_collecting_normal_balls(state: ArenaState, connection: RobotConnection, logger: Logger) -> None:
    while True:
        robot = _await_robot(state, connection, logger)
        if robot is None:
            print("Robot not detected after nudge — retrying main loop")
            continue

        if not state.balls:
            if update_ball_count_estimate(state) > 0: continue
            deliver_balls(state, connection)
            break

        get_logger().debug("Finding ball to collect")
        nearest = robot.get_nearest_ball(state.balls)
        if nearest is None:continue
        ball_point = nearest.position

        get_logger().debug("Turning towards ball")
        _turn_toward_point(state, connection, logger, ball_point)
        get_logger().debug("Turned towards ball")

        drive_into_ball(robot, ball_point, connection, state, logger)


def drive_into_ball(robot, ball_point, connection, state, logger):
    if robot.distance_to_point(ball_point) < 15:
        get_logger().debug("Collecting ball")
        _collect_ball(state, connection, logger, ball_point)
        update_ball_count_estimate(state, logger)
    else:
        _drive_toward_point(state, connection, logger, ball_point)

