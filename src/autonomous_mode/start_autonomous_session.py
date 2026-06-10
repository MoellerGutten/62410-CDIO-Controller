from src.autonomous_mode.deliver_balls import deliver_balls
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from logging import Logger
from src.autonomous_mode.movement_helpers import _collect_ball, _start_ball_intake, _turn_toward_point, _drive_toward_point
from src.autonomous_mode.state_helpers import _await_robot, update_ball_count_estimate


def start_autonomous_session(state: ArenaState, logger: Logger) -> None:
    connection = RobotConnection()
    _start_ball_intake(connection)

    while True:
        robot = _await_robot(state, connection, logger)
        if robot is None:
            print("Robot not detected after nudge — retrying main loop")
            continue

        print("Starting")
        if not state.balls:
            if update_ball_count_estimate(state, logger) > 0: continue
            deliver_balls(state, connection, logger)
            break

        print("Balls")

        vip = robot.get_nearest_vip_ball(state)
        if vip is not None:
            print("VIP ball detected — prio is the orange ball")
            ball_point = vip.position
        else:
            nearest = robot.get_nearest_ball(state)
            if nearest is None:
                continue
            ball_point = nearest.position

        _turn_toward_point(state, connection, logger, ball_point)

        print("Turned")

        if robot.distance_to_point(ball_point) < 15:
            _collect_ball(state, connection, logger, ball_point)
        else:
            print("drive")
            _drive_toward_point(state, connection, logger, ball_point)

        print("\n")  


