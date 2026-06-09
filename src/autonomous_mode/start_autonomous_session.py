from src.autonomous_mode.deliver_balls import deliver_balls
from src.state.state_manager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from logging import Logger
from time import sleep
from src.autonomous_mode.movement_helpers import _collect_ball, _start_ball_intake, _turn_toward_point, _drive_toward_point
from src.autonomous_mode.state_helpers import _await_robot, _test_for_ball_presence


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
            if _test_for_ball_presence(state, logger) > 0: continue
            deliver_balls(state, connection, logger)
            break

        print("Balls")

        ball_point = state.balls[0].position
        _turn_toward_point(state, connection, logger, ball_point)

        print("Turned")

        if state.robot is None: continue
        if state.robot.distance_to_point(ball_point) < 10:
            _collect_ball(state, connection, logger, ball_point)
        else:
            print("drive")   
            _drive_toward_point(state, connection, logger, ball_point)

        print("\n")  


