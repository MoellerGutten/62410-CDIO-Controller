from src.autonomous_mode.deliver_balls import deliver_balls
from src.state.state_manager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from logging import Logger
from time import sleep
from src.autonomous_mode.movement_helpers import _start_ball_intake, _turn_toward_ball, _drive_toward_ball
from src.autonomous_mode.state_helpers import _await_robot, test_for_ball_presence


def start_autonomous_session(state: ArenaState, logger: Logger) -> None:
    connection = RobotConnection()
    _start_ball_intake(connection)

    while True:
        robot = _await_robot(state, connection, logger)
        if robot is None:
            print("Robot not detected after nudge — retrying main loop")
            continue

        update_state(state, logger)

        if not state.balls:
            if test_for_ball_presence(state, logger) > 0: continue
            deliver_balls(state, connection, logger)
            break

        ball = state.balls[0]
        _turn_toward_ball(state, connection, logger, ball)
        _drive_toward_ball(state, connection, logger)

        update_state(state, logger)


