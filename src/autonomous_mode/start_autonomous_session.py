from src.autonomous_mode.deliver_balls import deliver_balls
from src.state.state_manager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from logging import Logger
from time import sleep


MAX_ROBOT_DETECTION_ATTEMPTS = 5


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
            deliver_balls(state, connection, logger)
            break

        ball = state.balls[0]
        _turn_toward_ball(state, connection, logger, ball)
        _drive_toward_ball(state, connection, logger)

        update_state(state, logger)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _start_ball_intake(connection: RobotConnection) -> None:
    inst = Instruction(
        name=CommandName.BALL_IN,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=500, speed=100),
    )
    connection.send_message(Message(instruction=inst))


def _await_robot(state: ArenaState, connection: RobotConnection, logger: Logger):
    """Poll for robot detection. If not found after MAX attempts, nudge the robot and return None."""
    for attempt in range(MAX_ROBOT_DETECTION_ATTEMPTS):
        update_state(state, logger)
        if state.robot is not None:
            return state.robot
        print(f"Robot not detected (attempt {attempt + 1}/{MAX_ROBOT_DETECTION_ATTEMPTS})")

    print("Robot still not detected — nudging robot")
    _nudge_robot(connection)
    return None


def _nudge_robot(connection: RobotConnection) -> None:
    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=0.3, speed=15),
    )
    connection.send_message(Message(instruction=inst))
    sleep(0.35)


def _turn_toward_ball(state: ArenaState, connection: RobotConnection, logger: Logger, ball) -> None:
    while True:
        if state.robot is None or ball is None:
            break
        if state.robot.is_facing_point(ball.position, tolerance_deg=3.0):
            break

        angle = state.robot.angle_to_point(ball.position)
        turn_ms = max(100, min(300, int(abs(angle) * 10)))
        turn_speed = max(10, min(20, int(abs(angle) * 0.4)))

        command = CommandName.TANK_RIGHT if angle > 0 else CommandName.TANK_LEFT
        l_speed = turn_speed if angle > 0 else -turn_speed
        r_speed = -turn_speed if angle > 0 else turn_speed

        inst = Instruction(
            name=command,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=turn_ms / 1000, lspeed=l_speed, rspeed=r_speed),
        )
        connection.send_message(Message(instruction=inst))
        sleep(turn_ms / 1000 + 0.05)

        update_state(state, logger)
        ball = state.balls[0] if state.balls else None


def _drive_toward_ball(state: ArenaState, connection: RobotConnection, logger: Logger) -> None:
    if not state.balls or state.robot is None:
        return

    ball = state.balls[0]
    distance = state.robot.distance_to_point(ball.position)
    fwd_ms = max(100, min(500, int(distance * 5)))
    fwd_speed = max(10, min(50, int(distance * 0.75)))

    print(f"Driving toward ball: distance={distance:.2f}, speed={fwd_speed}, duration={fwd_ms}ms")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=fwd_ms / 1000, speed=fwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(fwd_ms / 1000 + 0.05)