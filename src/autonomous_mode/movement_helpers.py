from src.state.state_manager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message, SequenceName
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.debug.log import get_logger
from time import sleep

# ── Movement Helpers ───────────────────────────────────────────────────────────────────

def _start_ball_intake(connection: RobotConnection) -> None:
    inst = Instruction(
        name=CommandName.BALL_IN,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=500, speed=100),
    )
    connection.send_message(Message(instruction=inst))

def _stop_ball_intake(connection: RobotConnection) -> None:
    inst = Instruction(
        name=CommandName.BALL_OFF,
        type=InstructionType.COMMAND,
        args=Arguments(),
    )
    connection.send_message(Message(instruction=inst))

def _start_ejaculation(connection: RobotConnection) -> None:
    inst = Instruction(
        name=SequenceName.EJECT,
        type=InstructionType.SEQUENCE,
        args=Arguments(speed=100),
    )
    connection.send_message(Message(instruction=inst))

def nudge_robot(connection: RobotConnection) -> None:
    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=0.3, speed=15),
    )
    connection.send_message(Message(instruction=inst))
    sleep(0.35)


def _turn_toward_point(state: ArenaState, connection: RobotConnection, point: list[int]) -> None:
    while True:
        if state.robot is None or point is None:
            break
        if state.robot.is_facing_point(point, tolerance_deg=8.0):
            break

        angle = state.robot.angle_to_point(point)
        turn_ms    = max(100, min(500, int(abs(angle) * 5)))
        turn_speed = max(10, min(100, int(abs(angle) * 0.4)))

        command = CommandName.TANK_RIGHT if angle > 0 else CommandName.TANK_LEFT
        l_speed = turn_speed if angle > 0 else -turn_speed
        r_speed = -turn_speed if angle > 0 else turn_speed

        get_logger().debug(f"Turning: command={command}, l_speed={l_speed}, r_speed={r_speed}")

        inst = Instruction(
            name=command,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=turn_ms / 1000, lspeed=l_speed, rspeed=r_speed),
        )
        connection.send_message(Message(instruction=inst))
        sleep(turn_ms / 1000 + 0.05)

        update_state(state)


def _drive_toward_point(state: ArenaState, connection: RobotConnection, point: list[int]) -> None:
    if state.robot is None:
        return

    distance = state.robot.distance_to_point(point)
    fwd_ms = max(100, min(2000, int(distance * 5)))
    fwd_speed =  max(30, min(100, int(distance * 0.8)))

    get_logger().debug(f"Driving: distance={distance:.2f}, speed={fwd_speed}, duration={fwd_ms}ms")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=fwd_ms / 1000, speed=fwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(fwd_ms / 1000 + 0.05)

def drive_and_collect_ball(robot, ball_point, connection, state):
    from src.autonomous_mode.state_helpers import update_ball_count_estimate
    if robot.distance_to_point(ball_point) < 10:
        _collect_ball(state, connection, ball_point)
        update_ball_count_estimate(state)
    else:
        _drive_toward_point(state, connection, ball_point)

def _collect_ball(state: ArenaState, connection: RobotConnection, point: list[int]) -> None:
    if state.robot is None:
        return

    distance = state.robot.distance_to_point(point)
    fwd_ms = 250
    fwd_speed = 75

    get_logger().debug(f"Collecting ball: distance={distance:.2f}, speed={fwd_speed}, duration={fwd_ms}ms")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=fwd_ms / 1000, speed=fwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(fwd_ms / 1000 + 0.05)


def adjust_heading(state: ArenaState, connection: RobotConnection, point: list[int]) -> None:
    get_logger().debug(f"adjusting heading")
    while True:
        if state.robot is None or point is None: break
        if state.robot.is_facing_point(point, tolerance_deg=2.0): break

        angle = state.robot.angle_to_point(point)
        turn_ms    = 100
        turn_speed = 10

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

        update_state(state)