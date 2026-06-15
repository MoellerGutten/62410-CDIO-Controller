from src.autonomous_mode.cross_avoidance_helpers import calculate_shortest_waypoint_path, dist_to_point
from src.state.state_manager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message, SequenceName
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.debug.log import get_logger
from time import sleep
from src.lib.movement_constants import BALL_INTAKE_ON_FOR_SECONDS, BALL_INTAKE_SPEED, EJACULATE_SPEED, NUDGE_SECONDS, NUDGE_SPEED, \
TURN_TO_POINT_PRECISE_TOLERANCE, TURN_TO_POINT_TOLERANCE, TURN_TO_POINT_PRECISE_SPEED, TURN_TO_POINT_PRECISE_MS, SLEEP_BUFFER_SECONDS, \
BACKWARD_SPEED, BACKWARD_MS, BURST_FORWARD_SPEED, BURST_FORWARD_MS, GO_TO_MAX_MOVES, GO_TO_DISTANCE_TOLERANCE
from src.lib.movement_algorithms import turn_to_point_turn_ms, turn_to_point_turn_speed, drive_forward_ms, drive_forward_speed
from src.lib.time import ms_to_seconds

# ── Movement Helpers ───────────────────────────────────────────────────────────────────

def _start_ball_intake(connection: RobotConnection) -> None:
    inst = Instruction(
        name=CommandName.BALL_IN,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=BALL_INTAKE_ON_FOR_SECONDS, speed=BALL_INTAKE_SPEED),
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
        args=Arguments(speed=EJACULATE_SPEED),
    )
    connection.send_message(Message(instruction=inst))

def nudge_robot(connection: RobotConnection) -> None:
    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=NUDGE_SECONDS, speed=NUDGE_SPEED),
    )
    connection.send_message(Message(instruction=inst))
    sleep(NUDGE_SECONDS + SLEEP_BUFFER_SECONDS)


def turn_to_point(state: ArenaState, connection: RobotConnection, point: tuple[float, float], precise_mode: bool = False) -> None:
    while True:
        if state.robot is None or point is None: break

        if state.robot.is_facing_point(point, TURN_TO_POINT_PRECISE_TOLERANCE if precise_mode else TURN_TO_POINT_TOLERANCE): break

        angle = state.robot.angle_to_point(point)
        if (precise_mode):
            turn_ms    = TURN_TO_POINT_PRECISE_MS
            turn_speed = TURN_TO_POINT_PRECISE_SPEED
        else:
            turn_ms    = turn_to_point_turn_ms(angle)
            turn_speed = turn_to_point_turn_speed(angle)

        command = CommandName.TANK_RIGHT if angle > 0 else CommandName.TANK_LEFT
        l_speed = turn_speed if angle > 0 else -turn_speed
        r_speed = -turn_speed if angle > 0 else turn_speed

        get_logger().debug(f"Turning: command={command}, l_speed={l_speed}, r_speed={r_speed}")

        inst = Instruction(
            name=command,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=ms_to_seconds(turn_ms), lspeed=l_speed, rspeed=r_speed),
        )
        connection.send_message(Message(instruction=inst))
        sleep(ms_to_seconds(turn_ms) + SLEEP_BUFFER_SECONDS)

        update_state(state)


def drive_forward(state: ArenaState, connection: RobotConnection, point: tuple[float, float]) -> None:
    if state.robot is None:
        return

    distance = state.robot.distance_to_point(point)
    fwd_ms = drive_forward_ms(distance)
    fwd_speed =  drive_forward_speed(distance)

    get_logger().debug(f"Driving: distance={distance:.2f}, speed={fwd_speed}, duration={fwd_ms}ms")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(fwd_ms), speed=fwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(fwd_ms) + SLEEP_BUFFER_SECONDS)


def drive_backward(state: ArenaState, connection: RobotConnection) -> None:
    if state.robot is None:
        return

    bwd_ms = BACKWARD_MS
    bwd_speed =  BACKWARD_SPEED

    get_logger().debug(f"Driving backward: speed={bwd_speed}, duration={bwd_ms}ms")

    inst = Instruction(
        name=CommandName.BACKWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(bwd_ms), speed=bwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(bwd_ms) + SLEEP_BUFFER_SECONDS)


def burst_into_ball(state: ArenaState, connection: RobotConnection, point: list[int]) -> None:
    if state.robot is None:
        return

    distance = state.robot.distance_to_point(point)
    burst_ms = BURST_FORWARD_MS
    burst_speed = BURST_FORWARD_SPEED

    get_logger().debug(f"Collecting ball: distance={distance:.2f}, speed={burst_speed}, duration={burst_ms}ms")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(burst_ms), speed=burst_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(burst_ms) + SLEEP_BUFFER_SECONDS)


# ── Abstracted Movement Helpers (1 layer up) ───────────────────────────────────────────────────────────────────

def go_to(state: ArenaState, connection: RobotConnection, target_point: tuple[float, float]):
    """
    1. Tag robot pos og tjek om den intercepter inflated bounding box
    2. Redirect robot og brug nærmeste* waypoint til at køre uden om\n
    2.1. Nærmeste* waypoint\n
    2.2. Tjek om den stadig kører igennem inflated bounding box\n
    2.3. Kør til waypoint som er tættest på originalt punkt\n
    3. Kør mod originalt punkt

    *Nærmeste = Kortest fra robot til punkt og waypoint til punkt
    """
    from src.autonomous_mode.state_helpers import await_robot
    robot = await_robot(state, connection)
    get_logger("go_to").debug(f"Going to point: {target_point}")

    distance = dist_to_point(robot.position, target_point)
    if state.cross:
        waypoints = calculate_shortest_waypoint_path(state, connection, target_point)
    waypoints.append(target_point)
    for waypoint in waypoints:
        _iter = 0
        while distance > GO_TO_DISTANCE_TOLERANCE or _iter <= GO_TO_MAX_MOVES:
            turn_to_point(state, connection, waypoint)
            drive_forward(state, connection, waypoint)

            distance = dist_to_point(robot.position, waypoint)
            _iter += 1
            robot = await_robot(state, connection)
