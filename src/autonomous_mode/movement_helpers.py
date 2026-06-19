from math import hypot
from src.autonomous_mode.cross_avoidance_helpers import calculate_shortest_waypoint_path, dist_to_point
from protocol import CommandName, Arguments, Instruction, InstructionType, Message
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.debug.log import get_logger
from time import sleep
from src.lib.constants import BALL_INTAKE_ON_FOR_SECONDS, BALL_INTAKE_SPEED, EJACULATE_SPEED, NUDGE_SECONDS, NUDGE_SPEED, \
TURN_TO_POINT_PRECISE_TOLERANCE, TURN_TO_POINT_TOLERANCE, TURN_TO_POINT_PRECISE_SPEED, TURN_TO_POINT_PRECISE_MS, SLEEP_BUFFER_SECONDS, \
BACKWARD_SPEED, BACKWARD_MS, BURST_FORWARD_SPEED, BURST_FORWARD_MS, GO_TO_MAX_MOVES, GO_TO_DISTANCE_TOLERANCE, \
ROBOT_TO_POINT_DISTANCE_BEFORE_BURST
from src.lib.algorithms import turn_to_point_turn_ms, turn_to_point_turn_speed, drive_forward_ms, drive_forward_speed
from src.lib.time import ms_to_seconds
from src.autonomous_mode.state_helpers import await_robot


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
        name=CommandName.BALL_OUT,
        type=InstructionType.COMMAND,
        args=Arguments(speed=EJACULATE_SPEED, seconds=5, block=True), # blocking to prevent short circuiting ejaculation
    )
    connection.send_message(Message(instruction=inst))
    _start_ball_intake(connection) # start intake after ejaculation

def turn_to_point(state: ArenaState, connection: RobotConnection, point: tuple[float, float], precise_mode: bool = False) -> None:
    robot = await_robot(state, connection)
    while True:
        if point is None:
            break

        if robot.is_facing_point(point, TURN_TO_POINT_PRECISE_TOLERANCE if precise_mode else TURN_TO_POINT_TOLERANCE):
            break

        angle = robot.angle_to_point(point)
        turn_ms    = turn_to_point_turn_ms(angle)
        turn_speed = turn_to_point_turn_speed(angle)

        #logger = get_logger("turn_to_point")
        #logger.debug(f"Turning to point. turn ms: {turn_ms}, turn speed: {turn_speed}, angel: {angle}")
    
        command = CommandName.TANK_RIGHT if angle > 0 else CommandName.TANK_LEFT
        l_speed = turn_speed if angle > 0 else -turn_speed
        r_speed = -turn_speed if angle > 0 else turn_speed

        inst = Instruction(
            name=command,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=ms_to_seconds(turn_ms), lspeed=l_speed, rspeed=r_speed),
        )
        connection.send_message(Message(instruction=inst))
        sleep(ms_to_seconds(turn_ms) + SLEEP_BUFFER_SECONDS)

        robot = await_robot(state, connection)


def drive_forward(state: ArenaState, connection: RobotConnection, point: tuple[float, float]) -> None:
    robot = await_robot(state, connection)

    distance = robot.distance_to_point(point)
    fwd_ms = drive_forward_ms(distance)
    fwd_speed =  drive_forward_speed(distance)

    #logger = get_logger("drive_forward")
    #logger.debug(f"Driving forward. fwd ms: {fwd_ms}, fwd speed: {fwd_speed}, distance: {distance}")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(fwd_ms), speed=fwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(fwd_ms) + SLEEP_BUFFER_SECONDS)

def drive_backward(state: ArenaState, connection: RobotConnection, speed: int = BACKWARD_SPEED, ms: int = BACKWARD_MS) -> None:
    if state.robot is None:
        return

    bwd_ms = ms
    bwd_speed = speed

    #logger = get_logger("drive_backward")
    #logger.debug(f"Driving backward. bwd ms: {bwd_ms}, bwd speed: {bwd_speed}")

    inst = Instruction(
        name=CommandName.BACKWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(bwd_ms), speed=bwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(bwd_ms) + SLEEP_BUFFER_SECONDS)


def burst_into_ball(state: ArenaState, connection: RobotConnection, point: tuple[float, float],
                    speed: int = BURST_FORWARD_SPEED, ms: int = BURST_FORWARD_MS) -> None:
    robot = await_robot(state, connection)

    distance = robot.distance_to_point(point)
    burst_ms = ms
    burst_speed = speed

    #logger = get_logger("burst_into_ball")
    #logger.debug(f"Busting. burst ms: {burst_ms}, burst speed: {burst_speed}")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(burst_ms), speed=burst_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(burst_ms) + SLEEP_BUFFER_SECONDS)

# Unused: replaced by creep_forward_step for cross-zone collection. Kept commented for reference.
# def move_slowly_towards_point(state: ArenaState, connection: RobotConnection, point: tuple[float, float],
#                               burst_speed: int = BURST_FORWARD_SPEED, burst_ms: int = BURST_FORWARD_MS) -> None:
#
#     logger = get_logger("move_slowly_towards_point")
#     robot = await_robot(state, connection)
#
#     logger.debug(f"Moving slowly towards {point}")
#
#     while robot.distance_to_point(point) > ROBOT_TO_POINT_DISTANCE_BEFORE_BURST:
#         inst = Instruction(
#             name=CommandName.FORWARD,
#             type=InstructionType.COMMAND,
#             args=Arguments(seconds=ms_to_seconds(200), speed=30),
#         )
#         connection.send_message(Message(instruction=inst))
#         sleep(ms_to_seconds(200) + SLEEP_BUFFER_SECONDS)
#         robot = await_robot(state, connection)
#
#     logger.debug("Within burst distance — collecting")
#     burst_into_ball(state, connection, point, speed=burst_speed, ms=burst_ms)


def creep_forward_step(state: ArenaState, connection: RobotConnection, ms: int = 200, speed: int = 30) -> None:

    if state.robot is None:
        return
    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(ms), speed=speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(ms) + SLEEP_BUFFER_SECONDS)


# ── Abstracted Movement Helpers (1 layer up) ───────────────────────────────────────────────────────────────────

def go_to(state: ArenaState
          , connection: RobotConnection
          , point: tuple[float, float]
          , approach_radius: float= 0.0
          , distance_tolerance: float = GO_TO_DISTANCE_TOLERANCE) -> None:
    """
    1. Tag robot pos og tjek om den intercepter inflated bounding box
    2. Redirect robot og brug nærmeste* waypoint til at køre uden om\n
    2.1. Nærmeste* waypoint\n
    2.2. Tjek om den stadig kører igennem inflated bounding box\n
    2.3. Kør til waypoint som er tættest på originalt punkt\n
    3. Kør mod originalt punkt

    *Nærmeste = Kortest fra robot til punkt og waypoint til punkt
    """
    with state.lock:
        state.target_point = point
    logger = get_logger("go_to")
    logger.debug(f"Go_to point: ({point[0]:.1f}, {point[1]:.1f})  with approach radius: {approach_radius}")

    waypoints = calculate_shortest_waypoint_path(state, point) if state.cross is not None else []
    waypoints.append(point)

    logger.debug(f"Waypoints: {waypoints}")

    for i, waypoint in enumerate(waypoints):
        _iter = 0
        robot = await_robot(state, connection)
        current_target = waypoint

        logger.debug(f"target Waypoint: ({current_target[0]:.1f}, {current_target[1]:.2f})  current wp number: {i}")

        if approach_radius > 0.0 and i == len(waypoints) - 1:
            logger.debug(f"Approaching final waypoint: ({current_target[0]:.1f}, {current_target[1]:.1f})  rob's pos: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")
            # TODO: flyt det her ud til helper
            dx = waypoint[0] - robot.position[0]
            dy = waypoint[1] - robot.position[1]
            d = hypot(dx, dy)
            if d > approach_radius:
                scale = (d - approach_radius) / d
                current_target = (
                    robot.position[0] + scale * dx,
                    robot.position[1] + scale * dy
                )
            else:
                # robotten er allerede inden for approach radius
                logger.debug(f"Already within approach radius: {d:.2f} <= {approach_radius} exiting goto")
                return

        distance = dist_to_point(robot.position, current_target)

        logger.debug(f"current waypoint: {waypoint}, current target point: ({current_target[0]:.1f}, {current_target[1]:.1f})")

        while distance > distance_tolerance and _iter <= GO_TO_MAX_MOVES:
            if _iter == 0: # for debug
                logger.debug(f"Starting to move towards waypoint: ({current_target[0]:.1f}, {current_target[1]:.1f})  rob's pos: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")
            turn_to_point(state, connection, current_target)
            drive_forward(state, connection, current_target)

            robot = await_robot(state, connection)
            distance = dist_to_point(robot.position, current_target)
            _iter += 1

        logger.debug(f"At waypoint - iterations to get to wp: {_iter} rob's pos: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")
