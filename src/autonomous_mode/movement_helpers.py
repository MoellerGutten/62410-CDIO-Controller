from math import hypot

from sympy.strategies.branch import debug

from src.autonomous_mode.cross_avoidance_helpers import calculate_shortest_waypoint_path, dist_to_point
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


def turn_to_point(state: ArenaState, connection: RobotConnection, point: tuple[float, float], precise_mode: bool = False) -> None:
    from src.autonomous_mode.state_helpers import await_robot
    robot = await_robot(state, connection)
    while True:
        if robot is None or point is None: break

        tolerance_deg = 2.0 if precise_mode else 6.0
        if robot.is_facing_point(point, tolerance_deg): break

        angle = state.robot.angle_to_point(point)
        if (precise_mode):
            turn_ms    = 100
            turn_speed = 10
        else:
            turn_ms    = max(100, min(500, int(abs(angle) * 5)))
            turn_speed = max(30, min(100, int(abs(angle) * 0.4)))

        command = CommandName.TANK_RIGHT if angle > 0 else CommandName.TANK_LEFT
        l_speed = turn_speed if angle > 0 else -turn_speed
        r_speed = -turn_speed if angle > 0 else turn_speed

        # get_logger().debug(f"Turning: command={command}, l_speed={l_speed}, r_speed={r_speed}")

        inst = Instruction(
            name=command,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=turn_ms / 1000, lspeed=l_speed, rspeed=r_speed),
        )
        connection.send_message(Message(instruction=inst))
        sleep(turn_ms / 1000 + 0.05)

        # update_state(state)
        robot = await_robot(state, connection)


def drive_forward(state: ArenaState, connection: RobotConnection, point: tuple[float, float]) -> None:
    # if state.robot is None:
    #     return
    from src.autonomous_mode.state_helpers import await_robot
    robot = await_robot(state, connection)

    distance = robot.distance_to_point(point)
    fwd_ms = max(100, min(2000, int(distance * 10)))
    fwd_speed =  max(30, min(100, int(distance * 0.8)))

    # get_logger().debug(f"Driving: distance={distance:.2f}, speed={fwd_speed}, duration={fwd_ms}ms")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=fwd_ms / 1000, speed=fwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(fwd_ms / 1000 + 0.05)


def drive_backward(state: ArenaState, connection: RobotConnection) -> None:
    if state.robot is None:
        return

    fwd_ms = 500
    fwd_speed =  50

    get_logger().debug(f"Driving backward: speed={fwd_speed}, duration={fwd_ms}ms")

    inst = Instruction(
        name=CommandName.BACKWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=fwd_ms / 1000, speed=fwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(fwd_ms / 1000 + 0.05)


def burst_into_ball(state: ArenaState, connection: RobotConnection, point: list[int]) -> None:
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


# ── Abstracted Movement Helpers (1 layer up) ───────────────────────────────────────────────────────────────────

def go_to(state: ArenaState
          , connection: RobotConnection
          , point: tuple[float, float]
          , approach_radius: float= 0.0) -> None:
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
    logger = get_logger("go_to")
    logger.debug(f"Go_to point: ({point[0]:.1f}, {point[0]:.1f})  with approach radius: {approach_radius}")

    distance = dist_to_point(robot.position, point)
    distance_tolerance = 6.0
    max_iter = 50
    waypoints = []
    if state.cross:
        waypoints = calculate_shortest_waypoint_path(state, connection, point)
    waypoints.append(point)

    logger.debug(f"Waypoints: {waypoints}")

    for i, waypoint in enumerate(waypoints):
        _iter = 0
        current_target = waypoint

        logger.debug(f"target Waypoint: ({current_target[0]:.1f}, {current_target[1]:.2f})  current wp number: {i}")

        if approach_radius > 0.0 and i == len(waypoints) - 1:
            robot = await_robot(state, connection)  # we need rob's current pos
            logger.debug(f"Approaching final waypoint: ({current_target[0]:.1f}, {current_target[1]:.1f})  rob's pos: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")

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

        logger.debug(f"current waypoint: {waypoint}, current target point: ({current_target[0]:.1f}, {current_target[1]:.1f})")

        while distance > distance_tolerance and _iter <= max_iter:
            if _iter == 0: # for debug
                logger.debug(f"Starting to move towards waypoint: ({current_target[0]:.1f}, {current_target[1]:.1f})  rob's pos: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")
            turn_to_point(state, connection, current_target)
            drive_forward(state, connection, current_target)

            distance = dist_to_point(robot.position, current_target)
            _iter += 1
            robot = await_robot(state, connection)

        logger.debug(f"At waypoint - iterations to get to wp: {_iter} rob's pos: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")
