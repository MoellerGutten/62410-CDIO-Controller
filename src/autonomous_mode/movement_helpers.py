from math import hypot
from src.autonomous_mode.cross_avoidance_helpers import calculate_shortest_waypoint_path, dist_to_point
from protocol import CommandName, Arguments, Instruction, InstructionType, Message
from src.lib.connection import RobotConnection
from src.lib.cross_waypoints import get_cross_waypoints
from src.model.arena_state import ArenaState
from src.debug.log import get_logger
from src.lib.constants import BALL_INTAKE_ON_FOR_SECONDS, BALL_INTAKE_SPEED, CROSS_ZONE_BACKWARD_MS, CROSS_ZONE_BACKWARD_SPEED, EJACULATE_SPEED, EJACULATE_SECONDS, GENTLE_BURST_DEFAULT_MAX_ITER, \
TURN_TO_POINT_PRECISE_TOLERANCE, TURN_TO_POINT_TOLERANCE, SLEEP_BUFFER_SECONDS, BACKWARD_SPEED, BACKWARD_MS, BURST_FORWARD_SPEED, GO_TO_MAX_MOVES, \
GO_TO_DISTANCE_TOLERANCE, DISTANCE_OF_WHEN_ROBOT_OUTSIDE_BALL_HIT_RADIUS_FRONT, DISTANCE_OF_WHEN_ROBOT_OUTSIDE_BALL_HIT_RADIUS_BACK, WEST_HEADING, EAST_HEADING
from src.lib.algorithms import burst_forward_ms, small_burst_forward_ms, turn_to_point_turn_ms, turn_to_point_turn_speed, drive_forward_ms, drive_forward_speed
from src.lib.time import ms_to_seconds
from src.autonomous_mode.state_helpers import await_robot
from src.model.ball import Ball
from src.model.robot import Robot
from time import time, sleep
from shapely.geometry import Point, Polygon

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
        args=Arguments(speed=EJACULATE_SPEED, seconds=EJACULATE_SECONDS, block=True), # blocking to prevent short circuiting ejaculation
    )
    connection.send_message(Message(instruction=inst))
    _start_ball_intake(connection) # start intake after ejaculation

def turn_to_point(state: ArenaState, connection: RobotConnection, point: tuple[float, float], precise_mode: bool = False) -> None:
    logger = get_logger("turn_to_point")
    robot = await_robot(state, connection)

    while True:
        if point is None:
            break

        tol_deg = TURN_TO_POINT_PRECISE_TOLERANCE if precise_mode else TURN_TO_POINT_TOLERANCE
        if robot.is_facing_point(point, tol_deg):
            break

        angle = robot.angle_to_point(point)
        turn_ms    = turn_to_point_turn_ms(angle)
        turn_speed = turn_to_point_turn_speed(angle)

        turn_ms = turn_ms/(1 + precise_mode)
    
        command = CommandName.TANK_RIGHT if angle > 0 else CommandName.TANK_LEFT
        l_speed = turn_speed if angle > 0 else -turn_speed
        r_speed = -turn_speed if angle > 0 else turn_speed

        if not robot.can_safely_turn_to_point(state.cross, point):
            if not robot.can_safely_turn_to_point_any_way(state.cross, point):
                logger.debug("Can't safely turn to point in any direction, backing up")
                burst_backward(state, connection)
                robot = await_robot(state, connection)
                continue

            logger.debug("Can't safely turn to point in the shortest direction, going the other way around")
            l_speed *= -1.3
            r_speed *= -1.3

        inst = Instruction(
            name=command,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=ms_to_seconds(turn_ms), lspeed=l_speed, rspeed=r_speed),
        )
        connection.send_message(Message(instruction=inst))
        sleep(ms_to_seconds(turn_ms) + SLEEP_BUFFER_SECONDS)

        robot = await_robot(state, connection)

def turn_to_heading(
    state: ArenaState,
    connection: RobotConnection,
    heading_deg: float,
    precise_mode: bool = False
) -> None:
    logger = get_logger("turn_to_point")
    robot = await_robot(state, connection)
    tol_deg = TURN_TO_POINT_PRECISE_TOLERANCE if precise_mode else TURN_TO_POINT_TOLERANCE

    while not robot.is_facing_heading(heading_deg, tol_deg):
        angle = robot.angle_to_heading(heading_deg)
        turn_ms    = turn_to_point_turn_ms(angle)
        turn_speed = turn_to_point_turn_speed(angle)
    
        command = CommandName.TANK_RIGHT if angle > 0 else CommandName.TANK_LEFT
        l_speed = turn_speed if angle > 0 else -turn_speed
        r_speed = -turn_speed if angle > 0 else turn_speed

        if not robot.can_safely_turn_to_heading(state.cross, heading_deg):
            if not robot.can_safely_turn_to_heading_any_way(state.cross, heading_deg):
                logger.debug("Can't safely turn to heading in any direction, backing up")
                burst_backward(state, connection)
                robot = await_robot(state, connection)
                continue

            logger.debug("Can't safely turn to heading in the shortest direction, going the other way around")
            l_speed *= -1.3
            r_speed *= -1.3

        inst = Instruction(
            name=command,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=ms_to_seconds(turn_ms), lspeed=l_speed, rspeed=r_speed),
        )
        connection.send_message(Message(instruction=inst))
        sleep(ms_to_seconds(turn_ms) + SLEEP_BUFFER_SECONDS)

        robot = await_robot(state, connection)
    
    logger.debug(f"Now facing heading {heading_deg} with tolerance {tol_deg}, current heading {robot.orientation}")


def drive_forward(state: ArenaState, connection: RobotConnection, point: tuple[float, float]) -> None:
  

    robot = await_robot(state, connection)
        

    logger = get_logger("drive_forward")

    distance = robot.distance_to_point(point)
    fwd_ms = drive_forward_ms(distance)
    fwd_speed =  drive_forward_speed(distance)

    logger.debug(f"forward speed {fwd_speed} m/s forward time: {fwd_ms}")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(fwd_ms), speed=fwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(fwd_ms) + SLEEP_BUFFER_SECONDS)

def burst_backward(state: ArenaState, connection: RobotConnection) -> None:
    logger = get_logger("burst_backward")

    if state.robot is None:
        return

    logger.debug("Bursting backwards")

    bwd_ms = BACKWARD_MS
    bwd_speed =  BACKWARD_SPEED

    inst = Instruction(
        name=CommandName.BACKWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(bwd_ms), speed=bwd_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(bwd_ms) + SLEEP_BUFFER_SECONDS)

def drive_backward_speed_ms(state: ArenaState, connection: RobotConnection, speed: float, ms: float) -> None:
    if state.robot is None:
        return
    
    inst = Instruction(
        name=CommandName.BACKWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(ms), speed=speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(ms) + SLEEP_BUFFER_SECONDS)


def creep_forward_step(state: ArenaState, connection: RobotConnection, ms: int = 100, speed: int = 30) -> None:
    if state.robot is None:
        return
    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(ms), speed=speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(ms) + SLEEP_BUFFER_SECONDS)


def escape_cross_zone(state: ArenaState, connection: RobotConnection):
    logger = get_logger("escape_cross_zone")
    waypoints = get_cross_waypoints(state.cross)
    if not waypoints:
        return
    logger.debug("Reversing out of the cross zone")
    # gentle, longer retreat tuned for the cross zone (see CROSS_ZONE_BACKWARD_* constants),
    # not the general drive_backward speed/duration
    inst = Instruction(
        name=CommandName.BACKWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(CROSS_ZONE_BACKWARD_MS), speed=CROSS_ZONE_BACKWARD_SPEED),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(CROSS_ZONE_BACKWARD_MS) + SLEEP_BUFFER_SECONDS)



def burst_into_ball(state: ArenaState, connection: RobotConnection, point: tuple[float, float]) -> None:
    robot = await_robot(state, connection)

    distance = robot.distance_to_point(point)
    burst_ms = burst_forward_ms(distance)
    burst_speed = BURST_FORWARD_SPEED

    logger = get_logger("burst_into_ball")
    logger.debug(f"Busting. burst ms: {burst_ms}, burst distance: {distance} cm")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(burst_ms), speed=burst_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(burst_ms) + SLEEP_BUFFER_SECONDS)

def burst_into_ball_slightly_smaller(state: ArenaState, connection: RobotConnection, point: tuple[float, float]) -> None:
    robot = await_robot(state, connection)

    distance = robot.distance_to_point(point)
    burst_ms = small_burst_forward_ms(distance)
    burst_speed = BURST_FORWARD_SPEED

    logger = get_logger("burst_into_ball")
    logger.debug(f"Busting. burst ms: {burst_ms}, burst distance: {distance} cm")

    inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(burst_ms), speed=burst_speed),
    )
    connection.send_message(Message(instruction=inst))
    sleep(ms_to_seconds(burst_ms) + SLEEP_BUFFER_SECONDS)

def move_slowly_towards_point(state: ArenaState, connection: RobotConnection, point: tuple[float, float]) -> None:
    logger = get_logger("move_slowly_towards_point")
    robot = await_robot(state, connection)

    logger.debug(f"Moving slowly towards {point}")

    while robot.distance_to_point(point) > 10:
        inst = Instruction(
        name=CommandName.FORWARD,
        type=InstructionType.COMMAND,
        args=Arguments(seconds=ms_to_seconds(200), speed=30),
        )
        connection.send_message(Message(instruction=inst))
        sleep(ms_to_seconds(200) + SLEEP_BUFFER_SECONDS)
        robot = await_robot(state, connection)


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
    with state.lock:
        state.target_point = point
    logger = get_logger("go_to")

    waypoints = calculate_shortest_waypoint_path(state, point) if state.cross is not None else []
    waypoint_zone = Polygon(get_cross_waypoints(state.cross))
    robot = await_robot(state, connection)
    if waypoint_zone.contains(Point(robot.position)) and state.cross is not None:
        logger.debug("Robot is in waypoint zone, escape before going to target")
        # robot is in waypoint zone, escape before going to target
        cross_quadrant = state.cross.get_point_quadrant(robot.position)
        escape_heading = EAST_HEADING if cross_quadrant == 1 or cross_quadrant == 4 else WEST_HEADING
        logger.debug(f"Turning to {escape_heading}")
        turn_to_heading(state, connection, escape_heading)
        while waypoint_zone.contains(Point(robot.position)):
            escape_point = robot.get_point_in_front(30)
            logger.debug(f"Escaping towards {escape_point}")
            drive_forward(state, connection, escape_point)
            robot = await_robot(state, connection)
        logger.debug("Escaped waypoint zone, recalculate waypoint path to target")
        waypoints = calculate_shortest_waypoint_path(state, point) if state.cross is not None else []

    waypoints.append(point)

    logger.debug(f"Waypoints: {waypoints}")

    for i, waypoint in enumerate(waypoints):
        _iter = 0
        robot = await_robot(state, connection)
        current_target = waypoint

        if approach_radius > 0.0 and i == len(waypoints) - 1:
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
        logger.debug(f"rob's pos: {robot.position[0]:.1f}, {robot.position[1]:.1f} target: {current_target} distance to target: {distance:.1f}")
        while distance > GO_TO_DISTANCE_TOLERANCE and _iter <= GO_TO_MAX_MOVES:
            turn_to_point(state, connection, current_target)
            drive_forward(state, connection, current_target)

            robot = await_robot(state, connection)
            distance = dist_to_point(robot.position, current_target)
            logger.debug(f"rob's pos: {robot.position[0]:.1f}, {robot.position[1]:.1f} target: {current_target} distance to target: {distance:.1f}")

            _iter += 1

        logger.debug(f"At waypoint - iterations to get to wp: {_iter} rob's pos: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")


def handle_balls_in_radius(state: ArenaState, robot: Robot, connection: RobotConnection, ball: Ball):
    logger = get_logger("handle_balls_in_radius")
    _max_iter = 3
    _iter = 0

    if robot.is_point_in_area_behind(ball.position):
        logger.debug("Ball is in area behind robot, driving forwards")

        while (robot.distance_to_point(ball.position) < DISTANCE_OF_WHEN_ROBOT_OUTSIDE_BALL_HIT_RADIUS_FRONT):
            if not robot.can_safely_drive_forward(state.cross, 10):
                logger.debug("Robot can not safely drive forwards, stop driving")
                return # return, not break
            
            if _iter >= _max_iter:
                logger.debug(f"Reached {_iter} iterations, stopping")
                return #return, not break

            drive_forward(state, connection, ball.position)
            robot = await_robot(state, connection)
            _iter += 1

    elif robot.is_point_within_turning_hit_radius(ball.position):
        logger.debug("Ball is in radius, driving backwards")

        while (robot.distance_to_point(ball.position) < DISTANCE_OF_WHEN_ROBOT_OUTSIDE_BALL_HIT_RADIUS_BACK):
            if not robot.can_safely_drive_backward(state.cross, 10):
                logger.debug("Robot can not safely drive backwards, stop backing")
                return # return, not break
            
            if _iter >= _max_iter:
                logger.debug(f"Reached {_iter} iterations, stopping")
                return

            burst_backward(state, connection)
            robot = await_robot(state, connection)
            _iter += 1


def gentle_burst(state: ArenaState, connection: RobotConnection, target_point: tuple[float, float], target_range: float, max_iter: float = GENTLE_BURST_DEFAULT_MAX_ITER):
    logger = get_logger("gentle_burst")
    _iter = 0
    while True:
        distance_to_target = await_robot(state, connection).distance_to_point(target_point)

        if _iter >= max_iter:
            logger.debug("Reached max iterations, stopping")
            break

        if distance_to_target <= target_range:
            logger.debug("Distance within limit, stopping")
            break

        burst_ms = 100
        burst_speed = 30
        inst = Instruction(
            name=CommandName.FORWARD,
            type=InstructionType.COMMAND,
            args=Arguments(seconds=ms_to_seconds(burst_ms), speed=burst_speed  * (3/2)),
        )
        connection.send_message(Message(instruction=inst))
        sleep(ms_to_seconds(burst_ms) + SLEEP_BUFFER_SECONDS)
        _iter += 1

def gentle_burst_edge_ball(
    state: ArenaState,
    connection: RobotConnection,
    target_point: tuple[float, float],
    target_range: float,
    max_iter: int = 10,
):
    logger = get_logger("gentle_burst_edge_ball")

    _iter = 0

    while True:
        distance = await_robot(state, connection).distance_to_point(target_point)

        if _iter >= max_iter:
            logger.debug("Reached max iterations")
            break

        logger.debug(f"Distance to posision: {distance}, target_range: {target_point}")
        if distance <= target_range:
            logger.debug("Distance within limit")
            break

        if distance > 24:
            burst_ms = 100
            burst_speed = 30
        elif distance > 20:
            burst_ms = 100
            burst_speed = 33
        else:
            burst_ms = 100
            burst_speed = 10

        logger.debug(f"distance={distance:.1f}, "f"burst_ms={burst_ms}, "f"burst_speed={burst_speed}")

        inst = Instruction(
            name=CommandName.FORWARD,
            type=InstructionType.COMMAND,
            args=Arguments(
                seconds=ms_to_seconds(burst_ms),
                speed=burst_speed * (3/2),
            ),
        )

        connection.send_message(Message(instruction=inst))
        sleep(ms_to_seconds(burst_ms) + SLEEP_BUFFER_SECONDS)

        _iter += 1
