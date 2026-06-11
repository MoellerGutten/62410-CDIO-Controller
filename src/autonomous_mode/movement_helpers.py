from src.state.state_manager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message, SequenceName
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.debug.log import get_logger
from src.model.cross import inflate_bounding_box
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

def _nudge_robot(connection: RobotConnection) -> None:
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
        if state.robot.is_facing_point(point, tolerance_deg=6.0):
            break

        angle = state.robot.angle_to_point(point)
        turn_ms    = max(100, min(500, int(abs(angle) * 5)))
        turn_speed = max(30, min(100, int(abs(angle) * 0.4)))

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

# ── Abstracted Movement Helpers (1 layer up) ───────────────────────────────────────────────────────────────────

def go_to(state: ArenaState, connection: RobotConnection, point: tuple[float, float]):
    """
    1. Tag robot pos og tjek om den intercepter inflated bounding box
    2. Redirect robot og brug nærmeste* waypoint til at køre uden om\n
    2.1. Nærmeste* waypoint\n
    2.2. Tjek om den stadig kører igennem inflated bounding box\n
    2.3. Kør til waypoint som er tættest på originalt punkt\n
    3. Kør mod originalt punkt
    
    *Nærmeste = Kortest fra robot til punkt og waypoint til punkt
    """

    pass

def calculate_shortest_waypoint_path(state: ArenaState, connection: RobotConnection, point: list[int]) -> list[tuple[float, float]]:
    """
    Bruteforce hvilket waypoint man skal køre til for at få
    den kortest samlede tur fra robot -> waypoint (evt. -> waypoint 2) -> target punkt 
    """

    intersections = intersect_line_with_box(state.robot.position, point, inflate_bounding_box(state.cross.bounding_box))
    result = ()


    if not intersections or len(intersections) != 2:
        raise ValueError("Has to have two intersections")

    # Intersected edges touching = shortest path using 1 waypoint
    if edges_are_parallel(intersections[0], intersections[1]) == False:
        result = tuple(set(intersections[0]) & (set(intersections[1])))

    # Edges are opposite = shortest path using 2 waypoints
    elif edges_are_parallel(intersections[0], intersections[1]) == True:
        if dist_to_point(state.robot.position, intersections[0][0]) < dist_to_point(state.robot.position, intersections[1][0]):
            close_edge_index = 0
            far_edge_index = 1
        else:
            close_edge_index = 1
            far_edge_index = 0
        closest_edge = intersections[close_edge_index]
        far_edge = intersections[far_edge_index]

        # Since vertex indexes are flipped in close and far edges we have to flip the index to go to the waypoints
        # that are in a "line". The cause of this is the for loop in the "intersect_line_with_box" function.
        if dist_to_point(state.robot.position, closest_edge[0]) < dist_to_point(state.robot.position, closest_edge[1]):
            close_edge_point_index = 0
            far_edge_point_index = 1
        else:
            close_edge_point_index = 0
            far_edge_point_index = 1
        
        waypoints = [closest_edge[close_edge_point_index], far_edge[far_edge_point_index]]
        result = waypoints

    return result

def dist_to_point(start: tuple[float, float], target_point: tuple[float, float]) -> float:
    return ((target_point[0] - start[0])**2 + (target_point[1] - start[1])**2)**0.5

def line_segment_intersect(p1, p2, p3, p4):
    """Find skæring mellem to linjestykker (p1-p2 og p3-p4).
    Returnerer skæringspunkt eller None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallelle linjer

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def intersect_line_with_box(line_start, line_end, box_points) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Find alle kanter i kassen som linjen skærer.

    Returns:
        Liste af tuples: ((kant_start, kant_slut), skæringspunkt)
    """
    results = []

    n = len(box_points)
    for i in range(n):
        edge_start = box_points[i]
        edge_end = box_points[(i + 1) % n]

        pt = line_segment_intersect(line_start, line_end, edge_start, edge_end)
        if pt is not None:
            if not any(abs(pt[0] - ex_pt[0]) < 1e-9 and abs(pt[1] - ex_pt[1]) < 1e-9
                       for _, ex_pt in results):
                results.append(((edge_start, edge_end)))

    return results


def edges_are_parallel(edge1, edge2):
    """Tjekker om to kanter er parallelle.

    Args:
        edge1: ((x1, y1), (x2, y2))
        edge2: ((x3, y3), (x4, y4))
    Returns: True hvis kanterne er parallelle
    """
    dx1 = edge1[1][0] - edge1[0][0]
    dy1 = edge1[1][1] - edge1[0][1]
    dx2 = edge2[1][0] - edge2[0][0]
    dy2 = edge2[1][1] - edge2[0][1]

    # Krydsprodukt == 0 betyder parallelle
    return abs(dx1 * dy2 - dy1 * dx2) < 1e-10