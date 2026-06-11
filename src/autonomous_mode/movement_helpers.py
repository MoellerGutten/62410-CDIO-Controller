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

def _go_to(state: ArenaState, connection: RobotConnection, point: list[int]):
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

def _calculate_shortest_waypoint_path(state: ArenaState, connection: RobotConnection, point: list[int]):
    """
    Bruteforce hvilket waypoint man skal køre til for at få
    den kortest samlede tur fra robot -> waypoint -> target punkt 
    """
    
    pass


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


def intersect_line_with_box(line_start, line_end, box_points):
    """Find alle skæringer mellem en linje og en kasse (4 punkter).

    Args:
        line_start: (x, y) - linjens startpunkt
        line_end:   (x, y) - linjens slutpunkt
        box_points: liste af 4 (x, y) punkter i rækkefølge

    Returns:
        Liste med 1 eller 2 skæringspunkter
    """
    intersections = []

    # Kasser har 4 kanter: 0-1, 1-2, 2-3, 3-0
    n = len(box_points)
    for i in range(n):
        edge_start = box_points[i]
        edge_end = box_points[(i + 1) % n]

        pt = line_segment_intersect(line_start, line_end, edge_start, edge_end)
        if pt is not None:
            # Undgå dubletter (hjørneskæringer)
            if not any(abs(pt[0] - ex[0]) < 1e-9 and abs(pt[1] - ex[1]) < 1e-9
                       for ex in intersections):
                intersections.append(pt)

    return intersections


def intersect_line_with_box(line_start, line_end, box_points):
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
                results.append(((edge_start, edge_end), pt))

    return results