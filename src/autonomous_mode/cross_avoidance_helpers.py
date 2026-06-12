from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.model.cross import Cross


def calculate_shortest_waypoint_path(state: ArenaState, connection: RobotConnection, point: tuple[float, float]) -> list[tuple[float, float]]:
    """
    Bruteforce hvilket waypoint man skal køre til for at få
    den kortest samlede tur fra robot -> waypoint (evt. -> waypoint 2) -> target punkt 
    """

    intersections = intersect_line_with_box(state.robot.position, point, state.cross.inflate_bounding_box(inflation_cm=10))
    result = []


    if not intersections or len(intersections) != 2:
        raise ValueError("Has to have two intersections")

    # Intersected edges touching = shortest path using 1 waypoint
    if edges_are_parallel(intersections[0], intersections[1]) == False:
        # TODO: points dont have the exact same coordinates, so this will have unintended behaviour
        result = tuple(set(intersections[0]) & (set(intersections[1])))[0] # dont ask about the index :)

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
            close_edge_point_index = 1
            far_edge_point_index = 0
        
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