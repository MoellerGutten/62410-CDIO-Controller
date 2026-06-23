from enum import Enum
from src.model.arena_corner import ArenaCorner


class ArenaEdge(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

def get_corner_edges(corner: ArenaCorner) -> tuple[ArenaEdge, ArenaEdge]:
    match corner:
        case ArenaCorner.NORTH_EAST:
            return (ArenaEdge.NORTH, ArenaEdge.EAST)
        case ArenaCorner.NORTH_WEST:
            return (ArenaEdge.NORTH, ArenaEdge.WEST)
        case ArenaCorner.SOUTH_WEST:
            return (ArenaEdge.SOUTH, ArenaEdge.WEST)
        case ArenaCorner.SOUTH_EAST:
            return (ArenaEdge.SOUTH, ArenaEdge.EAST)


def get_other_corner_edge(corner: ArenaCorner, edge: ArenaEdge) -> ArenaEdge:
    corner_edges = get_corner_edges(corner)
    return corner_edges[0] if edge == corner_edges[0] else corner_edges[1]
