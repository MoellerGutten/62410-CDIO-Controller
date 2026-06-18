from math import hypot
from src.lib.constants import EDGE_THRESHOLD, CORNER_THRESHOLD
from src.model.arena_corner import ArenaCorner
from src.model.arena_edge import ArenaEdge

class Ball:
    """Represents a table tennis ball detected on the field."""

    def __init__(self, position: tuple[float, float], is_vip: bool = False):
        """
        Args:
            position: (x, y) pixel coordinates of the ball.
            is_vip: Whether this is the special VIP ball.
        """
        self.position = position
        self.is_vip = is_vip

    def distance_to(self, other: "Ball") -> float:
        """Euclidean distance to another Ball."""
        return hypot(
            self.position[0] - other.position[0],
            self.position[1] - other.position[1],
        )

    def distance_to_point(self, point: tuple[float, float]) -> float:
        """Euclidean distance to an arbitrary (x, y) point."""
        return hypot(self.position[0] - point[0], self.position[1] - point[1])

    def is_edge_ball(self) -> bool:
        """Returns a boolean based on if the ball is next to an edge"""
        from src.state.arena_config import ArenaConfig
        width = ArenaConfig.width_cm
        height = ArenaConfig.height_cm
        if self.position[0] >= width - EDGE_THRESHOLD:
            return [True, (130.0, self.position[1])]
        if self.position[0] <= EDGE_THRESHOLD:
            return [True, (30.0, self.position[1])]
        if self.position[1] >= height - EDGE_THRESHOLD:
            return [True, (self.position[0], 90.0)]
        if self.position[1] <= EDGE_THRESHOLD:
            return [True, (self.position[0], 30.0)]
        return [False, self.position]
    
    def is_corner_ball(self) -> bool:
        """Returns True if the ball is in the corner zone."""
        is_edge, _ = self.is_edge_ball()

        if not is_edge:
            return False

        return self._distance_to_nearest_corner() <= CORNER_THRESHOLD
    
    def nearest_corner(self) -> ArenaCorner:
        """
        Returns the corner this ball belongs to.

        Assumes the ball is already classified as a corner ball.
        """
        from src.state.arena_config import ArenaConfig
        x, y = self.position

        width = ArenaConfig.width_cm
        height = ArenaConfig.height_cm

        # Closer to west than east
        if x < width / 2:
            # Closer to south than north
            if y < height / 2:
                return ArenaCorner.SOUTH_WEST
            return ArenaCorner.NORTH_WEST

        # Closer to east than west
        if y < height / 2:
            return ArenaCorner.SOUTH_EAST

        return ArenaCorner.NORTH_EAST
    
    def nearest_edge(self) -> ArenaEdge:
        """
        Returns the nearest arena edge the ball is touching/closest to.
        Assumes the ball is already classified as an edge ball.
        """
        from src.state.arena_config import ArenaConfig

        x, y = self.position

        width = ArenaConfig.width_cm
        height = ArenaConfig.height_cm

        # Distances to each edge
        dist_west = x
        dist_east = width - x
        dist_south = y
        dist_north = height - y

        # Return the closest edge
        if dist_west <= dist_east and dist_west <= dist_south and dist_west <= dist_north:
            return ArenaEdge.WEST
        if dist_east <= dist_west and dist_east <= dist_south and dist_east <= dist_north:
            return ArenaEdge.EAST
        if dist_south <= dist_west and dist_south <= dist_east and dist_south <= dist_north:
            return ArenaEdge.SOUTH
        return ArenaEdge.NORTH

    def _distance_to_nearest_corner(self) -> float:
        """
        Returns the distance to the nearest corner measured along the
        relevant wall. Balls adjacent to two walls are treated as corner balls.
        """
        from src.state.arena_config import ArenaConfig
        x, y = self.position
        width = ArenaConfig.width_cm
        height = ArenaConfig.height_cm

        near_west = x <= EDGE_THRESHOLD
        near_east = x >= width - EDGE_THRESHOLD
        near_south = y <= EDGE_THRESHOLD
        near_north = y >= height - EDGE_THRESHOLD

        # Ball is adjacent to two walls -> definitely in a corner
        if (near_west or near_east) and (near_south or near_north):
            return 0.0 # return 0 so ball is considered corner ball

        # West or east wall
        if near_west or near_east:
            return min(y, height - y)

        # South or north wall
        if near_south or near_north:
            return min(x, width - x)

        return float("inf") # not a corner ball

    def __repr__(self) -> str:
        return f"Ball(position={self.position}, is_vip={self.is_vip})"
