from math import hypot
from src.lib.cross_waypoints import get_cross_waypoints
from src.model.cross import Cross
from src.lib.constants import EDGE_THRESHOLD, CORNER_THRESHOLD, ARENA_WIDTH_CM, ARENA_HEIGHT_CM, EDGE_BALL_STAGING_POINT_OFFSET, CROSS_ARM_LENGTH, CROSS_ZONE_PADDING
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
        width = ARENA_WIDTH_CM
        height = ARENA_HEIGHT_CM
        x, y = self.position

        # offset is padded with the ball's distance to the wall

        # Ball is near the EAST edge (x close to arena width)
        if x >= width - EDGE_THRESHOLD:
            dist_to_wall = width - x
            offset = EDGE_BALL_STAGING_POINT_OFFSET + dist_to_wall
            return [True, (width - offset, y)]

        # Ball is near the WEST edge (x close to 0)
        if x <= EDGE_THRESHOLD:
            dist_to_wall = x
            offset = EDGE_BALL_STAGING_POINT_OFFSET + dist_to_wall
            return [True, (offset, y)]

        # Ball is near the NORTH edge (y close to arena height, y increases northward)
        if y >= height - EDGE_THRESHOLD:
            dist_to_wall = height - y
            offset = EDGE_BALL_STAGING_POINT_OFFSET + dist_to_wall
            return [True, (x, height - offset)]

        # Ball is near the SOUTH edge (y close to 0)
        if y <= EDGE_THRESHOLD:
            dist_to_wall = y
            offset = EDGE_BALL_STAGING_POINT_OFFSET + dist_to_wall
            return [True, (x, offset)]

        # Not near any edge — return original position unchanged
        return [False, self.position]
    
    def is_corner_ball(self) -> bool:
        """Returns True if the ball is in the corner zone."""
        is_edge, _ = self.is_edge_ball()

        if not is_edge:
            return False

        return self.distance_to_nearest_corner() <= CORNER_THRESHOLD
    
    def nearest_corner(self) -> ArenaCorner:
        """
        Returns the corner this ball belongs to.

        Assumes the ball is already classified as a corner ball.
        """
        x, y = self.position

        width = ARENA_WIDTH_CM
        height = ARENA_HEIGHT_CM

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
        x, y = self.position

        width = ARENA_WIDTH_CM
        height = ARENA_HEIGHT_CM

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

    def distance_to_nearest_corner(self) -> float:
        """
        Returns the distance to the nearest corner measured along the
        relevant wall. Balls adjacent to two walls are treated as corner balls.
        """
        x, y = self.position
        width = ARENA_WIDTH_CM
        height = ARENA_HEIGHT_CM

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

    def is_within_cross_zone(self, cross: Cross) -> bool:
        """Check whether the ball is within the cross zone. Returns False if cross is not found."""
        if cross is None:
            return False

        cx, cy = cross.position

        offset = CROSS_ARM_LENGTH + CROSS_ZONE_PADDING

        cross_zone_corners = [
            (cx - offset, cy - offset),  # top-left
            (cx + offset, cy - offset),  # top-right
            (cx + offset, cy + offset),  # bottom-right
            (cx - offset, cy + offset),  # bottom-left
        ]

        return self._is_within_box(cross_zone_corners)

    def is_within_waypoint_zone(self, state) -> bool:
        """Check whether the ball is within the waypoint zone. Returns False if cross is not found."""
        if state.cross is None:
            return False

        waypoint_zone_corners = get_cross_waypoints(state.cross)
        return self._is_within_box(waypoint_zone_corners)

    def _is_within_box(self, box_corners: list[tuple[float, float]]):
        """
        Check whether `self.position` lies inside the quadrilateral defined by `box_corners`.
        `box_corners` must be ordered sequentially around the perimeter (CW or CCW).
        Uses the ray-casting algorithm, so it works for any convex or concave
        simple polygon, not just axis-aligned rectangles.
        """
        x, y = self.position
        inside = False
        n = len(box_corners)

        for i in range(n):
            x1, y1 = box_corners[i]
            x2, y2 = box_corners[(i + 1) % n]

            if (y1 > y) != (y2 > y):
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_intersect:
                    inside = not inside

        return inside


    def is_within_cross_zone(self, cross: Cross) -> bool:
        if cross is None:
            return False

        cx, cy = cross.position

        offset = CROSS_ARM_LENGTH + CROSS_ZONE_PADDING

        cross_zone_corners = [
            (cx - offset, cy - offset),  # top-left
            (cx + offset, cy - offset),  # top-right
            (cx + offset, cy + offset),  # bottom-right
            (cx - offset, cy + offset),  # bottom-left
        ]

        return self._is_within_box(cross_zone_corners)

    def _is_within_box(self, box_corners: list[tuple[float, float]]) -> bool:
        """
        Check whether `self.position` lies inside the quadrilateral defined by `box_corners`.
        `box_corners` must be ordered sequentially around the perimeter (CW or CCW).
        Uses the ray-casting algorithm, so it works for any convex or concave
        simple polygon, not just axis-aligned rectangles.
        """
        x, y = self.position
        inside = False
        n = len(box_corners)

        for i in range(n):
            x1, y1 = box_corners[i]
            x2, y2 = box_corners[(i + 1) % n]

            if (y1 > y) != (y2 > y):
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_intersect:
                    inside = not inside

        return inside

    def __repr__(self) -> str:
        return f"Ball(position={self.position}, is_vip={self.is_vip})"
