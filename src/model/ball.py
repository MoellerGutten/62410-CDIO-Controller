from math import hypot
from src.model.cross import Cross
from src.lib.constants import CROSS_ARM_LENGTH, CROSS_ZONE_PADDING


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
        if self.position[0] >= 150.0:
            return [True, (130.0, self.position[1])]
        if self.position[0] <= 10.0:
            return [True, (30.0, self.position[1])]
        if self.position[1] >= 105.0:
            return [True, (self.position[0], 90.0)]
        if self.position[1] <= 10.0:
            return [True, (self.position[0], 30.0)]
        return [False, self.position]

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
