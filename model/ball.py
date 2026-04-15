import math


class Ball:
    """Represents a table tennis ball detected on the field."""

    def __init__(self, position: tuple[int, int], is_vip: bool = False):
        """
        Args:
            position: (x, y) pixel coordinates of the ball.
            is_vip: Whether this is the special VIP ball.
        """
        self.position = position
        self.is_vip = is_vip

    def distance_to(self, other: "Ball") -> float:
        """Euclidean distance to another Ball."""
        return math.hypot(
            self.position[0] - other.position[0],
            self.position[1] - other.position[1],
        )

    def distance_to_point(self, point: tuple[int, int]) -> float:
        """Euclidean distance to an arbitrary (x, y) point."""
        return math.hypot(self.position[0] - point[0], self.position[1] - point[1])

    def __repr__(self) -> str:
        return f"Ball(position={self.position}, is_vip={self.is_vip})"
