from math import hypot, degrees, atan2, radians, cos, sin
from src.model.ball import Ball

class Robot:
    """Represents the robot on the field."""

    def __init__(self, position: tuple[float, float], orientation: float):
        """
        Args:
            position:    (x, y) pixel coordinates of the robot's centre.
            orientation: Heading in degrees. 0° = right (+x axis),
                         increasing counter-clockwise.
        """
        self.position = position
        self.orientation = orientation % 360
        self.robot_width_cm = 14
        self.robot_length_cm = 24

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def get_nearest_ball(self, balls: list[Ball]) -> Ball | None:
        """
        Return the ball closest to the robot, or None if the list is empty.
        """

        if balls is None or len(balls) == 0:
            return None

        return min(balls, key=lambda ball: self.distance_to_point(ball.position))

    def get_nearest_vip_ball(self, balls: list[Ball]) -> Ball | None:
        """
        Return the VIP (orange) ball closest to the robot,
        or None if no VIP ball exists.
        """
        vip_balls = [b for b in balls if b.is_vip]
        if not vip_balls:
            return None
        return min(vip_balls, key=lambda ball: self.distance_to_point(ball.position))

    def distance_to_point(self, point: tuple[float, float]) -> float:
        """Euclidean distance from the robot to an arbitrary (x, y) point."""
        return hypot(self.position[0] - point[0], self.position[1] - point[1])

    # ------------------------------------------------------------------
    # Orientation helpers
    # ------------------------------------------------------------------

    def bearing_to_point(self, point: tuple[float, float]) -> float:
        """
        Absolute bearing (degrees) from the robot to a point.
        0° = right (+x), increasing counter-clockwise, result in [0, 360).
        """
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        return degrees(atan2(dy, dx)) % 360

    def angle_to_point(self, point: tuple[float, float]) -> float:
        """
        Signed angle (degrees) the robot must rotate to face a point.
        Positive = counter-clockwise, negative = clockwise.
        Result is in (-180, 180].
        """
        bearing = self.bearing_to_point(point)
        diff = (bearing - self.orientation + 180) % 360 - 180
        return diff

    def is_facing_point(self, point: tuple[float, float], tolerance_deg: float = 5.0) -> bool:
        """Return True if the robot is facing the point within *tolerance_deg*."""
        return abs(self.angle_to_point(point)) <= tolerance_deg

    def heading_vector(self) -> tuple[float, float]:
        """Unit vector in the direction the robot is currently facing."""
        rad = radians(self.orientation)
        return (cos(rad), sin(rad))

    def __repr__(self) -> str:
        return f"Robot(position={self.position}, orientation={self.orientation:.1f}°)"
