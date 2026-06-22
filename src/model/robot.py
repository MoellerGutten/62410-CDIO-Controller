from math import hypot, degrees, atan2, radians, cos, sin
import numpy as np
from src.lib.constants import ROBOT_WIDTH, ROBOT_LENGTH, DISTANCE_TO_POINT_TO_WITHIN_TURNING_HIT_RADIUS, DISTANCE_TO_ANGLE_TO_WITHIN_TURNING_HIT_RADIUS, LENGTH_OF_BOX_BEHIND_ROBOT
from src.debug.log import get_logger
from src.model.ball import Ball
from shapely.geometry import Point, Polygon

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
        self.robot_width_cm = ROBOT_WIDTH
        self.robot_length_cm = ROBOT_LENGTH

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def get_nearest_ball(self, balls: list[Ball]) -> Ball | None:
        """
        Return the ball closest to the robot, or None if the list is empty.
        Can also return VIP balls.
        """

        if balls is None or len(balls) == 0:
            return None

        return min(balls, key=lambda ball: self.distance_to_point(ball.position))
    
    def get_nearest_non_vip_ball(self, balls: list[Ball]) -> Ball | None:
        """
        Return the VIP ball that is closest to the robot, or None if there are no VIP balls on the field.
        """
        if balls is None or len(balls) == 0:
            return None
        if all(ball.is_vip for ball in balls):
            return None
        return min([ball for ball in balls if not ball.is_vip], key=lambda ball: self.distance_to_point(ball.position))

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
    
    def is_point_within_turning_hit_radius(self, point: tuple[float, float]) -> float:
        """Calculates if the given point is going to be hit, when the robot is turning"""
        return True if self.distance_to_point(point) <= DISTANCE_TO_POINT_TO_WITHIN_TURNING_HIT_RADIUS and abs(self.angle_to_point(point)) >= DISTANCE_TO_ANGLE_TO_WITHIN_TURNING_HIT_RADIUS else False
    

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
    
    def angle_to_heading(self, heading: float) -> float:
        """
        Signed angle (degrees) the robot must rotate to face *heading*.
        Positive = counter-clockwise, negative = clockwise.
        Result is in (-180, 180].
        """
        diff = (heading - self.orientation + 180) % 360 - 180
        return diff

    def is_facing_heading(self, heading: float, tolerance_deg: float = 5.0) -> bool:
        """Return True if the robot's orientation is within *tolerance_deg* of *heading*."""
        return abs(self.angle_to_heading(heading)) <= tolerance_deg

    def heading_vector(self) -> tuple[float, float]:
        """Unit vector in the direction the robot is currently facing."""
        rad = radians(self.orientation)
        return (cos(rad), sin(rad))

    def __repr__(self) -> str:
        return f"Robot(position={self.position}, orientation={self.orientation:.1f}°)"
    
    # ------------------------------------------------------------------
    # area helpers
    # ------------------------------------------------------------------

    def _get_area_behind(self):
        heading = (self.orientation + 180) % 360
        heading_rad = np.radians(heading)
        r = LENGTH_OF_BOX_BEHIND_ROBOT * np.array([cos(heading_rad), sin(heading_rad)])
        p0 = self.position
        perp = self.robot_width_cm/2 * np.array([-sin(heading_rad), cos(heading_rad)])
        p1 = p0 + r + perp
        p2 = p0 + perp
        p3 = p0 - perp
        p4 = p0 + r - perp

        return [p1, p2, p3, p4]
    
    def is_point_in_area_behind(self, point: tuple[float, float]):
        box = self._get_area_behind()
        polygon = Polygon(box)
        if polygon.contains(Point(point)):
            return True
        return False