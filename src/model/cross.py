from math import hypot


class Cross:
    """Represents the cross-shaped obstacle on the field."""

    def __init__(self, position: tuple[float, float], 
                 orientation: float, 
                 bounding_box: list[tuple[float, float]],
                 side_length: float = 10):
        """
        Args:
            position:    (x, y) pixel coordinates of the cross centre.
            orientation: Rotation of the cross in degrees, clamped to [0, 90]
                         due to its 4-fold symmetry.
            side_length:        Length of a side of the bounding box
        """
        self.position = position
        self.orientation = orientation # no need to % 90 here as it is done in arena tracker
        self.side_length = side_length
        self.bounding_box = bounding_box

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def distance_to_point(self, point: tuple[float, float]) -> float:
        """Euclidean distance from the cross centre to an arbitrary (x, y) point."""
        return hypot(self.position[0] - point[0], self.position[1] - point[1])

    # ------------------------------------------------------------------
    # Bounding box/waypoint zone helpers
    # ------------------------------------------------------------------

    def inflate_bounding_box(self, inflation_cm) -> list[tuple[float, float]]:
        result = []

        # The corners are slightly off, so we just force them to be the same :-)
        
        result.append((self.bounding_box[0][0] - inflation_cm, self.bounding_box[0][1] + inflation_cm))
        result.append((self.bounding_box[1][0] + inflation_cm, self.bounding_box[0][1] + inflation_cm))
        result.append((self.bounding_box[1][0] + inflation_cm, self.bounding_box[2][1] - inflation_cm))
        result.append((self.bounding_box[0][0] - inflation_cm, self.bounding_box[2][1] - inflation_cm))

        """
        Old version:
        result.append((self.bounding_box[0][0] - inflation_cm, self.bounding_box[0][1] + inflation_cm))
        result.append((self.bounding_box[1][0] + inflation_cm, self.bounding_box[1][1] + inflation_cm))
        result.append((self.bounding_box[2][0] + inflation_cm, self.bounding_box[2][1] - inflation_cm))
        result.append((self.bounding_box[3][0] - inflation_cm, self.bounding_box[3][1] - inflation_cm))
        """

        return result

    # ------------------------------------------------------------------
    # Orientation helpers
    # ------------------------------------------------------------------

    def arm_angles(self) -> list[float]:
        """
        Returns the absolute angles (degrees) of all four arms of the cross,
        derived from the cross's orientation. Results are in [0, 360].
        """
        base = self.orientation
        return [base % 360, (base + 90) % 360, (base + 180) % 360, (base + 270) % 360]

    def nearest_arm_angle(self, angle: float) -> float:
        """
        Returns the arm angle (degrees) closest to the given absolute *angle*.
        Useful for checking alignment when approaching the cross.
        """
        arms = self.arm_angles()
        return min(arms, key=lambda a: abs((a - angle + 180) % 360 - 180))

    def get_point_quadrant(self, point: tuple[float, float]) -> int:
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]

        if dx >= 0 and dy >= 0:
            return 1
        elif dx < 0 and dy >= 0:
            return 2
        elif dx < 0 and dy < 0:
            return 3
        else:
            return 4

    def __repr__(self) -> str:
        return f"Cross(position={self.position}, orientation={self.orientation:.1f}°, bounding_box={self.bounding_box})"
