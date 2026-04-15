import math


class Cross:
    """Represents the cross-shaped obstacle on the field."""

    def __init__(self, position: tuple[int, int], orientation: float):
        """
        Args:
            position:    (x, y) pixel coordinates of the cross centre.
            orientation: Rotation of the cross in degrees, clamped to [0, 90)
                         due to its 4-fold symmetry.
        """
        self.position = position
        self.orientation = orientation % 90

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def distance_to_point(self, point: tuple[int, int]) -> float:
        """Euclidean distance from the cross centre to an arbitrary (x, y) point."""
        return math.hypot(self.position[0] - point[0], self.position[1] - point[1])

    # ------------------------------------------------------------------
    # Orientation helpers
    # ------------------------------------------------------------------

    def arm_angles(self) -> list[float]:
        """
        Returns the absolute angles (degrees) of all four arms of the cross,
        derived from the cross's orientation. Results are in [0, 360).
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

    def __repr__(self) -> str:
        return f"Cross(position={self.position}, orientation={self.orientation:.1f}°)"
