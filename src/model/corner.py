from math import hypot


class Corner:
    """Represents one detected corner of the field boundary."""

    def __init__(self, position: tuple[float, float]):
        """
        Args:
            position: (x, y) pixel coordinates of the corner.
        """
        self.position = position

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def distance_to(self, other: "Corner") -> float:
        """Euclidean distance to another Corner."""
        return hypot(
            self.position[0] - other.position[0],
            self.position[1] - other.position[1],
        )

    def distance_to_point(self, point: tuple[float, float]) -> float:
        """Euclidean distance to an arbitrary (x, y) point."""
        return hypot(self.position[0] - point[0], self.position[1] - point[1])

    # ------------------------------------------------------------------
    # Factory / collection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def bounding_box(corners: list["Corner"]) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Given a list of Corner objects, returns the axis-aligned bounding box
        as ((min_x, min_y), (max_x, max_y)).
        """
        xs = [c.position[0] for c in corners]
        ys = [c.position[1] for c in corners]
        return (min(xs), min(ys)), (max(xs), max(ys))

    @staticmethod
    def centroid(corners: list["Corner"]) -> tuple[float, float]:
        """Returns the centroid (average position) of a list of corners."""
        n = len(corners)
        return (
            sum(c.position[0] for c in corners) / n,
            sum(c.position[1] for c in corners) / n,
        )

    def __repr__(self) -> str:
        return f"Corner(position={self.position})"
