from threading import Lock
from src.model.ball import Ball
from src.model.robot import Robot
from src.model.cross import Cross
from src.model.corner import Corner

class ArenaState:
    """
    Represents a snapshot of the state of the entire arena, including robot, balls, ball count estimate, cross, and corners.
    """
    def __init__(self):
        self.lock = Lock()
        self.robot: Robot | None = None
        self.balls: list[Ball] = []
        self.estimated_ball_count: int = 11 # 11 balls on the field
        self.estimated_balls_in_robot: int = 0
        self.estimated_balls_delivered: int = 0
        self.cross: Cross | None = None
        self.corners: list[Corner] = []
        self.all_balls_delivered: bool = False
        self.target_point: tuple[float, float] | None = None
        self.start_time = None 
        

    def __repr__(self) -> str:
        ball_strs = ", ".join(repr(b) for b in self.balls)
        corner_strs = ", ".join(repr(c) for c in self.corners)
        return (
            f"ArenaState(\n"
            f"  robot:   {self.robot!r}\n"
            f"  cross:   {self.cross!r}\n"
            f"  balls:   [{ball_strs}]\n"
            f"  estimated_ball_count:   [{self.estimated_ball_count}]\n"
            f"  corners: [{corner_strs}]\n"
            f")"
        )
    

state = ArenaState()
