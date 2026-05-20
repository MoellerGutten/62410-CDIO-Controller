from threading import Lock
from src.model.ball import Ball
from src.model.robot import Robot
from src.model.cross import Cross
from src.model.corner import Corner

class ArenaState:
    def __init__(self):
        self.lock = Lock()
        self.robot: Robot | None = None
        self.balls: list[Ball] = []
        self.cross: Cross | None = None
        self.corners: list[Corner] = []

    def __repr__(self) -> str:
        ball_strs = ", ".join(repr(b) for b in self.balls)
        corner_strs = ", ".join(repr(c) for c in self.corners)
        return (
            f"ArenaState(\n"
            f"  robot:   {self.robot!r}\n"
            f"  cross:   {self.cross!r}\n"
            f"  balls:   [{ball_strs}]\n"
            f"  corners: [{corner_strs}]\n"
            f")"
        )

state = ArenaState()
