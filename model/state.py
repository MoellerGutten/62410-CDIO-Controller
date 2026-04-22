import threading

class FieldState:
    def __init__(self):
        self.lock = threading.Lock()
        self.robot = None
        self.balls = []
        self.cross = None
        self.corners = []

    def __repr__(self) -> str:
        ball_strs = ", ".join(repr(b) for b in self.balls)
        corner_strs = ", ".join(repr(c) for c in self.corners)
        return (
            f"FieldState(\n"
            f"  robot:   {self.robot!r}\n"
            f"  cross:   {self.cross!r}\n"
            f"  balls:   [{ball_strs}]\n"
            f"  corners: [{corner_strs}]\n"
            f")"
        )

state = FieldState()