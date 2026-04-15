import threading

class FieldState:
    def __init__(self):
        self.lock = threading.Lock()
        self.robot = None
        self.balls = []
        self.cross = None
        self.corners = []

state = FieldState()