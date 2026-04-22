import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from model.ball import Ball
from model.cross import Cross
from model.state import FieldState

JSON_PATH = "image_recon/robot_coords.json"

class JSONHandler(FileSystemEventHandler):
    def __init__(self, state):
        self.state = state

    def on_modified(self, event):
        if event.is_directory or event.src_path != JSON_PATH:
            return

        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return  # file may be mid-write; try again on next event

        setState(self.state, data)


def update_state(state: FieldState, newState):
    # initalize state from JSON
    setState(state, newState)

    # Using a watchdog thread to monitor for robot_coords.json changes
    observer = Observer()
    handler = JSONHandler(state)
    observer.schedule(handler, path="image_recon", recursive=False)
    observer.start()

    # If changes in JSON
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()

def setState(state: FieldState, newState):
    # Balls and cross positions are scaled by the ratio of pixels to cm.
    with state.lock:
        ### BALLS ###
        # Removes balls from previous state
        tempBalls = []
        for ball in newState["balls"]:
            print("Ball: " + ball["label"] + "Is at pos: " + str(ball["x"]) + "," + str(ball["y"]))
            if ball["label"] == "OBall":
                tempBalls.append(Ball((ball["x"]*1383/167, ball["y"]*973.5/121.5), is_vip=True))
            else:
                tempBalls.append(Ball((ball["x"]*1383/167, ball["y"]*973.5/121.5), is_vip=False))
        state.balls = tempBalls
        ### CROSS ### 
        cross = newState["cross"]
        crossX = 0
        crossY = 0
        
        for point in cross["corners"]:
            crossX += point["x"]*1383/167
            crossY += point["y"]*973.5/121.5

        crossX = crossX/4
        crossY = crossY/4

        # TODO Fix cross orientation
        state.cross = Cross((crossX, crossY), 0)

        # TODO: Corners
        # TODO: Robot
