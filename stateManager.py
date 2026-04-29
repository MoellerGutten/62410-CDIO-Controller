
from image_recon.scripts import YOLO_controller
from model.ball import Ball
from model.cross import Cross
from model.state import FieldState
from debug.log import log_state



def update_state(state: FieldState, frame, model, M, M_inv, logger=None):
    # Gets data from camera
    robot_data, vis_frame = YOLO_controller.scan(frame, model, M, M_inv)
    setState(state, robot_data, logger)
        

def setState(state: FieldState, newState, logger=None):
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
    if logger:
        print("wog")
        log_state(logger, state)
