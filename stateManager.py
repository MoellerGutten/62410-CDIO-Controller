from model.ball import Ball
from model.cross import Cross
from model.robot import Robot
from model.state import FieldState
from debug.log import log_state
from image_recon.scripts.arena_tracker import ArenaTracker

def update_state(state: FieldState, logger=None):
    tracker = ArenaTracker()
    tracker.start()
    setState(state, tracker.scan(), logger)
        

def setState(state: FieldState, newState, logger=None):
    # Balls and cross positions are scaled by the ratio of pixels to cm.
    with state.lock:
        ### BALLS ###
        # Removes balls from previous state
        tempBalls = []
        for ball in newState.balls:
            print("Ball: " + ball.label+ "Is at pos: " + str(ball.position.x) + "," + str(ball.position.y))
            if ball.label == "OBall":
                tempBalls.append(Ball((ball.position.x*1383/167, ball.position.y*973.5/121.5), is_vip=True))
            else:
                tempBalls.append(Ball((ball.position.x*1383/167, ball.position.y*973.5/121.5), is_vip=False))
        state.balls = tempBalls
        ### CROSS ### 
        if newState.cross is not None:
            cross = newState.cross
            crossX = 0
            crossY = 0
            
            for point in cross.corners:
                crossX += point.x*1383/167
                crossY += point.y*973.5/121.5

            crossX = crossX/4
            crossY = crossY/4

            # TODO Fix cross orientation
            state.cross = Cross((crossX, crossY), 0)

        # TODO: Corners
        # TODO: Robot
        if newState.robot is not None:
            robotX = newState.robot.position.x
            robotY = newState.robot.position.y
            robotHeading = newState.robot.heading
            state.robot = Robot((robotX, robotY), robotHeading)

    if logger:
        print("wog")
        log_state(logger, state)
