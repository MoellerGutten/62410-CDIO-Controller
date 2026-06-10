from protocol import CommandName, Arguments, Instruction, InstructionType, Message, serialize_message
from src.autonomous_mode.movement_helpers import _drive_toward_point, _turn_toward_point, _start_ejaculation, _stop_ball_intake
from src.autonomous_mode.state_helpers import _await_robot
from src.debug.gui import FIELD_H
from src.model.arena_state import ArenaState
from src.state.state_manager import _get_tracker, update_state
from src.lib.connection import RobotConnection
from logging import Logger
import src.state.arena_config as conf

def deliver_balls(state: ArenaState, connection: RobotConnection, logger: Logger) -> None:
    tracker = _get_tracker()
    result = tracker.scan()
    arena_height = 121.5
    arena_width = 167
    
    # Sequence should be (excluding cross problem):
    # Aim for y = arena_h/2, i.e. straight up or down from robot (turn first ofc)
    # Drive to goal
    # Eject balls
    
    # Turning off ball collection
    print("Turning off ball motor")
    _stop_ball_intake(connection)
    
    drive_to_center(state, connection, logger, arena_height, arena_width)

    drive_to_goal(state, connection, logger, arena_height, arena_width)
        
    print("EJACULATING")
    _start_ejaculation(connection)


def drive_to_center(state, connection, logger, arena_height, arena_width):
    center_line_point = [state.robot.position[0], arena_height / 2]
    while True:
        robot = _await_robot(state, connection, logger)
        if robot is None:
            print("Robot not detected after nudge — retrying main loop")
            continue

        if (robot.distance_to_point(center_line_point) < 15):
            break

        # Goal point is middle point of goal a
        print("Intermediate target point (robot.x, FIELD_H/2): " + str(center_line_point))

        print("Turning towards center line point: " + str(center_line_point) + " current point: " + str(robot.position))
        _turn_toward_point(state, connection, logger, center_line_point)

        print("Driving towards center line point: " + str(center_line_point) + " current point: " + str(robot.position))
        _drive_toward_point(state, connection, logger, center_line_point)
        

def drive_to_goal(state, connection, logger, arena_height, arena_width):
    goal = [arena_width, arena_height/2]
    while True:
        robot = _await_robot(state, connection, logger)
        if robot is None:
            print("Robot not detected after nudge — retrying main loop")
            continue

        if (robot.distance_to_point(goal) < 10):
            break

        # Goal point is middle point of goal a
        print("Target goal coords: " + str(goal))

        print("Turning towards goal point: " + str(goal) + " current point: " + str(robot.position))
        _turn_toward_point(state, connection, logger, goal)
        print("Driving towards goal point: " + str(goal) + " current point: " + str(robot.position))
        _drive_toward_point(state, connection, logger, goal)