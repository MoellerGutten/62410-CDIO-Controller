from protocol import CommandName, Arguments, Instruction, InstructionType, Message, serialize_message
from src.autonomous_mode.movement_helpers import _drive_toward_point, _turn_toward_point, _start_ejaculation
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
    
    while True:
        robot = _await_robot(state, connection, logger)
        if robot is None:
            print("Robot not detected after nudge — retrying main loop")
            continue
        
        update_state(state, logger)
        
        # Goal point is middle point of goal a
        goal = [167, 121.5/2]
        center_line_point = [robot.position[0], arena_height / 2]
        print("Target goal coords: " + str(goal))
        print("Intermediate target point (robot.x, FIELD_H/2): " + str(center_line_point))

        print("Turning towards center line point: " + str(center_line_point) + " current point: " + str(robot.position))
        _turn_toward_point(state, connection, logger, center_line_point)
        print("Driving towards center line point: " + str(center_line_point) + " current point: " + str(robot.position))
        _drive_toward_point(state, connection, logger, center_line_point)

        print("Turning towards goal point: " + str(goal) + " current point: " + str(robot.position))
        _turn_toward_point(state, connection, logger, goal)
        print("Driving towards goal point: " + str(goal) + " current point: " + str(robot.position))
        _drive_toward_point(state, connection, logger, goal)

        _start_ejaculation(connection)
        break