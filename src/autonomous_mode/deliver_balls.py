from src.autonomous_mode.movement_helpers import _drive_toward_point, _turn_toward_point, _start_ejaculation, _stop_ball_intake
from src.autonomous_mode.state_helpers import _await_robot
from src.debug.gui import FIELD_H
from src.model.arena_state import ArenaState
from src.lib.connection import RobotConnection
from src.debug.log import get_logger

def deliver_balls(state: ArenaState, connection: RobotConnection) -> None:
    arena_height = 121.5
    arena_width = 167
    
    # Sequence should be (excluding cross problem):
    # Aim for y = arena_h/2, i.e. straight up or down from robot (turn first ofc)
    # Drive to goal
    # Eject balls
    
    # Turning off ball collection
    get_logger().debug("Turning off ball motor")
    _stop_ball_intake(connection)
    
    drive_to_center(state, connection, arena_height, arena_width)

    drive_to_goal(state, connection, arena_height, arena_width)
        
    get_logger().debug("Ejaculating")
    _start_ejaculation(connection)


def drive_to_center(state: ArenaState, connection: RobotConnection, arena_height: float, arena_width: float):
    center_line_point = [state.robot.position[0], arena_height / 2]
    while True:
        robot = _await_robot(state, connection)
        if robot is None:
            get_logger().warning("Robot not detected after nudge — retrying main loop")
            continue

        if (robot.distance_to_point(center_line_point) < 7.5):
            break

        # Goal point is middle point of goal a
        get_logger().debug("Intermediate target point (robot.x, FIELD_H/2): " + str(center_line_point))

        get_logger().debug("Turning towards center line point: " + str(center_line_point) + " current point: " + str(robot.position))
        _turn_toward_point(state, connection, center_line_point)

        get_logger().debug("Driving towards center line point: " + str(center_line_point) + " current point: " + str(robot.position))
        _drive_toward_point(state, connection, center_line_point)
        

def drive_to_goal(state: ArenaState, connection: RobotConnection, arena_height: float, arena_width: float):
    goal = [arena_width, arena_height/2]
    while True:
        robot = _await_robot(state, connection)
        if robot is None:
            get_logger().warning("Robot not detected after nudge — retrying main loop")
            continue

        if (robot.distance_to_point(goal) < 10):
            break

        # Goal point is middle point of goal a
        get_logger().debug("Target goal coords: " + str(goal))

        get_logger().debug("Turning towards goal point: " + str(goal) + " current point: " + str(robot.position))
        _turn_toward_point(state, connection, goal)
        get_logger().debug("Driving towards goal point: " + str(goal) + " current point: " + str(robot.position))
        _drive_toward_point(state, connection, goal)