from src.autonomous_mode.movement_helpers import _drive_toward_point, _turn_toward_point, _start_ejaculation, _stop_ball_intake
from src.autonomous_mode.state_helpers import await_robot
from src.debug.gui import FIELD_H
from src.model.arena_state import ArenaState
from src.lib.connection import RobotConnection
from src.debug.log import get_logger

def deliver_balls(state: ArenaState, connection: RobotConnection) -> None:
    arena_height = 121.5
    arena_width = 167
    
    get_logger().debug("Deliver balls")

    get_logger().debug("Turning off ball motor")
    _stop_ball_intake(connection)
    
    drive_to_center(state, connection, arena_height, arena_width)

    drive_to_goal(state, connection, arena_height, arena_width)
        
    get_logger().debug("Ejaculating")
    _start_ejaculation(connection)

    get_logger().debug("Deliver balls done\n")


def drive_to_center(state: ArenaState, connection: RobotConnection, arena_height: float, arena_width: float):
    center_line_point = [state.robot.position[0], arena_height / 2]
    while True:
        robot = await_robot(state, connection)

        if (robot.distance_to_point(center_line_point) < 10): break

        get_logger().debug("Drive to center")

        _turn_toward_point(state, connection, center_line_point)
        _drive_toward_point(state, connection, center_line_point)

        get_logger().debug("End of loop\n")

        

def drive_to_goal(state: ArenaState, connection: RobotConnection, arena_height: float, arena_width: float):
    goal = [arena_width, arena_height/2]
    while True:
        robot = await_robot(state, connection)

        if (robot.distance_to_point(goal) < 12): break

        get_logger().debug("Drive to goal")

        _turn_toward_point(state, connection, goal)
        _drive_toward_point(state, connection, goal)

        get_logger().debug("End of loop\n")