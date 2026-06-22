from src.autonomous_mode.movement_helpers import go_to, drive_forward, burst_into_ball, turn_to_heading, turn_to_point, burst_backward
from src.autonomous_mode.state_helpers import update_ball_count_estimate, await_robot
from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.debug.log import get_logger
from src.lib.connection import RobotConnection
from src.lib.constants import ROBOT_TO_POINT_DISTANCE_BEFORE_BURST
from src.autonomous_mode.corners import advance_to_corner_ball, get_staging_data, back_towards_wall_and_turn

def collect_waypoint_zone_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    pass

def collect_cross_zone_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    pass

def collect_edge_ball(state: ArenaState, ball: Ball, connection: RobotConnection, staging_point: tuple[float, float]):
    go_to(state, connection, staging_point)

    turn_to_point(state, connection, ball.position)
    while True:
        robot = await_robot(state, connection)
        if robot.distance_to_point(ball.position) < ROBOT_TO_POINT_DISTANCE_BEFORE_BURST:
            burst_into_ball(state, connection, ball.position)
            update_ball_count_estimate(state)
            burst_backward(state, connection)
            break
        else:
            turn_to_point(state, connection, ball.position, True)
            drive_forward(state, connection, ball.position)

def collect_normal_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    go_to(state, connection, ball.position, approach_radius=10.0)
    drive_forward(state, connection, ball.position)
    burst_into_ball(state, connection, ball.position)
    update_ball_count_estimate(state)

def collect_corner_ball(state: ArenaState, connection: RobotConnection, ball: Ball) -> None:
    logger = get_logger("collect_corner_ball")

    staging_point, staging_heading, along_edge, collection_heading = get_staging_data(ball)

    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point)
    
    logger.debug(f"Turning to staging heading {staging_heading}")
    turn_to_heading(state, connection, staging_heading, precise_mode=True)

    logger.debug("Backing to wall and adjusting heading")
    try:
        # backing towards wall can throw if heading is incorrect and max attempts is reached
        back_towards_wall_and_turn(state, connection, along_edge, collection_heading)
    except:
        logger.error("Failed to collect corner ball, unable to drive backwards to edge. Giving up:(")
        return

    logger.debug("Going to collect ball")
    advance_to_corner_ball(state, connection, ball)
