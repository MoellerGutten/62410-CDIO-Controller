from src.autonomous_mode.movement_helpers import go_to, burst_into_ball, turn_to_heading, turn_to_point, burst_backward, handle_balls_in_radius
from src.model.arena_state import ArenaState
from src.model.ball import Ball
from src.debug.log import get_logger
from src.lib.connection import RobotConnection
from src.lib.constants import GO_TO_BALL_EDGE_APPROACH_RADIUS, GO_TO_BALL_APPROACH_RADIUS, EDGE_BALL_GENTLE_BURST_TARGET_RANGE
from src.autonomous_mode.corners import advance_to_corner_ball, get_staging_data, back_towards_wall_and_turn, gentle_burst
from src.autonomous_mode.state_helpers import await_robot, update_ball_count_estimate

def collect_edge_ball(state: ArenaState, ball: Ball, connection: RobotConnection, staging_point: tuple[float, float]):
    go_to(state, connection, staging_point)
    turn_to_point(state, connection, ball.position)
    go_to(state, connection, ball.position, GO_TO_BALL_EDGE_APPROACH_RADIUS)
    gentle_burst(state, connection, ball.position, EDGE_BALL_GENTLE_BURST_TARGET_RANGE, 20)
    burst_backward(state, connection)

def collect_normal_ball(state: ArenaState, ball: Ball, connection: RobotConnection):
    handle_balls_in_radius(state, connection, ball)

    go_to(state, connection, ball.position, approach_radius=GO_TO_BALL_APPROACH_RADIUS)
    robot = await_robot(state, connection)
    if robot.distance_to_point(ball.position) > 28.0: # TODO: adjust and make constant
        # return early if ball is in a galaxy far far away.
        update_ball_count_estimate(state)
        return
    turn_to_point(state, connection, ball.position, precise_mode=True)
    burst_into_ball(state, connection, ball.position)

def collect_corner_ball(state: ArenaState, connection: RobotConnection, ball: Ball) -> None:
    logger = get_logger("collect_corner_ball")

    staging_point, staging_heading, along_edge, collection_heading, true_corner_target_point = get_staging_data(ball)

    logger.debug(f"Going to staging point at {staging_point}")
    go_to(state, connection, staging_point)

    is_edge_ball, edge_ball_staging_point = ball.is_edge_ball()
    if ball.distance_to_nearest_corner() > 12 and is_edge_ball and await_robot(state, connection).is_facing_edge(ball.nearest_edge()):
        logger.debug("Wall hugging not needed, proceeding with edge ball collection")
        collect_edge_ball(state, ball, connection, edge_ball_staging_point)
        return
    
    logger.debug(f"Turning to staging heading {staging_heading}")
    turn_to_heading(state, connection, staging_heading)

    logger.debug("Backing to wall and adjusting heading")
    try:
        # backing towards wall can throw if heading is incorrect and max attempts is reached
        back_towards_wall_and_turn(state, connection, along_edge, collection_heading)
    except:
        logger.error("Failed to collect corner ball, unable to drive backwards to edge. Giving up:(")
        return

    logger.debug("Going to collect ball")
    advance_to_corner_ball(state, connection, ball, true_corner_target_point)
