from numpy import median

from src.state.state_manager import update_state
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.debug.log import get_logger
<<<<<<< HEAD
from src.autonomous_mode.movement_helpers import nudge_robot
=======
from src.autonomous_mode.movement_helpers import _collect_ball, _drive_toward_point, _nudge_robot, adjust_heading
>>>>>>> main
from time import time

# ── State Helpers ───────────────────────────────────────────────────────────────────

MAX_ROBOT_DETECTION_ATTEMPTS = 5

def await_robot(state: ArenaState, connection: RobotConnection):
    """Block until the robot is detected, nudging periodically if needed."""
    logger = get_logger("await_robot")
    attempt = 0
    while True:
        update_state(state)
        if state.robot is not None:
            return state.robot

        attempt += 1
        logger.warning(f"Robot not detected (attempt {attempt})")

        if attempt % MAX_ROBOT_DETECTION_ATTEMPTS == 0:
            logger.warning("Robot still not detected — nudging robot")
            nudge_robot(connection)


def update_ball_count_estimate(state: ArenaState) -> int:
    begin = time()
    ball_counts = []
    BALL_COUNT_ESTIMATION_SNAPSHOTS = 50
    for _ in range(BALL_COUNT_ESTIMATION_SNAPSHOTS):
        update_state(state)
        ball_counts.append(len(state.balls))
    estimated_ball_count = round(sum(ball_counts) / len(ball_counts))
    with state.lock:
        state.estimated_ball_count = estimated_ball_count
    get_logger().debug(f"update_ball_count_estimate took {time() - begin}s with {BALL_COUNT_ESTIMATION_SNAPSHOTS} snapshots")
    return estimated_ball_count


def has_vip_balls(state: ArenaState) -> bool:
    balls_contain_vip = False
    for ball in state.balls:
        if ball.is_vip:
            balls_contain_vip = True
            break
    return balls_contain_vip
