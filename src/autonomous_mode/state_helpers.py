from numpy import median

from src.state.state_manager import update_state
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from src.debug.log import get_logger
from src.autonomous_mode.movement_helpers import _nudge_robot

# ── State Helpers ───────────────────────────────────────────────────────────────────

MAX_ROBOT_DETECTION_ATTEMPTS = 5

def _await_robot(state: ArenaState, connection: RobotConnection):
    """Poll for robot detection. If not found after MAX attempts, nudge the robot and return None."""
    for attempt in range(MAX_ROBOT_DETECTION_ATTEMPTS):
        update_state(state)
        if state.robot is not None:
            return state.robot
        get_logger().warning(f"Robot not detected (attempt {attempt + 1}/{MAX_ROBOT_DETECTION_ATTEMPTS})")

    get_logger().warning("Robot still not detected — nudging robot")
    _nudge_robot(connection)
    return None

def update_ball_count_estimate(state: ArenaState) -> int:
    ball_counts = []
    BALL_COUNT_ESTIMATION_SNAPSHOTS = 10
    for _ in range(BALL_COUNT_ESTIMATION_SNAPSHOTS):
        update_state(state)
        ball_counts.append(len(state.balls))
    estimated_ball_count = round(sum(ball_counts) / len(ball_counts))
    with state.lock:
        state.estimated_ball_count = estimated_ball_count
    return estimated_ball_count

def has_vip_balls(state: ArenaState, logger: Logger = None) -> bool:
    balls_contain_vip = False
    for ball in state.balls:
        if ball.is_vip:
            balls_contain_vip = True
            break
    return balls_contain_vip
