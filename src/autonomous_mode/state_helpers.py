from numpy import median

from src.state.state_manager import update_state
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from logging import Logger
from src.autonomous_mode.movement_helpers import _nudge_robot

# ── State Helpers ───────────────────────────────────────────────────────────────────

MAX_ROBOT_DETECTION_ATTEMPTS = 5

def _await_robot(state: ArenaState, connection: RobotConnection, logger: Logger):
    """Poll for robot detection. If not found after MAX attempts, nudge the robot and return None."""
    for attempt in range(MAX_ROBOT_DETECTION_ATTEMPTS):
        update_state(state, logger)
        if state.robot is not None:
            return state.robot
        print(f"Robot not detected (attempt {attempt + 1}/{MAX_ROBOT_DETECTION_ATTEMPTS})")

    print("Robot still not detected — nudging robot")
    _nudge_robot(connection)
    return None

def update_ball_count_estimate(state: ArenaState, logger: Logger = None) -> int:
    ball_counts = []
    BALL_COUNT_ESTIMATION_SNAPSHOTS = 10
    for _ in range(BALL_COUNT_ESTIMATION_SNAPSHOTS):
        update_state(state, logger)
        ball_counts.append(len(state.balls))
    estimated_ball_count = round(sum(ball_counts) / len(ball_counts))
    with state.lock:
        state.estimated_ball_count = estimated_ball_count
    return estimated_ball_count
