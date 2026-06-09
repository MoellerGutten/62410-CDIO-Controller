from numpy import median

from src.autonomous_mode.deliver_balls import deliver_balls
from src.state.state_manager import update_state
from protocol import CommandName, Arguments, Instruction, InstructionType, Message
from src.lib.connection import RobotConnection
from src.model.arena_state import ArenaState
from logging import Logger
from time import sleep
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


def _test_for_ball_presence(state: ArenaState, logger: Logger = None, amount: int = 10) -> int:
    arena_states = []
    for __ in range(amount):
        update_state(state, logger) 
        arena_states.append(len(state.balls))
    
    return median(arena_states)

