from src.debug.log import get_logger
from src.model.arena_state import ArenaState
from src.state.arena_tracker import ArenaTracker

def _get_tracker() -> ArenaTracker:
    """Return the singleton tracker, starting it if needed."""
    tracker = ArenaTracker()
    tracker.start()     # no-op if already running
    return tracker


def update_state(state: ArenaState) -> None:
    """
    One-shot scan: capture one frame and update state.
    Call this inline inside a control loop.
    """
    tracker  = _get_tracker()
    new      = tracker.scan()

    with state.lock:
        state.robot = new.robot
        state.balls = new.balls
        state.cross = new.cross


def update_balls(state: ArenaState) -> None:
    """
    Lightweight scan that updates only ball (and cross) positions, skipping the
    expensive ArUco robot detection. Use when the robot pose is not needed —
    e.g. ball-count estimation while the robot is stationary. Leaves
    ``state.robot`` untouched so it is never clobbered with a stale ``None``.
    """
    tracker = _get_tracker()
    new     = tracker.scan(detect_robot=False)

    with state.lock:
        state.balls = new.balls
        state.cross = new.cross


def poll_state(state: ArenaState) -> None:
    """
    Continuous background loop: keeps scanning and updating state.
    Intended to run in a dedicated daemon thread.
    Do NOT also call update_state() from other threads while this runs.
    Blocks forever.
    """
    tracker = _get_tracker()
    get_logger().info("Background polling started")
    while True:
        new = tracker.scan()
        with state.lock:
            state.robot = new.robot
            state.balls = new.balls
            state.cross = new.cross
