from logging import Logger

from src.model.arena_state import ArenaState
from src.state.arena_tracker import ArenaTracker
from src.debug.log import log_state


def _get_tracker() -> ArenaTracker:
    """Return the singleton tracker, starting it if needed."""
    tracker = ArenaTracker()
    tracker.start()     # no-op if already running
    return tracker


def update_state(state: ArenaState, logger: Logger = None) -> None:
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

    if logger:
        log_state(logger, state)


def poll_state(state: ArenaState, logger: Logger = None) -> None:
    """
    Continuous background loop: keeps scanning and updating state.
    Intended to run in a dedicated daemon thread.
    Do NOT also call update_state() from other threads while this runs.
    Blocks forever.
    """
    tracker = _get_tracker()
    print("[state_manager] Background polling started.")
    while True:
        new = tracker.scan()
        with state.lock:
            state.robot = new.robot
            state.balls = new.balls
            state.cross = new.cross
        if logger:
            log_state(logger, state)