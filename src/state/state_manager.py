from src.model.ball import Ball
from src.model.cross import Cross
from src.model.robot import Robot
from src.model.state import FieldState
from src.debug.log import log_state
from src.state.arena_tracker import ArenaTracker, ScanResult
from src.debug.gui import FIELD_W, FIELD_H
from logging import Logger


def _get_tracker() -> ArenaTracker:
    """Return the singleton tracker, starting it if needed."""
    tracker = ArenaTracker()
    tracker.start()   # no-op if already running (guarded by _running flag)
    return tracker


def update_state(state: FieldState, logger:Logger=None) -> None:
    """
    One-shot scan: capture one frame and update state.

    Call this directly when you need a fresh snapshot inline
    (e.g. inside an autonomous control loop).
    """
    tracker = _get_tracker()
    _set_state(state, tracker.scan(), logger)


def poll_state(state: FieldState, logger:Logger=None) -> None:
    """
    Continuous background loop: keeps scanning and updating state.

    Intended to run in a dedicated daemon thread from controller.py.
    Do NOT also call update_state() from other threads while this is
    running — both would scan from the same camera handle simultaneously.

    Blocks forever (until the process exits or an exception is raised).
    """
    tracker = _get_tracker()
    print("[state_manager] Background polling started.")
    while True:
        _set_state(state, tracker.scan(), logger)


def _set_state(state: FieldState, new_state: ScanResult, logger:Logger=None) -> None:
    """Apply a ScanResult to the shared FieldState (thread-safe)."""
    with state.lock:
        # --- Balls ---
        temp_balls = []
        for ball in new_state.balls:
            is_vip = ball.label.lower() == "oball"
            temp_balls.append(Ball(
                (ball.position.x * FIELD_W / 167,
                 ball.position.y * FIELD_H / 121.5),
                is_vip=is_vip,
            ))
        state.balls = temp_balls

        # --- Cross ---
        if new_state.cross is not None:
            cross = new_state.cross
            cx = sum(p.x * FIELD_W / 167   for p in cross.corners) / 4
            cy = sum(p.y * FIELD_H / 121.5 for p in cross.corners) / 4
            state.cross = Cross((cx, cy), 0)   # TODO: fix orientation
        else:
            state.cross = None

        # --- Robot ---
        if new_state.robot is not None:
            state.robot = Robot(
                (new_state.robot.position.x * FIELD_W / 167,
                 new_state.robot.position.y * FIELD_H / 121.5),
                new_state.robot.heading,
            )

    if logger:
        log_state(logger, state)