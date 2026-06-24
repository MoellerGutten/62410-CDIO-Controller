from argparse import ArgumentParser, Namespace
from threading import Thread
from src.interactive_mode.start_interactive_session import start_interactive_session
from src.autonomous_mode.start_autonomous_session import start_autonomous_session
from src.state.state_manager import poll_state
from src.debug.gui import run_gui, get_test_field_state
from src.model.arena_state import ArenaState
from src.debug.log import get_logger, setup_logger

def start() -> None:
    setup_logger()
    args = parse_args()
    state = get_test_field_state()

    controller_thread = Thread(
        target=run_controller,
        kwargs={"state": state, "args": args},
        name="controller",
        daemon=True,
    )
    controller_thread.start()

    # Only start the background polling thread when NOT in autonomous mode.
    # Autonomous mode calls update_state() inline inside its own loop, so
    # running a parallel poll_state() thread would cause two simultaneous
    # camera reads from the same handle.
    if not args.it:
        # Autonomous mode manages its own scans — no background thread needed.
        pass
    else:
        state_thread = Thread(
            target=poll_state,
            kwargs={"state": state},
            name="state-poller",
            daemon=True,
        )
        state_thread.start()

    if args.gui:
        get_logger().info("Running controller with GUI")
        run_gui(state)
    else:
        get_logger().info("Running controller without GUI")
        controller_thread.join()


def run_controller(state: ArenaState, args: Namespace) -> None:
    if args.it:
        start_interactive_session()
    else:
        start_autonomous_session(state, args.test)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Show pygame field renderer")
    parser.add_argument("--it",  action="store_true", help="Run interactive session")
    parser.add_argument("--test", action="store_true", help="Run in test mode, meaning ball count is grabbed from image")
    return parser.parse_args()
