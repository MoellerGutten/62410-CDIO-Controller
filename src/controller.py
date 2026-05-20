from argparse import ArgumentParser
from threading import Thread
from src.debug.log import setup_state_logger
from src.interactive_mode.start_interactive_session import start_interactive_session
from src.autonomous_mode.start_autonomous_session import start_autonomous_session
from src.state.state_manager import poll_state
from src.debug.gui import run_gui, get_test_field_state
from src.model.state import FieldState
from logging import Logger

def start():
    args = parse_args()
    state = get_test_field_state()
    logger = setup_state_logger() if args.log else None

    controller_thread = Thread(
        target=run_controller,
        kwargs={"state": state, "args": args, "logger": logger},
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
            kwargs={"state": state, "logger": logger},
            name="state-poller",
            daemon=True,
        )
        state_thread.start()

    if args.gui:
        print("Running controller with GUI")
        run_gui(state)
    else:
        print("Running controller")
        controller_thread.join()


def run_controller(state: FieldState, args, logger: Logger):
    if args.it:
        start_interactive_session()
    else:
        start_autonomous_session(state, logger)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Show pygame field renderer")
    parser.add_argument("--it",  action="store_true", help="Run interactive session")
    parser.add_argument("--log", action="store_true", help="Log state changes to file")
    return parser.parse_args()
