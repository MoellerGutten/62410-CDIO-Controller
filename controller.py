from connection import connect
from protocol import serialize_message
from input import build_message_from_short_command, parse_input

from stateManager import poll_state, update_state   # poll_state for thread, update_state for inline

import argparse
import threading
import sys
import os
from debug.log import setup_state_logger
from autonomous.start import start_autonomous_session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "debug"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from gui import run_gui, get_test_field_state
from state import FieldState


def start(args):
    state = get_test_field_state()
    logger = setup_state_logger() if args.log else None

    controller_thread = threading.Thread(
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
        state_thread = threading.Thread(
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


def run_controller(state: FieldState, args, logger):
    if args.it:
        start_interactive_session()
    else:
        start_autonomous_session(state, logger)


def start_interactive_session():
    sock = connect()
    while True:
        inp = input("Robot instruction > ").strip()
        if inp.lower() == "exit":
            break
        if not inp:
            continue
        name, kwargs = parse_input(inp)
        msg = build_message_from_short_command(name, kwargs)
        serialized = serialize_message(msg) + "\n"
        sock.sendall(serialized.encode("utf-8"))
        data = sock.recv(1024)
        print("Robot response:", data.decode("utf-8").strip())
    print("\nClosing connection.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Show pygame field renderer")
    parser.add_argument("--it",  action="store_true", help="Run interactive session")
    parser.add_argument("--log", action="store_true", help="Log state changes to file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start(args)