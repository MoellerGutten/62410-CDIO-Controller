from connection import connect
from protocol import serialize_message
from input import build_message_from_short_command, parse_input

# For detecting changes in the json file (new data from camera)
from stateManager import update_state

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
        daemon=True
    )
    controller_thread.start()

    state_thread = threading.Thread(
        target=update_state,
        kwargs={"state": state, "logger": logger},
        daemon=True
    )
    state_thread.start()

    if (args.gui):
        print("Running controller with GUI")
        run_gui(state)
        
    else:
        print("Running controller")
        controller_thread.join()


def run_controller(state: FieldState, args, logger):
    if (args.it):
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
            continue # Skip empty lines
        else:
            name, kwargs = parse_input(inp)
            msg = build_message_from_short_command(name, kwargs)
            serialized = serialize_message(msg) + "\n"
            encoded = serialized.encode("utf-8")
            sock.sendall(encoded)
            data = sock.recv(1024)
            print("Robot response:", data.decode("utf-8").strip())
    print("\nClosing connection.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Show pygame field renderer")
    parser.add_argument("--it", action="store_true", help="Run interactive session")
    parser.add_argument("--log", action="store_true", help="Log state changes to file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    start(args)