import json
import socket
from config import Config
from protocol import serialize_message
from input import build_message_from_short_command, parse_input

# For detecting changes in the json file (new data from camera)
from JSONObserver import update_state

import argparse
import threading
import sys
import os

from model.ball import Ball
from model.cross import Cross

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "debug"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from gui import run_gui, get_test_field_state
from state import FieldState

def start(args):
    state = get_test_field_state()
    controller_thread = threading.Thread(
        target=run_controller,
        kwargs={"state": state, "args": args},
        daemon=True
    )
    controller_thread.start()
    try:
        with open("image_recon/robot_coords.json", mode="r", encoding="utf-8") as read_file:
            state_data = json.load(read_file)

        # Only starts state thread if proper JSON data exists for initialization.
        state_thread = threading.Thread(
            target=update_state,
            kwargs={"state": state, "newState": state_data},
            daemon=True
        )
        state_thread.start()
    except:
        print("ERROR: Couldn't initialize state, due to missing json. Using test state instead")

    if (args.gui):
        print("Running controller with GUI")
        run_gui(state)
        
    else:
        print("Running controller")
        controller_thread.join()


def run_controller(state: FieldState, args):
    if (args.it):
        start_interactive_session()
    else:
        start_autonomous_session()

def connect():
    config = Config()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            host = config.getStr("EV3_HOST")
            port = config.getNum("EV3_PORT")
            print(f"Connecting to {host}:{port}...")
            sock.connect((host, port))
            print("Connected! Type 'exit' to exit.")

            return sock

        except ConnectionRefusedError:
            print("Error: Could not connect. Is the robot running?")
            exit()
        except KeyboardInterrupt:
            print("\nClosing connection.")
            exit()

def start_autonomous_session():
    print("autonomous")
    pass

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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    start(args)