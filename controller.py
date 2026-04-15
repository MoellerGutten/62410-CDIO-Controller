import socket
from config import Config
from protocol import serialize_message
from input import build_message_from_short_command, parse_input
import argparse
import threading
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "debug"))

from gui import run_gui

def start(args):
    controller_thread = threading.Thread(
        target=run_controller,
        kwargs={"args": args},
        daemon=True
    )
    controller_thread.start()

    if (args.gui):
        print("Running controller with GUI")
        run_gui()
    else:
        print("Running controller")
        controller_thread.join()

def run_controller():
    start_interactive_session()

def start_interactive_session():
    config = Config()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            host = config.getStr("EV3_HOST")
            port = config.getNum("EV3_PORT")
            print(f"Connecting to {host}:{port}...")
            sock.connect((host, port))
            print("Connected! Type 'exit' to exit.")

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

        except ConnectionRefusedError:
            print("Error: Could not connect. Is the robot running?")
        except KeyboardInterrupt:
            print("\nClosing connection.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Show pygame field renderer")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    start(args)