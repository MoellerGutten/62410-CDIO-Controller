import socket
from config import Config
from protocol import serialize_message
from input import build_message_from_short_command, parse_input

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
                inp = input("Robot command > ").strip()

                if inp.lower() == "exit":
                    break

                if not inp:
                    continue # Skip empty lines
                else:
                    # Send the command
                    name, kwargs = parse_input(inp)
                    sock.sendall((serialize_message(build_message_from_short_command(name, kwargs)) + "\n").encode("utf-8"))

                    data = sock.recv(1024)
                    print("Robot response:", data.decode("utf-8").strip())

        except ConnectionRefusedError:
            print("Error: Could not connect. Is the robot running?")
        except KeyboardInterrupt:
            print("\nClosing connection.")

if __name__ == "__main__":
    start_interactive_session()