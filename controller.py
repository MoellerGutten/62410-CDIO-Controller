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

if __name__ == "__main__":
    start_interactive_session()