from src.lib.config import Config
from socket import socket, AF_INET, SOCK_STREAM
from sys import exit
from protocol import Message, serialize_message

class RobotConnection:

    def __init__(self):
        """Initialize a robot connection instance based on the controller config file."""
        config = Config()
        self.socket = socket(AF_INET, SOCK_STREAM)

        try:
            host = config.getStr("EV3_HOST")
            port = config.getNum("EV3_PORT")
            print(f"Connecting to {host}:{port}...")
            self.socket.connect((host, port))
            print("Connected! Type 'exit' to exit.")
        except ConnectionRefusedError:
            self.socket.close()
            print("Error: Could not connect. Is the robot running?")
            exit(1)
        except KeyboardInterrupt:
            self.socket.close()
            print("\nClosing connection.")
            exit(1)

    def send_message(self, message: Message):
        """Send a protocol message object via the robot connection instance and receive a response. The response is decoded, stripped and returned"""
        serialized_message = serialize_message(message)
        encoded_message = serialized_message.encode("utf-8")
        self.socket.sendall(encoded_message)
        return self.socket.recv(1024).decode("utf-8").strip()
