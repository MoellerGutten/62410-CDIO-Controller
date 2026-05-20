from src.lib.config import Config
from socket import socket, AF_INET, SOCK_STREAM
from sys import exit

def connect():
    config = Config()
    sock = socket(AF_INET, SOCK_STREAM)
    try:
        host = config.getStr("EV3_HOST")
        port = config.getNum("EV3_PORT")
        print(f"Connecting to {host}:{port}...")
        sock.connect((host, port))
        print("Connected! Type 'exit' to exit.")
        return sock
    except ConnectionRefusedError:
        sock.close()
        print("Error: Could not connect. Is the robot running?")
        exit(1)
    except KeyboardInterrupt:
        sock.close()
        print("\nClosing connection.")
        exit(1)