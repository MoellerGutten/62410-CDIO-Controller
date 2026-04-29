from config import Config
import socket
import sys

def connect():
    config = Config()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
        sys.exit(1)
    except KeyboardInterrupt:
        sock.close()
        print("\nClosing connection.")
        sys.exit(1)