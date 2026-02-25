# controller.py
import socket
import sys

EV3_IP = "10.42.0.182"   # replace with the EV3's WiFi IP
EV3_PORT = 9999

def send_command(cmd: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((EV3_IP, EV3_PORT))
        # Send command with newline for readability
        sock.sendall((cmd + "\n").encode("utf-8"))

        # Read optional reply (up to 1 KB)
        data = sock.recv(1024)
        if data:
            print("Reply:", data.decode("utf-8").strip())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pc_client.py <command string>")
        sys.exit(1)

    command = " ".join(sys.argv[1:])
    send_command(command)
