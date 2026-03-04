import socket
import sys

EV3_IP = "10.33.112.57" 
EV3_PORT = 9999

def start_interactive_session():
    # The 'with' block starts HERE so the connection stays open
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            print(f"Connecting to {EV3_IP}...")
            sock.connect((EV3_IP, EV3_PORT))
            print("Connected! Type 'exit' to quit.")

            while True:
                # Get user input from the terminal
                cmd = input("Robot command > ").strip()

                if cmd.lower() == 'exit':
                    break
                
                if not cmd:
                    continue

                # Send the command
                sock.sendall((cmd + "\n").encode("utf-8"))

                # Wait for the Robot to finish and reply
                # This is important so you don't send 2 commands at once
                data = sock.recv(1024)
                print("Robot:", data.decode("utf-8").strip())

        except ConnectionRefusedError:
            print("Error: Could not connect. Is the robot script running?")
        except KeyboardInterrupt:
            print("\nClosing connection.")

if __name__ == "__main__":
    start_interactive_session()