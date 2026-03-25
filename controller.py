import socket
from config import Config
from protocol import Instruction, Arguments, Message, serialize_message

def send_instruction_file(sock):
    with open('instructions.txt', 'r') as f:
        for line in f:
            instruction = line.strip()
            
            if not instruction: 
                continue # Skip empty lines

            print(f"Executing from file: {instruction}")
            sock.sendall((instruction + "\n").encode("utf-8"))
            
            data = sock.recv(1024)
            print("Robot:", data.decode("utf-8").strip())
            
def insert_instruction(instruction):
    with open('instructions.txt', 'a') as f:
        f.write(instruction)

def insert_instructionln(instruction=""):
    with open('instructions.txt', 'a') as f:
        f.write(instruction + "\n")


def start_interactive_session(config: Config):
    # The 'with' block starts HERE so the connection stays open
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            print(serialize_message(Message(Instruction(name="fwd", type="c_", args = Arguments(speed=50,seconds=2,brake=True,block=True)))))
            host = config.getStr("EV3_HOST")
            port = config.getNum("EV3_PORT")
            print(f"Connecting to {host}:{port}...")
            sock.connect((host, port))
            print("Connected! Type 'exit' to quit.")
            
            print("sending forward")
            sock.sendall(serialize_message(Instruction(name="c_fwd", args=Arguments())).encode("utf-8"))
            print("sent forward")
            
            data = sock.recv(1024)
            print(f"ack: {data.decode("utf-8").strip()}")

            while True:
                # Get user input from the terminal
                cmd = input("Robot command > ").strip()

                if cmd.lower() == "exit" or cmd.lower() == "quit":
                    break

                if not cmd:
                    continue # Skip empty lines

                if cmd.lower() == "execute order 66":
                    send_instruction_file(sock)
                else:
                    # Send the command
                    sock.sendall((cmd + "\n").encode("utf-8"))

                    data = sock.recv(1024)
                    print("Robot:", data.decode("utf-8").strip())

        except ConnectionRefusedError:
            print("Error: Could not connect. Is the robot running?")
        except KeyboardInterrupt:
            print("\nClosing connection.")

if __name__ == "__main__":
    config = Config()
    start_interactive_session(config)