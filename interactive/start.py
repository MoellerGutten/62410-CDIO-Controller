from connection import connect
from input import parse_input, build_message_from_short_command
from protocol import serialize_message

def start_interactive_session():
    sock = connect()
    while True:
        inp = input("Robot instruction > ").strip()
        if inp.lower() == "exit":
            break
        if not inp:
            continue
        name, kwargs = parse_input(inp)
        msg = build_message_from_short_command(name, kwargs)
        serialized = serialize_message(msg) + "\n"
        sock.sendall(serialized.encode("utf-8"))
        data = sock.recv(1024)
        print("Robot response:", data.decode("utf-8").strip())
    print("\nClosing connection.")