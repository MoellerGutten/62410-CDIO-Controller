from src.lib.connection import RobotConnection
from src.interactive_mode.interactice_command_util import parse_input, build_message_from_short_command

def start_interactive_session() -> None:
    connection = RobotConnection()
    while True:
        inp = input("Robot instruction > ").strip()
        if inp.lower() == "exit":
            break
        if not inp:
            continue
        name, kwargs = parse_input(inp)
        msg = build_message_from_short_command(name, kwargs)
        response = connection.send_message(msg)
        print("Robot response:", response)
    print("\nClosing connection.")
