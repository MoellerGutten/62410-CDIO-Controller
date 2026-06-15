from src.lib.connection import RobotConnection
from src.interactive_mode.interactice_command_util import parse_input, build_message_from_short_command
from src.debug.log import get_logger

def start_interactive_session() -> None:
    logger = get_logger("start_interactive_session")
    connection = RobotConnection()
    while True:
        inp = input("Robot instruction > ").strip()
        if inp.lower() == "exit":
            break
        if not inp:
            continue
        try:
            name, kwargs = parse_input(inp)
            msg = build_message_from_short_command(name, kwargs)
            response = connection.send_message(msg)
            logger.info(f"Robot response: {response}")
        except:
            logger.error("Invalid command")
    logger.info("\nClosing connection.")
