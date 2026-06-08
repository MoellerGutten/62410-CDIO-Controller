from logging import Logger, getLogger, DEBUG, Formatter, FileHandler
from datetime import datetime
from src.model.arena_state import ArenaState
from os import makedirs

def setup_state_logger() -> Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"logs/state_{timestamp}.log"
    logger = getLogger("state")
    logger.setLevel(DEBUG)
    makedirs("logs", exist_ok=True)
    handler = FileHandler(path, encoding="utf-8")
    handler.setFormatter(Formatter("[%(asctime)s] %(message)s\n"))
    logger.addHandler(handler)
    return logger

def log_state(logger: Logger, state: ArenaState) -> None:
    logger.debug(repr(state))