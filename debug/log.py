import logging
from datetime import datetime
from model.state import FieldState
import os

def setup_state_logger() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"logs/state_{timestamp}.log"
    logger = logging.getLogger("state")
    logger.setLevel(logging.DEBUG)
    os.makedirs("logs", exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s\n"))
    logger.addHandler(handler)
    return logger

def log_state(logger: logging.Logger, state: FieldState):
    logger.debug(repr(state))