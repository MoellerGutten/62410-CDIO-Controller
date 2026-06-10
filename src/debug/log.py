from logging import Logger, getLogger, DEBUG, Formatter, FileHandler, StreamHandler
from datetime import datetime
from os import makedirs
from sys import stdout

_logger: Logger | None = None

def _setup_logger() -> Logger:
    global _logger

    makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"logs/{timestamp}.log"
    logger = getLogger("controller")
    logger.setLevel(DEBUG)
    formatter = Formatter("[%(asctime)s] %(message)s\n")

    file_handler = FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = StreamHandler(stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    _logger = logger
    return logger

def get_logger() -> Logger:
    if _logger is None:
        return _setup_logger()
    else:
        return _logger