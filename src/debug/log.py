from logging import Logger, getLogger, DEBUG, Formatter, FileHandler
from datetime import datetime
from os import makedirs

_logger: Logger | None = None

def _setup_logger() -> None:
    global _logger

    makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"logs/{timestamp}.log"
    logger = getLogger("controller")
    logger.setLevel(DEBUG) # process all log levels including DEBUG
    handler = FileHandler(path, encoding="utf-8")
    handler.setFormatter(Formatter("[%(asctime)s] %(message)s\n"))
    logger.addHandler(handler)
    _logger = logger
    return logger

def get_logger() -> Logger:
    if _logger is None:
        return _setup_logger()
    else:
        return _logger
