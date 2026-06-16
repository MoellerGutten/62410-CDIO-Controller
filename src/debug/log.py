from logging import Logger, getLogger, DEBUG, Formatter, FileHandler, StreamHandler
from datetime import datetime
from os import makedirs
from sys import stdout
from threading import Lock

_logger: Logger | None = None
_setup_lock = Lock()

FORMAT = "[%(asctime)s] [%(levelname)s]%(source_tag)s %(message)s"


class _Formatter(Formatter):
    def format(self, record):
        if not hasattr(record, "source"):
            record.source = None
        record.source_tag = f" [{record.source}]" if record.source else ""
        return super().format(record)


class ExtendedLogger:
    def __init__(self, source: str | None = None):
        self._logger = _logger
        self._extra = {"source": source} if source else {}

    def debug(self, msg: str)    -> None: self._logger.debug(msg, extra=self._extra)
    def info(self, msg: str)     -> None: self._logger.info(msg, extra=self._extra)
    def warning(self, msg: str)  -> None: self._logger.warning(msg, extra=self._extra)
    def error(self, msg: str)    -> None: self._logger.error(msg, extra=self._extra)
    def critical(self, msg: str) -> None: self._logger.critical(msg, extra=self._extra)


def setup_logger() -> Logger:
    """Call this once at program entry, before spawning threads."""
    global _logger

    with _setup_lock:
        if _logger is not None:
            return _logger  # already initialized, don't add handlers again

        makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"logs/{timestamp}.log"
        logger = getLogger("controller")
        logger.setLevel(DEBUG)
        logger.propagate = False
        formatter = _Formatter(FORMAT)

        logger.handlers.clear()
        file_handler = FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = StreamHandler(stdout)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        _logger = logger
        return logger


def get_logger(source: str | None = None) -> ExtendedLogger:
    if _logger is None:
        raise RuntimeError("get_logger called when _logger is None, call setup_logger first")
    return ExtendedLogger(source)