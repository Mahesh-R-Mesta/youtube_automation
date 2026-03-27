import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "youtube_automation",
    log_file: str = "pipeline.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Configure and return a logger with both console and file handlers.
    Idempotent — calling multiple times returns the same configured logger.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)-8s]  %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File — DEBUG and above (full trace for debugging)
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # If the log file can't be created (e.g., read-only filesystem), skip it
        logger.warning("Could not create log file '%s'. File logging disabled.", log_file)

    return logger


# Module-level singleton — import directly:
#   from utils.logger import logger
logger = setup_logger()
