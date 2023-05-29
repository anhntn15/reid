import sys
import logging

from config import LOGGING_FILE


def get_custom_logger(logger_name, log_level, log_suffix=None):
    logger = logging.getLogger(logger_name)
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(f'{LOGGING_FILE}_{log_suffix}' if log_suffix else LOGGING_FILE)

    formatter = logging.Formatter(
        "[{levelname}] {asctime} {name}:{lineno} {message}", style="{"
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(log_level)
    logger.propagate = False

    return logger
