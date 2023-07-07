import logging
import logging.handlers
import sys


def set_logger(module_name: str):
    logger = logging.getLogger(module_name)
    logger.handlers = []
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(message)s")
    )
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return logger
