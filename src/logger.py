import logging
import sys
import os
from datetime import datetime


def configure_logger(name="logger", is_file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if is_file:
            os.makedirs('logs', exist_ok=True)
            log_filename = datetime.strftime(datetime.now(), 'logs/logs_%Y%m%d_%H%M%S.log')
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger