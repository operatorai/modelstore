import logging
import sys


def get_logger():
    log = logging.getLogger(name="modelstore")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    log.setLevel(logging.INFO)
    log.addHandler(handler)
    return log


logger = get_logger()
