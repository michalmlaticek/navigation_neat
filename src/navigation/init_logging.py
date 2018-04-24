import logging
import logging.handlers
from pathlib import Path


def init(log_path, log_name):
    logging.basicConfig(
        format="%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("{}/{}".format(log_path, log_name)),
            logging.StreamHandler()
        ],
        level=logging.INFO)
