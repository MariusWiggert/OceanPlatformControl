"""random functions which are needed as part of the generative error model
code but do not belong to a specific class."""

from contextlib import redirect_stdout
import yaml
import sys
import numpy as np
import os
import logging
from typing import List, Dict


def get_path_to_project(static_path: str) -> str:
    file_split = static_path.split("/")
    end_idx = file_split.index("OceanPlatformControl")
    relative_path = "/".join(file_split[:end_idx+1])
    return relative_path


def load_config(yaml_file_config: str) -> Dict:
    config_path = os.path.join(get_path_to_project(os.getcwd()), "scenarios/generative_error_model", yaml_file_config)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"Loaded config at: {config_path}")
    return config


#------------------------------ Decorator Functions ---------------------------------#

def timer(func):
    import time

    def wrapper():
        start = time.time()
        func()
        time_taken = time.time() - start
        hours = time_taken//3600
        minutes = (time_taken - (time_taken//3600)*3600)//60
        seconds = round(time_taken - hours*3600 - minutes*60, 2)
        print(f"\nTime taken: {int(hours)} hours {int(minutes)} minutes {seconds} seconds.")
    return wrapper


def log_std_out(func):
    filename = os.path.join("/home/jonas/Downloads/", f"{func.__name__}_log.txt")

    def wrapper():
        with open(filename, 'w') as f:
            with redirect_stdout(f):
                func() 
        # return wrapper
    print("\nLog saving in progress.")       
    return wrapper


#-------------------------- LOGGING ---------------------------#

def setup_logger(log_root, now_string):
    logging.basicConfig(
        format="[%(levelname)s | %(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_root, f'{now_string}.log'))
        ])
    logger = logging.getLogger()
    return logger


def refresh_logger(args, logger):
    # Logging fix for stale file handler
    logger.removeHandler(logger.handlers[1])
    fh = logging.FileHandler(os.path.join(args.log_root, f'{args.now_string}.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(args.formatter)
    logger.addHandler(fh)
    return logger


#------------------------ CONVERSIONS -------------------------#

def convert_km_to_degree(dx: np.ndarray, dy: np.ndarray) -> List[np.ndarray]:
    """Takes difference in lon km and lat km and convert them into differences in degrees."""
    dlat = dy * (360 / 39806.64)
    dlon = (dx * (360/40075.2)) / np.cos(dlat * np.pi/360)
    return dlon, dlat


def convert_degree_to_km(lon: np.ndarray, lat: np.ndarray) -> List[np.ndarray]:
    """Takes two sets of points, each with a lat and lon degree, and computes the distance between each pair in km.
    Note: e.g. pts1 -> np.array([lon, lat])."""
    # https://stackoverflow.com/questions/24617013/convert-latitude-and-longitude-to-x-and-y-grid-system-using-python
    x = lon * 40075.2 * np.cos(lat * np.pi/360)/360
    y = lat * (39806.64/360)
    return x, y
