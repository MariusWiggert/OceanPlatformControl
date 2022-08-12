"""random functions which are needed as part of the generative error model
code but do not belong to a specific class."""

import ocean_navigation_simulator.utils.units
from ocean_navigation_simulator.generative_error_model.variogram.Variogram import Variogram

from typing import List
from contextlib import redirect_stdout
import yaml
import sys
import logging
import numpy as np
import os


def load_config():
    yaml_file_config = "/home/jonas/Documents/Thesis/OceanPlatformControl/scenarios/generative_error_model/config_buoy_data.yaml"
    with open(yaml_file_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_variogram_to_npy(variogram: Variogram, file_path: str):
    if variogram.bins is None:
        raise Exception("Need to build variogram first before you can save it!")

    data_to_save = {"bins":variogram.bins,
                    "bins_count":variogram.bins_count,
                    "res": [variogram.lon_res, variogram.lat_res, variogram.t_res],
                    "units": variogram.units
                    }
    np.save(file_path, data_to_save)
    print(f"\nSaved variogram data to: {file_path}")


def load_variogram_from_npy(file_path: str):
    data = np.load(file_path, allow_pickle=True)
    bins = data.item().get("bins")
    bins_count = data.item().get("bins_count")
    res = data.item().get("res")
    units = data.item().get("units")
    return bins, bins_count, res, units


#------------------- Decorator Functions ---------------------------------#

def timer(func):
    import time
    def wrapper():
        start = time.time()
        func()
        time_taken = time.time() - start
        hours = time_taken//3600
        minutes = (time_taken - (time_taken//3600)*3600)//60
        seconds = round(time_taken - (hours)*3600 - (minutes)*60, 2)
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

def setup_logger(args, log_root, now_string):
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_root, f'{now_string}.log'))
    ])
    logger = logging.getLogger()

    # args.log_root = log_root
    # args.formatter = formatter
    # args.now_string = now_string
    # return args, logger
    return logger

def refresh_logger(args, logger):
    # Logging fix for stale file handler
    logger.removeHandler(logger.handlers[1])
    fh = logging.FileHandler(os.path.join(args.log_root, f'{args.now_string}.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(args.formatter)
    logger.addHandler(fh)
    return logger
