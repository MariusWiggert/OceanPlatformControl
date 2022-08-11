"""random functions which are needed as part of the generative error model
code but do not belong to a specific class."""

import ocean_navigation_simulator.utils.units
from ocean_navigation_simulator.generative_error_model.variogram.Variogram import Variogram

from typing import List
import yaml
import numpy as np


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


def timer(func):
    import time
    def wrapper():
        start = time.time()
        func()
        time_taken = time.time() - start
        time_taken = 35994.07
        hours = time_taken//3600
        minutes = (time_taken - (time_taken//3600)*3600)//60
        seconds = round(time_taken - (hours)*3600 - (minutes)*60, 2)
        print(f"\nTime taken: {int(hours)} hours {int(minutes)} minutes {seconds} seconds.")
    return wrapper
