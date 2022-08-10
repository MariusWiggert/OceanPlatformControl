"""random functions which are needed as part of the generative error model
code but do not belong to a specific class."""

import ocean_navigation_simulator.utils.units

from typing import List
import yaml
import numpy as np


def load_config():
    yaml_file_config = "/home/jonas/Documents/Thesis/OceanPlatformControl/scenarios/generative_error_model/config_buoy_data.yaml"
    with open(yaml_file_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_variogram_from_npy(path: str):
    data = np.load(path, allow_pickle=True)
    bins = data.item().get("bins")
    bins_count = data.item().get("bins_count")
    res = data.item().get("res")
    return bins, bins_count, res


def timer(func):
    import time
    def wrapper():
        start = time.time()
        func()
        print(f"Time taken: {time.time()-start} seconds.")
    return wrapper


# TODO: implement this, needed for variogram and sampling from simplex noise
def convert_degree_to_km(degrees: List[float]) -> List[float]:
    pass