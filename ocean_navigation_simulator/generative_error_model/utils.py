"""random functions which are needed as part of the generative error model
code but do not belong to a specific class."""

from contextlib import redirect_stdout
import yaml
import sys
import numpy as np
import os
import logging
from typing import Dict, Tuple
import datetime


def get_path_to_project(static_path: str) -> str:
    file_split = static_path.split("/")
    end_idx = file_split.index("OceanPlatformControl")
    relative_path = "/".join(file_split[:end_idx+1])
    return relative_path


def load_config(yaml_file_config: str) -> Dict:
    config_path = os.path.join(get_path_to_project(os.getcwd()), "scenarios/generative_error_model", yaml_file_config)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"\nLoaded config at: {config_path}.\n")
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

def convert_km_to_degree(dx: np.ndarray, dy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Takes difference in lon km and lat km and convert them into differences in degrees."""
    dlat = dy * (360 / 39806.64)
    dlon = (dx * (360/40075.2)) / np.cos(dlat * np.pi/360)
    return dlon, dlat


def convert_degree_to_km(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Takes two sets of points, each with a lat and lon degree, and computes the distance between each pair in km.
    Note: e.g. pts1 -> np.array([lon, lat])."""
    # https://stackoverflow.com/questions/24617013/convert-latitude-and-longitude-to-x-and-y-grid-system-using-python
    x = lon * 40075.2 * np.cos(lat * np.pi/360)/360
    y = lat * (39806.64/360)
    return x, y


def hour_range_file_name(file_name: str):
    idx_time1 = file_name.index("2022")
    idx_time2 = file_name[idx_time1 + 19:].index("2022") + idx_time1 + 19
    date1 = datetime.datetime.strptime(file_name[idx_time1: idx_time1 + 19], "%Y-%m-%dT%H:%M:%S")
    date2 = datetime.datetime.strptime(file_name[idx_time2: idx_time2 + 19], "%Y-%m-%dT%H:%M:%S")
    date_diff = date2 - date1
    return date_diff.days*24 + date_diff.seconds // 3600, date1


def time_range_hrs(start_date: datetime.datetime, num_hrs: int):
    out_list = []
    for i in range(num_hrs):
        out_list.append(start_date + datetime.timedelta(hours=i))
    return out_list


def same_file_times(file_name1: str, file_name2: str):
    start_date1 = file_name1.index("2022")
    start_date2 = file_name2.index("2022")
    time1 = file_name1[start_date1: start_date1 + 19]
    time2 = file_name2[start_date2: start_date2 + 19]
    if time1 == time2:
        return True
    else:
        return False, time1, time2


def get_datetime_from_file_name(file_name: str):
    idx_time = file_name.index("2022")
    try:
        date = datetime.datetime.strptime(file_name[idx_time: idx_time + 19], "%Y-%m-%dT%H:%M:%S")
    except:
        date = datetime.datetime.strptime(file_name[idx_time: idx_time + 19], "%Y-%m-%d %H:%M:%S")
    return date


def get_time_matched_file_lists(list1, list2):
    file_name_per_day = dict()
    # add forecast file names to dict
    for item1 in list1:
        item1_date = get_datetime_from_file_name(item1)
        if item1_date in list(file_name_per_day.keys()):
            file_name_per_day[item1_date] = [*file_name_per_day[item1_date], item1]
        else:
            file_name_per_day[item1_date] = [item1]
    # add buoy file names to dict if forecast equivalent exists
    for item2 in list2:
        item2_date = get_datetime_from_file_name(item2)
        if item2_date in list(file_name_per_day.keys()):
            file_name_per_day[item2_date] = [*file_name_per_day[item2_date], item2]
    out_list1 = []
    out_list2 = []
    dates = sorted(list(file_name_per_day.keys()))
    for date in dates:
        num_files = len(file_name_per_day[date])
        if num_files == 2:
            out_list1.append(file_name_per_day[date][0])
            out_list2.append(file_name_per_day[date][1])
        if num_files > 2 and num_files % 2 == 0:
            out_list1.extend(file_name_per_day[date][:int(num_files/2)])
            out_list2.extend(file_name_per_day[date][int(num_files/2):])
    return out_list1, out_list2


def datetime2str(datetime_obj: datetime.datetime):
    return datetime.datetime.strftime(datetime_obj, "%Y-%m-%dT%H:%M:%S")
