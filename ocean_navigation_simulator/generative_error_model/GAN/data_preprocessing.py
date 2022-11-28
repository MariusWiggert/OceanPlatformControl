from ocean_navigation_simulator.generative_error_model.utils import hour_range_file_name, time_range_hrs, datetime2str

import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple
import datetime


def save_nc_as_npy(file_dir: str, file_list: list, output_dir: str, lon_range: Tuple, lat_range: Tuple):
    """Takes a .nc file and saves it as a .npy file."""

    for file in file_list:
        output_file_path = os.path.join(output_dir, "".join(file.split(".")[:-1])) + ".npy"
        if os.path.exists(output_file_path):
            print(f"File {''.join(output_file_path.split('/')[-1])} already exists!")
            continue
        data = xr.open_dataset(os.path.join(file_dir, file))
        if "longitude" in list(data.dims):
            data = data.sel(longitude=slice(*lon_range), latitude=slice(*lat_range))
            data_u, data_v = data["utotal"].values, data["vtotal"].values
        elif "lon" in list(data.dims):
            data = data.sel(lon=slice(*lon_range), lat=slice(*lat_range))
            data_u, data_v = data["water_u"].values, data["water_v"].values
        data = np.concatenate([data_u, data_v], axis=1)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(output_file_path, data)


def save_sparse_as_npy(file_dir: str, file_list: list, output_dir: str, lon_range: Tuple, lat_range: Tuple, type="buoy"):
    """Takes sparse buoy data, turns it into a sparse matrix and saves it as an .npy file."""
    for file in file_list:
        output_file_path = os.path.join(output_dir, "".join(file.split(".")[:-1])) + ".npy"
        if os.path.exists(output_file_path):
            print(f"File {''.join(output_file_path.split('/')[-1])} already exists!")
            continue
        data = pd.read_csv(os.path.join(file_dir, file))
        data["hour"] = data["time"].apply(lambda x: x[:13])
        hours_according_file_name, start_date = hour_range_file_name(file)
        # advance by 1 hr due to 30 minute shift between between buoy time and FC time
        start_date = start_date + datetime.timedelta(hours=1)
        hour_range = time_range_hrs(start_date, hours_according_file_name)
        output_data = np.zeros((hours_according_file_name,
                                2,
                                len(list(np.arange(*lon_range, 1/12)))+1,
                                len(list(np.arange(*lat_range, 1/12)))+1))
        # need to make sure that the hours of the buoy data match the dense fc/hc data
        # if there is no buoy data for a particular hour, an empty frame is inserted
        added_non_zeros = 0
        for time_step, hour in enumerate(hour_range):
            hour = datetime2str(hour)[:13].replace("T", " ")
            if hour in data["hour"].values:
                added_non_zeros += 1
                data_time_step = data[data["hour"].values == hour]
                # convert from sparse to dense
                points = np.array([data_time_step["lon"], data_time_step["lat"]])
                nearest_grid_points = round_to_multiple(points)
                lon_idx = np.searchsorted(np.arange(*lon_range, 1/12), nearest_grid_points[0])
                lat_idx = np.searchsorted(np.arange(*lat_range, 1/12), nearest_grid_points[1])
                if type == "buoy":
                    # TODO: Figure out if these should be switched (lon, lat)
                    output_data[time_step, 0, lon_idx, lat_idx] = data_time_step["u"].values
                    output_data[time_step, 1, lon_idx, lat_idx] = data_time_step["v"].values
                elif type == "forecast":
                    output_data[time_step, 0, lon_idx, lat_idx] = data_time_step["u_forecast"].values
                    output_data[time_step, 1, lon_idx, lat_idx] = data_time_step["v_forecast"].values
                else:
                    raise ValueError("Specified type does not exist!")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(output_file_path, output_data)
        print(f"Added non-zero values at {added_non_zeros}/{len(hour_range)} time steps.")


def round_to_multiple(numbers: np.ndarray, multiple: float = 1 / 12):
    return multiple * np.round_(numbers / multiple)


def area_handler(area: str):
    if area == "area1":
        lon_range, lat_range = (-146.25, -125), (15, 36.25)
    elif area == "area3":
        lon_range, lat_range = (-116.25, -95), (-11.25, 10)
    else:
        raise NotImplementedError("Not implemented for specified area!")
    return lon_range, lat_range


def save_sparse_npy_datasets(input_dir: str, output_np_dir: str, output_np_test_dir: str, area: str, type="forecast"):
    lon_range, lat_range = area_handler(area)
    files = sorted(os.listdir(input_dir))
    len_train = int(0.9*len(files))
    train_val_files = files[:len_train]
    test_files = files[len_train:]

    # save training/val files
    # save_sparse_as_npy(input_dir, train_val_files, output_np_dir, lon_range, lat_range, type=type)
    # save test files
    save_sparse_as_npy(input_dir, test_files, output_np_test_dir, lon_range, lat_range, type=type)


def save_dense_npy_datasets(input_dir: str, output_np_dir: str, output_np_test_dir: str, area: str):
    lon_range, lat_range = area_handler(area)
    files = sorted(os.listdir(input_dir))
    len_train = int(0.9*len(files))
    train_val_files = files[:len_train]
    test_files = files[len_train:]

    # save_nc_as_npy(input_dir, train_val_files, output_np_dir, lon_range, lat_range)
    save_nc_as_npy(input_dir, test_files, output_np_test_dir, lon_range, lat_range)


def main():
    area = "area1"

    # buoy data
    sparse_dir = f"data/drifter_data/dataset_forecast_error/{area}_edited2"
    sparse_np_dir = f"data/drifter_data/buoy_preprocessed/{area}"
    sparse_np_test_dir = f"data/drifter_data/buoy_preprocessed_test/{area}"
    save_sparse_npy_datasets(sparse_dir, sparse_np_dir, sparse_np_test_dir, area)

    # # forecast data
    # fc_dir = f"data/drifter_data/forecasts/{area}"
    # fc_np_dir = f"data/drifter_data/forecasts_preprocessed/{area}"
    # fc_np_test_dir = f"data/drifter_data/forecasts_preprocessed_test/{area}"
    # save_dense_npy_datasets(fc_dir, fc_np_dir, fc_np_test_dir, area)

    # hc_dir = f"data/drifter_data/hindcasts/{area}"
    # hc_np_dir = f"data/drifter_data/hindcasts_preprocessed/{area}"
    # hc_np_test_dir = f"data/drifter_data/hindcasts_preprocessed_test/{area}"
    # save_dense_npy_datasets(hc_dir, hc_np_dir, hc_np_test_dir, area)


if __name__ == "__main__":
    main()
