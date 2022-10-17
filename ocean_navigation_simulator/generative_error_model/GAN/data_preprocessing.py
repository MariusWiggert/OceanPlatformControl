import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple


def save_nc_as_npy(file_dir: str, output_dir: str, lon_range: Tuple, lat_range: Tuple):
    """Takes a .nc file and saves it as a .npy file."""

    files = sorted(os.listdir(file_dir))
    for file in files:
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
        np.save(os.path.join(output_dir, "".join(file.split(".")[:-1])) + ".npy", data)


def save_sparse_as_npy(file_dir: str, output_dir: str, lon_range: Tuple, lat_range: Tuple):
    """Takes sparse buoy data, turns it into a sparse matrix and saves it as an .npy file."""
    files = sorted(os.listdir(file_dir))
    for file in files:
        data = pd.read_csv(os.path.join(file_dir, file))
        data["hour"] = data["time"].apply(lambda x: x[:13])
        hours = sorted(list(set(data["hour"].tolist())))
        # TODO: make sure number of hours matches the time specified in name of file!
        output_data = np.zeros((len(hours), 2, len(list(np.arange(*lon_range, 1/12)))+1, len(list(np.arange(*lat_range, 1/12)))+1))
        for time_step, hour in enumerate(hours):
            data_time_step = data[data["hour"].values == hour]
            # convert from sparse to dense
            points = np.array([data_time_step["lon"], data_time_step["lat"]])
            nearest_grid_points = round_to_multiple(points)
            lon_idx = np.searchsorted(np.arange(*lon_range, 1/12), nearest_grid_points[0])
            lat_idx = np.searchsorted(np.arange(*lat_range, 1/12), nearest_grid_points[1])
            output_data[time_step, 0, lon_idx, lat_idx] = data_time_step["u"].values
            output_data[time_step, 1, lon_idx, lat_idx] = data_time_step["v"].values
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, "".join(file.split(".")[:-1])) + ".npy", output_data)


def round_to_multiple(numbers: np.ndarray, multiple: float = 1 / 12):
    return multiple * np.round_(numbers / multiple)


if __name__ == "__main__":
    fc_dir = "../../../data/drifter_data/forecasts/area1"
    hc_dir = "../../../data/drifter_data/hindcasts/area1"
    sparse_dir = "../../../data/drifter_data/dataset_forecast_error/area1"

    fc_np_dir = "../../../data/drifter_data/forecasts_preprocessed"
    hc_np_dir = "../../../data/drifter_data/hindcasts_preprocessed"
    sparse_np_dir = "../../../data/drifter_data/buoy_preprocessed"

    lon_range, lat_range = (-146.25, -125), (15, 36.25)
    # save_nc_as_npy(fc_dir, fc_np_dir, lon_range, lat_range)
    # save_nc_as_npy(hc_dir, hc_np_dir, lon_range, lat_range)
    save_sparse_as_npy(sparse_dir, sparse_np_dir, lon_range, lat_range)

    # print(np.load(os.path.join(sparse_np_dir, os.listdir(sparse_np_dir)[0]), allow_pickle=True).shape)
    # print(np.load(os.path.join(fc_np_dir, os.listdir(fc_np_dir)[0]), allow_pickle=True).shape)
