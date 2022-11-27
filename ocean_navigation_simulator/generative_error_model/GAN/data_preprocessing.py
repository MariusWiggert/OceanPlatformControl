from ocean_navigation_simulator.generative_error_model.utils import hour_range_file_name, time_range_hrs, datetime2str

import os
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple


def save_nc_as_npy(file_dir: str, output_dir: str, lon_range: Tuple, lat_range: Tuple):
    """Takes a .nc file and saves it as a .npy file."""

    files = sorted(os.listdir(file_dir))
    for file in files:
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


def save_sparse_as_npy(file_dir: str, output_dir: str, lon_range: Tuple, lat_range: Tuple):
    """Takes sparse buoy data, turns it into a sparse matrix and saves it as an .npy file."""
    files = sorted(os.listdir(file_dir))
    for file in files:
        output_file_path = os.path.join(output_dir, "".join(file.split(".")[:-1])) + ".npy"
        if os.path.exists(output_file_path):
            print(f"File {''.join(output_file_path.split('/')[-1])} already exists!")
            continue
        data = pd.read_csv(os.path.join(file_dir, file))
        data["hour"] = data["time"].apply(lambda x: x[:13])
        hours = sorted(list(set(data["hour"].tolist())))
        hours_according_file_name, start_date = hour_range_file_name(file)
        if hours != hours_according_file_name:
            print("No buoy data available for every hour")
        # get time range in hours to iterate over and populate array
        hour_range = time_range_hrs(start_date, hours_according_file_name)
        output_data = np.zeros((hours_according_file_name,
                                2,
                                len(list(np.arange(*lon_range, 1/12)))+1,
                                len(list(np.arange(*lat_range, 1/12)))+1))
        # need to make sure that the hours of the buoy data match the dense fc/hc data
        # if there is no buoy data for a particular hour, an empty frame is inserted
        for time_step, hour in enumerate(hour_range):
            hour = datetime2str(hour)[:13].replace("T", " ")
            if hour in data["hour"].values:
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
        np.save(output_file_path, output_data)


def round_to_multiple(numbers: np.ndarray, multiple: float = 1 / 12):
    return multiple * np.round_(numbers / multiple)


def run(area: str, buoy=False):
    if area == "area1":
        lon_range, lat_range = (-146.25, -125), (15, 36.25)
    elif area == "area3":
        lon_range, lat_range = (-116.25, -95), (-11.25, 10)
    else:
        raise NotImplementedError("Not implemented for specified area!")

    fc_dir = f"data/drifter_data/forecasts/{area}"
    hc_dir = f"data/drifter_data/hindcasts/{area}"
    sparse_dir = f"data/drifter_data/dataset_forecast_error/{area}_edited2"

    fc_np_dir = f"data/drifter_data/forecasts_preprocessed/{area}"
    hc_np_dir = f"data/drifter_data/hindcasts_preprocessed/{area}"
    sparse_np_dir = f"data/drifter_data/buoy_preprocessed/{area}"

    # save_nc_as_npy(fc_dir, fc_np_dir, lon_range, lat_range)
    # save_nc_as_npy(hc_dir, hc_np_dir, lon_range, lat_range)
    if buoy:
        save_sparse_as_npy(sparse_dir, sparse_np_dir, lon_range, lat_range)
    print("Finished.")


if __name__ == "__main__":
    run("area1", buoy=True)
