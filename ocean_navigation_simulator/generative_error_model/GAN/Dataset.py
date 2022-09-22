import torch
import numpy as np
import xarray as xr
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import glob


class BuoyForecastError(Dataset):

    def __init__(self, data_dir, transform=None) -> None:
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return

    def __getitem__(self, idx):

        file_idx = idx // 227
        time_step_idx = idx % 227

        fc_dir = os.path.join(self.data_dir, "forecasts/area1")
        error_dir = os.path.join(self.data_dir, "dataset_forecast_error/area1")

        fc_file_name = sorted(os.listdir(fc_dir))[file_idx]
        error_file_name = sorted(os.listdir(error_dir))[file_idx]

        # get time step for FC
        fc_file = xr.load_dataset(os.path.join(fc_dir, fc_file_name)).sel(longitude=slice(-140, -120 - 1/12),
                                                                          latitude=slice(20, 30 - 1/12))
        fc_file_u = fc_file["utotal"]
        fc_file_v = fc_file["vtotal"]
        fc_u = fc_file_u.isel(time=time_step_idx).values.squeeze()
        fc_u = fc_u[:, :, np.newaxis]
        fc_v = fc_file_v.isel(time=time_step_idx).values.squeeze()
        fc_v = fc_v[:, :, np.newaxis]
        fc = np.concatenate([fc_u, fc_v], axis=2)

        # get time step for error
        error_file = pd.read_csv(os.path.join(error_dir, error_file_name))
        error_file["hour"] = error_file["time"].apply(lambda x: x[:13])
        hours = sorted(set(error_file["hour"].tolist()))
        sparse_error_time_step = error_file[error_file["hour"].values == hours[time_step_idx]]

        # get sparse data on FC grid.
        points = np.array([sparse_error_time_step["lon"], sparse_error_time_step["lat"]])
        nearest_grid_points = round_to_multiple(points)
        lon_idx = np.searchsorted(fc_file["longitude"].values, nearest_grid_points[0])
        lat_idx = np.searchsorted(fc_file["latitude"].values, nearest_grid_points[1])

        # write sparse error arrays
        sparse_error_u = np.zeros_like(fc_u)
        sparse_error_u[lat_idx, lon_idx] = sparse_error_time_step["u_error"].values.reshape(-1, 1)
        sparse_error_v = np.zeros_like(fc_v)
        sparse_error_v[lat_idx, lon_idx] = sparse_error_time_step["v_error"].values.reshape(-1, 1)
        sparse_error = np.concatenate([sparse_error_u, sparse_error_v], axis=2)

        if self.transform:
            # apply transformations
            pass
        return fc, sparse_error


def round_to_multiple(numbers: np.ndarray, multiple: float = 1 / 12):
    return multiple * np.round_(numbers / multiple)


def test():
    data_dir = "/home/jonas/Documents/Thesis/OceanPlatformControl/data/drifter_data"
    dataset = BuoyForecastError(data_dir)
    dataset_item = dataset.__getitem__(0)
    print(f"dataset item output shapes: {dataset_item[0].shape}, {dataset_item[1].shape}")


if __name__ == "__main__":
    test()
