import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset
import os
from warnings import warn
import torchvision.transforms as transforms


class BuoyForecastError(Dataset):

    def __init__(self, fc_dir, gt_dir, transform=None) -> None:
        self.fc_dir = fc_dir
        self.gt_dir = gt_dir
        self.transform = transform

        # get all required file names
        self.fc_file_names = sorted(os.listdir(self.fc_dir))
        self.error_file_names = sorted(os.listdir(self.gt_dir))

        if len(self.fc_file_names) != len(self.error_file_names):
            warn("Datasets do not contain same number of files!")

    def __len__(self):
        return

    def __getitem__(self, idx):

        file_idx = idx // 227
        time_step_idx = idx % 227

        self.fc_file_name = self.fc_file_names[file_idx]
        self.error_file_name = self.error_file_names[file_idx]

        # get time step for FC
        fc_file = xr.load_dataset(os.path.join(self.fc_dir, self.fc_file_name)).sel(longitude=slice(-140, -120 - 1/12),
                                                         latitude=slice(20, 30 - 1/12))
        fc_file_u = fc_file["utotal"]
        fc_file_v = fc_file["vtotal"]
        fc_u = fc_file_u.isel(time=time_step_idx).values.squeeze()
        fc_u = fc_u[:, :, np.newaxis]
        fc_v = fc_file_v.isel(time=time_step_idx).values.squeeze()
        fc_v = fc_v[:, :, np.newaxis]
        fc = np.concatenate([fc_u, fc_v], axis=2)

        # get time step for error
        error_file = pd.read_csv(os.path.join(self.gt_dir, self.error_file_name))
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
    data_dir = "/home/jonas/Documents/Thesis/OceanPlatformControl"
    fc_dir = os.path.join(data_dir, "data/drifter_data/forecasts/area1")
    gt_dir = os.path.join(data_dir, "data/drifter_data/dataset_forecast_error/area1")
    dataset = BuoyForecastError(fc_dir, gt_dir)
    dataset_item = dataset.__getitem__(0)
    print(f"dataset item output shapes: {dataset_item[0].shape}, {dataset_item[1].shape}")


if __name__ == "__main__":
    test()
