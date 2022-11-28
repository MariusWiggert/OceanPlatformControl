from ocean_navigation_simulator.generative_error_model.utils import get_datetime_from_file_name,\
    get_time_matched_file_lists

import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset
import os
from warnings import warn
import glob
import torchvision.transforms as transforms


class BuoyForecastError(Dataset):

    def __init__(self, fc_dir, gt_dir, sparse_type="error", length=None, transform=None) -> None:
        self.fc_dir = fc_dir
        self.gt_dir = gt_dir
        self.sparse_type = sparse_type
        self.length = length
        self.transform = transform
        self.hours_in_file = 24

        # get all required file names
        self.fc_file_names = sorted(os.listdir(self.fc_dir))
        self.sparse_data_file_names = sorted(os.listdir(self.gt_dir))

        if len(self.fc_file_names) != len(self.sparse_data_file_names):
            warn("Datasets do not contain same number of files!")

    def __len__(self):
        if self.length:
            return self.length
        else:
            return len(self.fc_file_names) * self.hours_in_file

    def __getitem__(self, idx):

        file_idx = idx // self.hours_in_file
        time_step_idx = idx % self.hours_in_file

        fc_file_name = self.fc_file_names[file_idx]
        sparse_data_file_name = self.sparse_data_file_names[file_idx]

        # get time step for FC
        fc_file = xr.load_dataset(os.path.join(self.fc_dir, fc_file_name)).sel(longitude=slice(-140, -120),
                                                                                    latitude=slice(20, 30))
        fc_file_u = fc_file["utotal"]
        fc_file_v = fc_file["vtotal"]
        fc_u = fc_file_u.isel(time=time_step_idx).values.squeeze()
        fc_u = fc_u[np.newaxis, :, :]
        fc_v = fc_file_v.isel(time=time_step_idx).values.squeeze()
        fc_v = fc_v[np.newaxis, :, :]
        fc = np.concatenate([fc_u, fc_v], axis=0)

        # get time step for error
        sparse_data = pd.read_csv(os.path.join(self.gt_dir, sparse_data_file_name))
        sparse_data["hour"] = sparse_data["time"].apply(lambda x: x[:13])
        hours = sorted(set(sparse_data["hour"].tolist()))
        # if len(hours) != self.hours_in_file:
        #     warn("Time axis are not equal between FC and Sparse GT!")
        # print(len(hours), time_step_idx, idx)
        sparse_data_time_step = sparse_data[sparse_data["hour"].values == hours[time_step_idx]]

        # get sparse data on FC grid.
        points = np.array([sparse_data_time_step["lon"], sparse_data_time_step["lat"]])
        nearest_grid_points = round_to_multiple(points)
        lon_idx = np.searchsorted(fc_file["longitude"].values, nearest_grid_points[0])
        lat_idx = np.searchsorted(fc_file["latitude"].values, nearest_grid_points[1])

        # write sparse error arrays
        sparse_data_u = np.zeros_like(fc_u).squeeze()
        sparse_data_v = np.zeros_like(fc_v).squeeze()
        if self.sparse_type == "error":
            sparse_data_u[lat_idx, lon_idx] = sparse_data_time_step["u_error"].values.squeeze()
            sparse_data_v[lat_idx, lon_idx] = sparse_data_time_step["v_error"].values.squeeze()
        elif self.sparse_type == "forecast":
            sparse_data_u[lat_idx, lon_idx] = sparse_data_time_step["u_forecast"].values.squeeze()
            sparse_data_v[lat_idx, lon_idx] = sparse_data_time_step["v_forecast"].values.squeeze()

        sparse_data_u = sparse_data_u[np.newaxis, :, :]
        sparse_data_v = sparse_data_v[np.newaxis, :, :]

        sparse_data = np.concatenate([sparse_data_u, sparse_data_v], axis=0)

        if self.transform:
            # apply transformations
            pass
        return fc, sparse_data


class BuoyForecastErrorNpy(Dataset):
    def __init__(self, fc_dir, buoy_dir, areas=("area1"), concat_len=1, transform=None):
        """Note: use 24*8 for length because not all files have full 9 days."""
        self.fc_dir = fc_dir
        self.buoy_dir = buoy_dir
        self.transform = transform
        self.hours_in_file = 24 * 8
        self.concat_len = concat_len

        self.area_lens = {}
        self.fc_file_paths = []
        self.buoy_file_paths = []
        for area in list(areas):
            try:
                fc_file_paths = sorted(glob.glob(f"{fc_dir}/{area}/*.npy"))
                buoy_file_paths = sorted(glob.glob(f"{buoy_dir}/{area}/*.npy"))
            except:
                raise ValueError("Specified area does not exist!")

            if len(fc_file_paths) != len(buoy_file_paths):
                print("\nNumber of forecast and buoy files is different!\n")

            # compare dates of files and make sure they match!
            fc_file_paths, buoy_file_paths = get_time_matched_file_lists(fc_file_paths, buoy_file_paths)

            self.area_lens[area] = len(fc_file_paths) * (self.hours_in_file//self.concat_len)
            self.fc_file_paths.extend(fc_file_paths)
            self.buoy_file_paths.extend(buoy_file_paths)

        self.fc_data = [np.load(file_path, mmap_mode="r+", allow_pickle=True) for file_path in self.fc_file_paths]
        self.buoy_data = [np.load(file_path, mmap_mode="r+", allow_pickle=True) for file_path in self.buoy_file_paths]

    def __len__(self):
        return min(len(self.fc_file_paths), len(self.buoy_file_paths)) * (self.hours_in_file//self.concat_len)

    def __getitem__(self, idx):
        if self.concat_len == 1:
            file_idx = idx // self.hours_in_file
            time_step_idx = idx % self.hours_in_file
            fc_data = self.fc_data[file_idx][time_step_idx].squeeze()
            buoy_data = self.buoy_data[file_idx][time_step_idx].squeeze()
        else:
            file_idx = (idx * self.concat_len + self.concat_len - 1) // self.hours_in_file
            time_step_idx = (idx * self.concat_len) % self.hours_in_file
            fc_data = self.fc_data[file_idx][time_step_idx: time_step_idx + self.concat_len].squeeze()
            buoy_data = self.buoy_data[file_idx][time_step_idx].squeeze()
            fc_data = fc_data.reshape(-1, fc_data.shape[-2], fc_data.shape[-1])

        assert fc_data.shape[0] == 2 * self.concat_len, "Error with concatting time steps!"
        return fc_data, buoy_data


def round_to_multiple(numbers: np.ndarray, multiple: float = 1 / 12):
    return multiple * np.round_(numbers / multiple)


def main():
    fc_dir = "data/drifter_data/forecasts_preprocessed"
    gt_dir = "data/drifter_data/buoy_preprocessed"
    dataset = BuoyForecastErrorNpy(fc_dir, gt_dir, areas=["area1"])
    dataset_item = dataset.__getitem__(0)
    print(f"dataset item output shapes: {dataset_item[0].shape}, {dataset_item[1].shape}")


if __name__ == "__main__":
    main()
