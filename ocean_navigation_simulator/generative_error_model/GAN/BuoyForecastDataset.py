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
    def __init__(self, fc_dir, buoy_dir, areas=("area1"), concat_len=1, dataset_names=None, transform=None):
        """Note: use 24*8 for length because not all files have full 9 days."""
        self.fc_dir = fc_dir
        self.buoy_dir = buoy_dir
        self.dataset_names = dataset_names
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

        self.fc_data = [np.load(file_path, mmap_mode="r+", allow_pickle=True)[:self.hours_in_file]
                        for file_path in self.fc_file_paths]
        self.buoy_data = [np.load(file_path, mmap_mode="r+", allow_pickle=True)[:self.hours_in_file]
                          for file_path in self.buoy_file_paths]

        if transform:
            if dataset_names is None:
                raise ValueError("Need to specify dataset names to match to mean and std")
            # self.fc_data_mean, self.fc_data_std = compute_mean_std(self.fc_data)
            # self.buoy_data_mean, self.buoy_data_std = compute_mean_std(self.buoy_data)

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

        if self.transform:
            transformed_data = []
            for data, name in zip([fc_data, buoy_data], self.dataset_names):
                # move channels to last dim
                data = np.moveaxis(data, 0, -1)
                mean, std = get_dataset_mean_std(name)
                data_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
                transformed_data.append(data_transforms(data))
            return transformed_data
        return fc_data, buoy_data


def round_to_multiple(numbers: np.ndarray, multiple: float = 1 / 12):
    return multiple * np.round_(numbers / multiple)


def get_dataset_mean_std(name):
    """Return pre-computed means and stds for different datasets.
    Not ideal way to do it but its purely for convenience, since computing these takes quite long.
    """

    if name == "buoy":  # buoy_preprocessed
        mean = [-2.69476191e-05, 1.67980179e-05]
        std = [0.00549126, 0.00467224]
    elif name == "fc":   # forecasts_preprocessed
        mean = [-0.11190069, -0.04891662]
        std = [0.14017195, 0.11500811]
    elif name == "synthetic_data":   # synthetic data from TSN
        mean = [-0.05854823, -0.06621165]
        std = [0.17166248, 0.14639856]
    else:
        print("WARNING: No normalization metrics found! Check config!")
        mean = [0, 0]
        std = [1, 1]
    return mean, std


def compute_mean_std(data):
    """Compute mean and std for both data in dataset.
    Note: Data expected to be loaded like above in Dataset using a list of numpy references.
    Expect data to be list of items of shape [frames, 2, 256, 256].
    Note 2: Cannot use standard mean method for sparse data -> compute mean the longer way
    """

    mean = np.array([np.sum(file, axis=(0, 2, 3))/np.sum(file != 0, axis=(0, 2, 3)) for file in data]).mean(axis=0)
    helper_mean = np.zeros_like(data[0])
    helper_mean[:, 0, :, :], helper_mean[:, 1, :, :] = mean[0], mean[1]
    std = np.sqrt(np.array([np.sum((file - helper_mean)**2, axis=(0, 2, 3))/(np.prod(file.shape)/2)
                            for file in data]).sum(axis=0)/len(data))
    return mean, std


def compute_mean_std2(data):
    """Using the more classical way of computing running mean and std."""

    cnt = 0
    fst_moment = np.zeros(2)
    snd_moment = np.zeros(2)

    for file in data:
        f, c, h, w = file.shape
        file_channel_pixels = f * h * w
        sum_ = np.sum(file, axis=(0, 2, 3))
        sum_of_square_ = np.sum(file ** 2, axis=(0, 2, 3))
        fst_moment = (cnt * fst_moment + sum_) / (cnt + file_channel_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square_) / (cnt + file_channel_pixels)
        cnt += file_channel_pixels
    mean, std = fst_moment, np.sqrt(snd_moment - fst_moment ** 2)
    return mean, std


def main():
    fc_dir = "../../../data/drifter_data/forecasts_preprocessed"
    # gt_dir = "../../../data/drifter_data/buoy_preprocessed"
    gt_dir = "../../../data/drifter_data/synthetic_data/train_val"
    dataset = BuoyForecastErrorNpy(fc_dir, gt_dir, areas=["area1"], dataset_names=("fc", "buoy"), transform=True)
    dataset_item = dataset.__getitem__(0)
    print(f"dataset item output shapes: {dataset_item[0].shape}, {dataset_item[1].shape}")


if __name__ == "__main__":
    main()
