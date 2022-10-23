from torch.utils.data import Dataset
import os
import xarray as xr
import numpy as np
import glob


class ForecastHindcastDataset(Dataset):

    def __init__(self, fc_dir, hc_dir, transform=None):
        self.fc_dir = fc_dir
        self.hc_dir = hc_dir
        self.transform = transform
        self.hours_in_file = 24 * 8

        self.fc_file_names = sorted(os.listdir(self.fc_dir))
        self.hc_file_names = sorted(os.listdir(self.hc_dir))

    def __len__(self):
        return min(len(self.fc_file_names), len(self.hc_file_names)) * self.hours_in_file

    def __getitem__(self, idx):
        file_idx = idx // self.hours_in_file
        time_step_idx = idx % self.hours_in_file

        fc_file_name = self.fc_file_names[file_idx]
        # fc_file = xr.load_dataset(os.path.join(self.fc_dir, fc_file_name)).sel(longitude=slice(-146.25, -125),
        #                                                                        latitude=slice(15, 36.25))
        fc_file = xr.load_dataset(os.path.join(self.fc_dir, fc_file_name)).sel(longitude=slice(-116.25, -95),
                                                                               latitude=slice(-11.25, 10))
        fc_file_u = fc_file["utotal"]
        fc_file_v = fc_file["vtotal"]
        fc_u = fc_file_u.isel(time=time_step_idx).values.squeeze()
        fc_u = fc_u[np.newaxis, :, :]
        fc_v = fc_file_v.isel(time=time_step_idx).values.squeeze()
        fc_v = fc_v[np.newaxis, :, :]
        fc = np.concatenate([fc_u, fc_v], axis=0)

        hc_file_name = self.hc_file_names[file_idx]
        # hc_file = xr.load_dataset(os.path.join(self.hc_dir, hc_file_name)).sel(lon=slice(-146.25, -125),
        #                                                                        lat=slice(15, 36.25))
        hc_file = xr.load_dataset(os.path.join(self.hc_dir, hc_file_name)).sel(lon=slice(-116.25, -95),
                                                                               lat=slice(-11.25, 10))
        hc_file_u = hc_file["water_u"]
        hc_file_v = hc_file["water_v"]
        hc_u = hc_file_u.isel(time=time_step_idx).values.squeeze()
        hc_u = hc_u[np.newaxis, :, :]
        hc_v = hc_file_v.isel(time=time_step_idx).values.squeeze()
        hc_v = hc_v[np.newaxis, :, :]
        hc = np.concatenate([hc_u, hc_v], axis=0)

        if self.transform:
            # apply transformations
            pass
        return fc, hc


class ForecastHindcastDatasetNpy(Dataset):
    def __init__(self, fc_dir, hc_dir, areas=("area1"), concat_len=1, transform=None):
        self.fc_dir = fc_dir
        self.hc_dir = hc_dir
        self.transform = transform
        self.hours_in_file = 24 * 8
        self.concat_len = concat_len

        self.area_lens = {}
        self.fc_file_paths = []
        self.hc_file_paths = []
        for area in areas:
            try:
                fc_file_paths = sorted(glob.glob(f"{fc_dir}/{area}/*.npy"))
                hc_file_paths = sorted(glob.glob(f"{hc_dir}/{area}/*.npy"))
                if len(fc_file_paths) != len(hc_file_paths):
                    raise ValueError("Number of forecast and hindcast files is different!")
                self.area_lens[area] = len(fc_file_paths) * (self.hours_in_file//self.concat_len)
                self.fc_file_paths.extend(fc_file_paths)
                self.hc_file_paths.extend(hc_file_paths)
            except:
                raise ValueError("Specified area does not exist!")

        self.fc_data = [np.load(file_path, mmap_mode="r+", allow_pickle=True) for file_path in self.fc_file_paths]
        self.hc_data = [np.load(file_path, mmap_mode="r+", allow_pickle=True) for file_path in self.hc_file_paths]

    def __len__(self):
        return min(len(self.fc_file_paths), len(self.hc_file_paths)) * (self.hours_in_file//self.concat_len)

    def __getitem__(self, idx):
        if self.concat_len == 1:
            file_idx = idx // self.hours_in_file
            time_step_idx = idx % self.hours_in_file
            fc_data = self.fc_data[file_idx][time_step_idx].squeeze()
            hc_data = self.hc_data[file_idx][time_step_idx].squeeze()
        else:
            file_idx = (idx * self.concat_len + self.concat_len - 1) // self.hours_in_file
            time_step_idx = (idx * self.concat_len) % self.hours_in_file
            fc_data = self.fc_data[file_idx][time_step_idx: time_step_idx + self.concat_len].squeeze()
            hc_data = self.hc_data[file_idx][time_step_idx].squeeze()
            fc_data = fc_data.reshape(-1, fc_data.shape[-2], fc_data.shape[-1])

        assert fc_data.shape[0] == 2*self.concat_len, "Error with concatting time steps!"
        return fc_data, hc_data


def test_xr():
    data_dir = "/home/jonas/Documents/Thesis/OceanPlatformControl"
    fc_dir = os.path.join(data_dir, "data/drifter_data/forecasts/area3")
    hc_dir = os.path.join(data_dir, "data/drifter_data/hindcasts/area3")
    dataset = ForecastHindcastDataset(fc_dir, hc_dir)
    dataset_item = dataset.__getitem__(0)
    print(f"dataset item output shapes: {dataset_item[0].shape}, {dataset_item[1].shape}")

    import matplotlib.pyplot as plt
    plt.imshow(dataset_item[0][0], origin="lower")
    plt.show()
    plt.imshow(dataset_item[1][0], origin="lower")
    plt.show()
    plt.imshow(dataset_item[1][0] - dataset_item[0][0], origin="lower")
    plt.show()


def test_npy():
    data_dir = "/home/jonas/Documents/Thesis/OceanPlatformControl"
    fc_dir = os.path.join("data/drifter_data/forecasts_preprocessed")
    hc_dir = os.path.join("data/drifter_data/hindcasts_preprocessed")
    dataset = ForecastHindcastDatasetNpy(fc_dir, hc_dir, areas=["area1"], concat_len=2)
    print(f"Dataset length: {len(dataset)}")
    dataset_item = dataset.__getitem__(int(30144/2)-1)
    print(f"dataset item output shapes: {dataset_item[0].shape}, {dataset_item[1].shape}")


if __name__ == "__main__":
    test_npy()
