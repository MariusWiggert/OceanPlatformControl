from torch.utils.data import Dataset
import os
import xarray as xr
import numpy as np


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
        fc_file = xr.load_dataset(os.path.join(self.fc_dir, fc_file_name)).sel(longitude=slice(-146.25, -125),
                                                                               latitude=slice(15, 36.25))
        fc_file_u = fc_file["utotal"]
        fc_file_v = fc_file["vtotal"]
        fc_u = fc_file_u.isel(time=time_step_idx).values.squeeze()
        fc_u = fc_u[np.newaxis, :, :]
        fc_v = fc_file_v.isel(time=time_step_idx).values.squeeze()
        fc_v = fc_v[np.newaxis, :, :]
        fc = np.concatenate([fc_u, fc_v], axis=0)

        hc_file_name = self.hc_file_names[file_idx]
        hc_file = xr.load_dataset(os.path.join(self.hc_dir, hc_file_name)).sel(lon=slice(-146.25, -125),
                                                                               lat=slice(15, 36.25))
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


def test():
    data_dir = "/home/jonas/Documents/Thesis/OceanPlatformControl"
    fc_dir = os.path.join(data_dir, "data/drifter_data/forecasts/area1")
    hc_dir = os.path.join(data_dir, "data/drifter_data/hindcasts/area1")
    dataset = ForecastHindcastDataset(fc_dir, hc_dir)
    idx = np.random.randint(1000)
    dataset_item = dataset.__getitem__(0)
    print(f"dataset item output shapes: {dataset_item[0].shape}, {dataset_item[1].shape}")

    import matplotlib.pyplot as plt
    plt.imshow(dataset_item[0][0], origin="lower")
    plt.show()

    plt.imshow(dataset_item[1][0], origin="lower")
    plt.show()

    plt.imshow(dataset_item[1][0] - dataset_item[0][0], origin="lower")
    plt.show()


if __name__ == "__main__":
    test()
