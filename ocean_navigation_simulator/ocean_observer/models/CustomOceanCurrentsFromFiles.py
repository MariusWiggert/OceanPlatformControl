import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomOceanCurrentsDatasetSubgrid(Dataset):
    IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
    MARGIN = 0.5
    GULF_MEXICO_WITHOUT_MARGIN = [[-97.84, -76.42], [18.08, 30]]
    GULF_MEXICO = [[GULF_MEXICO_WITHOUT_MARGIN[0][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[0][1] - MARGIN],
                   [GULF_MEXICO_WITHOUT_MARGIN[1][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[1][1] - MARGIN]]

    def __init__(self, folder: str):
        # folder = "data_exported"
        Xs, ys = list(), list()
        for path in os.listdir(folder):
            # check if current path is a file
            if os.path.isfile(os.path.join(folder, path)):
                if path.endswith("_X.npy"):
                    Xs.append(path)
                if path.endswith("_y.npy"):
                    ys.append(path)
        print(Xs, ys)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        lon, lat, time = self.inputs[idx][0:3]

        lo, la, ti = np.logical_and(lon[0] < self.whole_grid_fc['lon'], self.whole_grid_fc['lon'] < lon[1]), \
                     np.logical_and(lat[0] < self.whole_grid_fc['lat'], self.whole_grid_fc['lat'] < lat[1]), \
                     np.logical_and(np.datetime64(time[0]) < self.whole_grid_fc['time'],
                                    self.whole_grid_fc['time'] < np.datetime64(time[1]))
        X = self.whole_grid_fc.isel(lon=lo, lat=la, time=ti).to_array().to_numpy()

        # X = self.whole_grid_fc.sel(lon=slice(*lon), lat=slice(*lat), time=slice(*time)).to_array().to_numpy()
        lon, lat, time = self.outputs[idx][0:3]
        lo, la, ti = np.logical_and(lon[0] < self.whole_grid_hc['lon'], self.whole_grid_hc['lon'] < lon[1]), \
                     np.logical_and(lat[0] < self.whole_grid_hc['lat'], self.whole_grid_hc['lat'] < lat[1]), \
                     np.logical_and(np.datetime64(time[0]) < self.whole_grid_hc['time'],
                                    self.whole_grid_hc['time'] < np.datetime64(time[1]))
        y = self.whole_grid_hc.isel(lon=lo, lat=la, time=ti).to_array().to_numpy()

        if list(X.shape) != self.input_shape or list(y.shape) != self.output_shape:
            print(time[0], time[1], lon[0], lat[0])
            return None
        X, y = torch.tensor(X, dtype=self.dtype), torch.tensor(y, dtype=self.dtype)
        X[torch.isnan(X)] = 0
        return X, y
