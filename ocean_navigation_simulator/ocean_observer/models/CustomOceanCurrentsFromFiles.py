import os

import numpy as np
from torch.utils.data import Dataset


class CustomOceanCurrentsFromFiles(Dataset):
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
            full_path = os.path.join(folder, path)
            if os.path.isfile(full_path):
                if path.endswith("_X.npy"):
                    Xs.append(full_path)
                if path.endswith("_y.npy"):
                    ys.append(full_path)
        print("loading files:", Xs, ys)
        self.X = np.concatenate([np.load(fname) for fname in Xs])
        self.y = np.concatenate([np.load(fname) for fname in ys])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
