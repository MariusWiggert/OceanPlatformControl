import os
from typing import List

import numpy as np
from torch.utils.data import Dataset


class CustomOceanCurrentsFromFiles(Dataset):
    IDX_LON, IDX_LAT, IDX_TIME = 0, 1, 2
    MARGIN = 0.5
    GULF_MEXICO_WITHOUT_MARGIN = [[-97.84, -76.42], [18.08, 30]]
    GULF_MEXICO = [[GULF_MEXICO_WITHOUT_MARGIN[0][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[0][1] - MARGIN],
                   [GULF_MEXICO_WITHOUT_MARGIN[1][0] + MARGIN, GULF_MEXICO_WITHOUT_MARGIN[1][1] - MARGIN]]

    def __init__(self, folders: List[str], max_items: int = None):
        if max_items is not None and max_items < 0:
            max_items = None
        # folder = "data_exported"
        Xs, ys = list(), list()
        for folder in folders:
            for path in sorted(os.listdir(folder)):
                # check if current path is a file
                full_path = os.path.join(folder, path)
                if os.path.isfile(full_path):
                    if path.lower().endswith("_x.npy"):
                        Xs.append(full_path)
                    if path.lower().endswith("_y.npy"):
                        ys.append(full_path)
        print("loading files:", Xs, ys)
        # TODO: remove the slicing
        # self.X = np.concatenate([np.load(fname) for fname in Xs])
        # self.y = np.concatenate([np.load(fname) for fname in ys])
        self.X = [np.load(fname, mmap_mode="r") for fname in Xs]
        self.y = [np.load(fname, mmap_mode="r") for fname in ys]
        self.lens = np.array([len(x) for x in self.X])
        self.len = self.lens.sum()
        assert self.len == np.array([len(y) for y in self.y]).sum()
        print(f"shapes X: {len(self.X), self.X[0].shape}, y: {len(self.y), self.y[0].shape}")

        # todo: fix that
        if max_items is not None and max_items > 0:
            print(f"Only considering the first {max_items} items over {len(self.X)}.")
            # self.X = self.X[:max_items]
            # self.y = self.y[:max_items]
            self.len = max_items

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        i = 0
        while idx >= len(self.X[i]):
            idx -= len(self.X[i])
            i += 1
        x, y = self.X[i][idx], self.y[i][idx]

        if np.isnan(x).any() or np.isnan(y).any():
            return None, None

        return x, y
        # return self.X[idx], self.y[idx]
