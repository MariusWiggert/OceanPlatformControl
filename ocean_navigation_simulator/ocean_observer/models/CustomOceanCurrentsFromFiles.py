import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomOceanCurrentsFromFiles(Dataset):
    def __init__(
        self,
        folders: List[str] = None,
        max_items: int = None,
        tile_size=None,
        filename_with_path: str = None,
        return_GP_FC_IMP_FC: bool = False,
    ):
        self.return_GP_FC_IMP_FC = return_GP_FC_IMP_FC
        if max_items is not None and max_items < 0:
            max_items = None
        # folder = "data_exported"
        Xs, ys = list(), list()

        if folders is None:
            if filename_with_path is None:
                raise AttributeError("No filename nor folder specified.")
            Xs.append(filename_with_path + "_x.npy")
            ys.append(filename_with_path + "_y.npy")
        else:
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
        self.X = [
            np.load(fname, mmap_mode="r") for fname in Xs
        ]  # [np.load(fname, mmap_mode="r") for fname in Xs]
        self.y = [
            np.load(fname, mmap_mode="r") for fname in ys
        ]  # [np.load(fname, mmap_mode="r") for fname in ys]
        self.lens = np.array([len(x) for x in self.X])
        self.len = self.lens.sum()

        self.tile_size = tile_size
        if self.tile_size is not None:
            self.tile_size = np.array(self.tile_size[-3:])
            first_elem = self.X[0][0]
            self.diffs = (np.array(first_elem.shape[-3:]) - self.tile_size) // 2
            self.diffs = [(d, -d if d else None) for d in self.diffs]
        assert self.len == np.array([len(y) for y in self.y]).sum()
        print(f"shapes X: {len(self.X), self.X[0].shape}, y: {len(self.y), self.y[0].shape}")

        if max_items is not None and max_items > 0:
            print(f"Only considering the first {max_items} items over {len(self.X)}.")
            # self.X = self.X[:max_items]
            # self.y = self.y[:max_items]
            self.len = max_items

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        i = 0
        channels_FC = [4, 5]
        channels_IMP_FC = [6, 7]
        while idx >= len(self.X[i]):
            idx -= len(self.X[i])
            i += 1
        x, y = self.X[i][idx], self.y[i][idx]

        # Normally all the cells that contain nans have been ignored

        if self.tile_size is not None:
            d = self.diffs
            x = x[..., d[0][0] : d[0][1], d[1][0] : d[1][1], d[2][0] : d[2][1]]
            y = y[..., d[0][0] : d[0][1], d[1][0] : d[1][1], d[2][0] : d[2][1]]

        if np.isnan(x).any() or np.isnan(y).any():
            if self.return_GP_FC_IMP_FC:
                return None, None, None, None
            return None, None
        x, y = torch.tensor(x), torch.tensor(y)
        if self.return_GP_FC_IMP_FC:
            return x, y, x[channels_FC], x[channels_IMP_FC]
        return x, y
