from Dataset import load_dataset, DatasetName, load_single_file
from variogram.VariogramAnalysis import VariogramAnalysis, timer

import argparse
from typing import Tuple, List
import numpy as np
import os


def load_from_npy(path: str):
    data = np.load(path, allow_pickle=True)
    bins = data.item().get("bins")
    bins_count = data.item().get("bins_count")
    return bins, bins_count


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin_res", nargs="+", type=List[float], help="defines the ranges of bins [lon, lat, time]")
    parser.add_argument("--total_bin_res", default=22, type=int, help="only used for 1D variogram")
    parser.add_argument("--chunk_size", default=1e6, type=float, help="what size chunk the computation is performed on")
    parser.add_argument("--detrended", default=True, type=bool, help="if the detrended data should be used")
    parser.add_argument("--cross_buoy_pairs_only", default=False, type=bool, help="read name m8")
    return parser


@timer
def main():
    # args =  parse().parse_args()
    dataset_name = DatasetName.AREA1
    data = load_dataset(dataset_name) # 300,000 pts
    # data = load_single_file(dataset_name, file_idx=0)  # 8,000 pts

    # Variogram calculation
    V = VariogramAnalysis(data)
    V.detrend(detrend_var="lat", num_bins=1)

    # define hyper-params
    bin_res = (0.05,0.05,1)
    detrended = True
    cross_buoy_pairs_only = False
    print("Begin computing variogram.")

    # compute variogram
    bins, bins_count = V.build_variogram_gen(bin_res, 2, chunk_size=int(1e6),\
        detrended=detrended, cross_buoy_pairs_only=cross_buoy_pairs_only)

    # save to numpy array
    file_name = f"variogram_{dataset_name.name}_{bin_res[0]}_{bin_res[1]}_{bin_res[2]}_{detrended}_{cross_buoy_pairs_only}.npy"
    file_path = os.path.join("/home/jonas/Documents/Thesis/OceanPlatformControl/data/drifter_data/variogram", file_name)
    data_to_save = {"bins":bins, "bins_count":bins}
    np.save(file_path, data_to_save)


if __name__ == "__main__":
    main()