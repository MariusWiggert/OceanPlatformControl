from ocean_navigation_simulator.generative_error_model.Dataset import load_dataset, DatasetName, load_single_file
from ocean_navigation_simulator.generative_error_model.utils import timer, save_variogram_to_npy
from ocean_navigation_simulator.generative_error_model.variogram.Variogram import Variogram

import argparse
import numpy as np
import datetime
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin_res", nargs=3, action="append", type=float, help="defines the ranges of bins [lon, lat, time]")
    parser.add_argument("--num_workers", default=2, type=int, help="for multiprocessing")
    parser.add_argument("--chunk_size", default=1e6, type=float, help="what size chunk the computation is performed on")
    parser.add_argument("--detrended", default=True, type=bool, help="if the detrended data should be used")
    parser.add_argument("--cross_buoy_pairs_only", default=False, type=bool, help="read name m8")
    parser.add_argument("--dataset_size", default="large", type=str, help="{large -> month, small -> single file}")
    return parser


@timer
def main():
    args =  parse().parse_args()

    # load data
    dataset_name = DatasetName.AREA1
    if args.dataset_size == "large":
        data = load_dataset(dataset_name) # 300,000 pts
    else:
        data = load_single_file(dataset_name, file_idx=0)  # 8,000 pts

    # initialize object + detrend
    V = Variogram(data)
    V.detrend(detrend_var="lat", num_bins=1)

    # define hyper-params
    bin_res = args.bin_res[0]
    num_workers = args.num_workers
    detrended = args.detrended
    cross_buoy_pairs_only = args.cross_buoy_pairs_only
    print("\nBegin computing variogram.")
    print(f"""PARAMS:
        bin_res={bin_res},d
        num_workers={args.num_workers},
        detrended={args.detrended},
        cross_buoy_pairs_only={args.cross_buoy_pairs_only}
        dataset_size={args.dataset_size}\n""")


    # compute variogram
    bins, bins_count = V.build_variogram_gen(bin_res, num_workers, chunk_size=int(1e6),\
        detrended=detrended, cross_buoy_pairs_only=cross_buoy_pairs_only)

    # save to numpy array
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    file_name = f"{now_string}_variogram_{dataset_name.name}_{bin_res[0]}_{bin_res[1]}_{bin_res[2]}_{detrended}_{cross_buoy_pairs_only}.npy"
    file_path = os.path.join("/home/jonas/Documents/Thesis/OceanPlatformControl/data/drifter_data/variogram", file_name)
    save_variogram_to_npy(V, file_path)


if __name__ == "__main__":
    main()