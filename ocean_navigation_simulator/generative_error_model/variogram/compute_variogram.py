from ocean_navigation_simulator.generative_error_model.Dataset import Dataset
from ocean_navigation_simulator.generative_error_model.variogram.utils import save_variogram_to_npy
from ocean_navigation_simulator.generative_error_model.utils import timer, setup_logger, load_config
from ocean_navigation_simulator.generative_error_model.variogram.Variogram import Variogram

import argparse
import datetime
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin_res", nargs=3, action="append", type=float, help="defines the ranges of bins [lon, lat, time]")
    parser.add_argument("--num_workers", default=2, type=int, help="for multiprocessing")
    parser.add_argument("--chunk_size", default=1e6, type=float, help="what size chunk the computation is performed on")
    parser.add_argument("--detrended", default=True, type=bool, help="if the detrended data should be used")
    parser.add_argument("--cross_buoy_pairs_only", default=False, type=bool, help="read name m8")
    parser.add_argument("--units", default="km", type=str, help="choices: {'km', 'degrees'}")
    parser.add_argument("--dataset_name", default="area1", type=str, help="Region the data is in.")
    parser.add_argument("--dataset_size", default="large", type=str, help="{large -> month, small -> single file}")
    parser.add_argument("--data_overlap", default=True, type=bool, help="Should errors between two forecasts overlap in time")
    return parser


def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False


@timer
def main():
    args = parse().parse_args()
    args.detrended = t_or_f(args.detrended)
    args.cross_buoy_pairs_only = t_or_f(args.cross_buoy_pairs_only)
    args.data_overlap = t_or_f(args.data_overlap)

    # setup logging
    config = load_config()
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir = os.path.join(config["data_dir"], "logging")
    logger = setup_logger(log_dir, now_string)

    # load data
    dataset = Dataset(args.dataset_name)
    if args.dataset_size == "large":
        data = dataset.load_dataset(args.data_overlap)  # ~300,000 pts
    else:
        data = dataset.load_single_file(file_idx=0)  # ~8,000 pts

    # initialize object + detrend
    V = Variogram(data)
    V.detrend(detrend_var="lat", num_bins=1)

    # log hyper-params
    logger.info("Begin computing variogram.\n")
    logger.info(f"""PARAMS:
        bin_res={args.bin_res[0]},
        num_workers={args.num_workers},
        detrended={args.detrended},
        cross_buoy_pairs_only={args.cross_buoy_pairs_only},
        units={args.units},
        dataset_size={args.dataset_size},
        data_overlap={args.data_overlap}\n""")

    # compute variogram
    bins, bins_count = V.build_variogram_gen(args.bin_res[0], args.num_workers, chunk_size=int(1e6),
        detrended=args.detrended, cross_buoy_pairs_only=args.cross_buoy_pairs_only, logger=logger)

    # save to .npy
    file_name_part1 = f"{now_string}_variogram_{args.dataset_name}_{args.bin_res[0][0]}_{args.bin_res[0][1]}_"
    file_name_part2 = f"{args.bin_res[0][2]}_{args.detrended}_{args.cross_buoy_pairs_only}_{args.data_overlap}.npy"
    file_name = file_name_part1 + file_name_part2
    file_path = os.path.join(os.path.join(config["data_dir"], "variogram"), file_name)
    save_variogram_to_npy(V, file_path)


if __name__ == "__main__":
    main()
