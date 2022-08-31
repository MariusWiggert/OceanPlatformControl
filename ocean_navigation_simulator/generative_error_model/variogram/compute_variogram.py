from ocean_navigation_simulator.generative_error_model.Dataset import Dataset
from ocean_navigation_simulator.generative_error_model.variogram.utils import save_variogram_to_npy
from ocean_navigation_simulator.generative_error_model.utils import timer, setup_logger, load_config, get_path_to_project
from ocean_navigation_simulator.generative_error_model.variogram.Variogram import Variogram

import argparse
import datetime
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin_res", nargs="*", action="append", type=float, help="defines the ranges of bins [lon, lat, time]")
    parser.add_argument("--yaml_file_config", default="scenarios/generative_error_model/config_buoy_data.yaml", type=str)
    parser.add_argument("--is_3d", action="store_true", help="whether to compute a 3d or 2d variogram")
    parser.add_argument("--num_workers", default=2, type=int, help="for multiprocessing")
    parser.add_argument("--chunk_size", default=1e6, type=float, help="what size chunk the computation is performed on")
    parser.add_argument("--cross_buoy_pairs_only", action="store_true", help="read name m8")
    parser.add_argument("--units", default="km", type=str, help="choices: {'km', 'degrees'}")
    parser.add_argument("--dataset_type", default="forecast", type=str, help="{'forecast', 'hindcast', 'synthetic'}")
    parser.add_argument("--dataset_name", default="area1", type=str, help="Region the data is in.")
    parser.add_argument("--dataset_size", default="large", type=str, help="{large -> month, small -> single file}")
    parser.add_argument("--no_data_overlap", action="store_false", help="Should errors between two forecasts overlap in time")
    return parser


@timer
def main():
    args = parse().parse_args()

    # load config
    project_dir = get_path_to_project(os.getcwd())
    config = load_config("config_buoy_data.yaml")
    # setup logging
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_dir = os.path.join(project_dir, config["data_dir"], "logging")
    logger = setup_logger(log_dir, now_string)

    # load data
    dataset = Dataset(config["data_dir"], args.dataset_type, args.dataset_name)
    logger.info(f"Using {args.dataset_type.upper()} data.\n")
    if args.dataset_size == "large":
        data = dataset.load_dataset(overlap=args.no_data_overlap)
    else:
        data = dataset.load_single_file(file_idx=0)

    # initialize object + detrend
    v = Variogram(data)
    v.detrend(detrend_var="lat", num_bins=1)

    # log hyper-params
    logger.info("Begin computing variogram.\n")
    logger.info(f"""PARAMS:
        bin_res={args.bin_res[0]},
        is_3d={args.is_3d},
        num_workers={args.num_workers},
        chunk_size={int(args.chunk_size)},
        cross_buoy_pairs_only={args.cross_buoy_pairs_only},
        units={args.units},
        dataset_size={args.dataset_size},
        data_overlap={args.no_data_overlap}\n""")

    # compute variogram
    v.build_variogram_gen(args.bin_res[0], args.num_workers, chunk_size=int(args.chunk_size),
                          cross_buoy_pairs_only=args.cross_buoy_pairs_only, is_3d=args.is_3d, logger=logger)
    # save to .npy
    resolution = list(v.res_tuple)
    dim_name_map = {True: "3d", False: "2d"}
    file_name_p1 = f"{now_string}_{dim_name_map[args.is_3d]}_{args.dataset_type}_variogram_{args.dataset_name}_"
    file_name_p2 = f"{resolution}_{args.cross_buoy_pairs_only}_{args.no_data_overlap}.npy"
    file_name = file_name_p1 + file_name_p2
    file_path = os.path.join(project_dir, config["data_dir"], "variogram", file_name)
    save_variogram_to_npy(v, file_path)


if __name__ == "__main__":
    main()
