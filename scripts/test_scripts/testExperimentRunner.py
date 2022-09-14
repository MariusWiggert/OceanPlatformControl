import argparse
import csv
import datetime
import os
import sys
from _csv import writer
from datetime import datetime
from datetime import timezone

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from npy_append_array import NpyAppendArray
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch

from ocean_navigation_simulator.ocean_observer.ExperimentRunner import ExperimentRunner


def conditional_parameters(str_accepted: list[str], to_return, is_kernel_1: bool = True):
    return tune.sample_from(
        lambda s: to_return if (s.config.kernel if is_kernel_1 else s.config.kernel_2) in str_accepted else None)


# General search space
# Rational quadratic
# search_space = {
#     "filename_problems": "all_problems_3",
#     # product and sum are not supported yet
#     "kernel": "matern",  # tune.grid_search("matern"),
#     # "kernel": "expSineSquared",  # "matern"
#     "sigma_exp": tune.qrandn(2, 2, 0.0001),
#     # if matern or rbf
#     "scaling": conditional_parameters(["rbf", "matern"], {
#         "latitude": tune.loguniform(1e-2, 5),
#         "longitude": tune.loguniform(1e-2, 5),  # tune.loguniform(1, 1e6),
#         "time": tune.loguniform(7200, 43200)
#     }),
#
#     # if matern
#     "nu": conditional_parameters(["matern"], tune.choice([0.1, 0.5, 1.5, 2.5])),
#     # values not in [.5, 1.5, 2.5, inf] are 10x longer to compute
#
#     # if rational quadratic or expsinesquared(=periodic)
#     "length_scale": conditional_parameters(["RationalQuadratic", "ExpSineSquared"], tune.uniform(1, 100000)),
#     # "length_scale": tune.uniform(1, 100000),
#     "length_scale_bounds": "fixed",
#     # if rational quadratic
#     "alpha": conditional_parameters(["RationalQuadratic"], tune.loguniform(1e-5, 2.5)),
#     "alpha_bounds": conditional_parameters(["RationalQuadratic"], "fixed"),
#
#     # if expSineSquared
#     "periodicity": conditional_parameters(["ExpSineSquared"], tune.loguniform(0.01, 10)),
#     "periodicity_bounds": conditional_parameters(["ExpSineSquared"], "fixed"),
#
#     # Second kernel
#     "kernel_2": tune.choice(["RationalQuadratic", "ExpSineSquared", "rbf"]),
#     "sigma_exp_2": tune.qrandn(1, 1, 0.0001),
#     "scaling_2": conditional_parameters(["rbf", "matern"], {
#         "latitude": tune.loguniform(1e-2, 5),
#         "longitude": tune.loguniform(1e-2, 5),  # tune.loguniform(1, 1e6),
#         "time": tune.loguniform(7200, 43200)
#     }, is_kernel_1=False),
#     "length_scale_2": conditional_parameters(["RationalQuadratic", "ExpSineSquared"], tune.uniform(1, 100000),
#                                              is_kernel_1=False),
#     "length_scale_bounds_2": "fixed",
#     # if rational quadratic
#     "alpha_2": conditional_parameters(["RationalQuadratic"], tune.loguniform(1e-5, 2.5)),
#     "alpha_bounds_2": conditional_parameters(["RationalQuadratic"], "fixed"),
#     # if expSineSquared
#     "periodicity_2": conditional_parameters(["ExpSineSquared"], tune.loguniform(0.01, 10), is_kernel_1=False),
#     "periodicity_bounds_2": conditional_parameters(["ExpSineSquared"], "fixed", is_kernel_1=False),
# }


# Matern search space
search_space = {
    "filename_problems": "all_problems_3",
    # product and sum are not supported yet
    # "kernel": tune.choice([{"product": ("matern", "rbf")}]),
    "kernel": "matern",  # tune.grid_search(["matern", "rbf", "ExpSineSquared", "RationalQuadratic"]),
    "sigma_exp": tune.qrandn(1, 1, 0.0001),
    # if matern or rbf
    "scaling": {
        "latitude": tune.loguniform(1e-2, 5),
        "longitude": tune.loguniform(1e-2, 5),  # tune.loguniform(1, 1e6),
        "time": tune.loguniform(7200, 43200)},
    # "lon_scale": tune.loguniform(1, 1e6),
    # "time_scale": tune.loguniform(1, 1e6),
    # if matern
    "nu": tune.choice([1e-5, 1e-4, 0.001, 0.1, 0.5, 1.5]),
    # values not in [.5, 1.5, 2.5, inf] are 10x longer to compute

    # if rational quadratic or expsinesquared(=periodic)

    # "length_scale": conditional_parameters(["RationalQuadratic", "ExpSineSquared"], tune.uniform(1, 1e6)),
    "length_scale_bounds": "fixed",
    # if rational quadratic
    # "alpha": conditional_parameters(["RationalQuadratic"], tune.loguniform(1e-5, 2.5)),
    # "alpha_bounds": conditional_parameters(["RationalQuadratic"], "fixed"),

    # if expSineSquared
    # "periodicity": conditional_parameters(["ExpSineSquared"], tune.loguniform(0.01, 1000)),
    # "periodicity_bounds": conditional_parameters(["ExpSineSquared"], "fixed"),
    # "length_scale": conditional_parameters(["ExpSineSquared"], tune.loguniform(1, 1e6)),

    # not supported yet
    # if product
    # "1": {
    #     "lat_scale": tune.loguniform(1, 1e6),
    #     "lon_scale": tune.loguniform(1, 1e6),
    #     "time_scale": tune.loguniform(1, 1e6),
    #     "length_scale_bounds": "fixed",
    #     # if matern
    #     "nu": tune.choice([0.001, 0.1, 0.5, 1.5, 2.5, 5, 10])
    # },
    # "2": {
    #     "lat_scale": tune.loguniform(1, 1e6),
    #     "lon_scale": tune.loguniform(1, 1e6),
    #     "time_scale": tune.loguniform(1, 1e6),
    #     "length_scale_bounds": "fixed"
    # }
}


# Matern_bayes
# search_space = {
#     "num_threads": 32,
#     "filename_problems": "all_problems_3",
#     # product and sum are not supported yet
#     # "kernel": tune.choice([{"product": ("matern", "rbf")}]),
#     "kernel": "matern",  # tune.grid_search(["matern", "rbf", "ExpSineSquared", "RationalQuadratic"]),
#     "sigma_exp": tune.uniform(0.0001, 10),
#     # if matern or rbf
#     "scaling": {
#         "latitude": tune.uniform(1, 1e4),
#         "longitude": tune.uniform(1, 1e4),  # tune.loguniform(1, 1e6),
#         "time": tune.uniform(1000, 5000000)},
#     # "lon_scale": tune.loguniform(1, 1e6),
#     # "time_scale": tune.loguniform(1, 1e6),
#     # if matern
#     # "nu": tune.choice([1e-5, 1e-4, 0.001, 0.1, 0.5, 1.5]),
#     "nu": tune.uniform(0.01, 1.5),
#     "length_scale_bounds": "fixed",
#
# }


# import os
# import ray
#
# runtime_env = {"working_dir": os.getcwd()}
# print("runtime_env:", runtime_env)


def write_row_csv(path, row):
    # open the file in the write mode
    f = open(path, 'a')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(row)

    # close the file
    f.close()


def train(config_init):
    # Create a file for the log of all the results
    file_csv = os.path.join(os.path.dirname(os.getcwd()), "results.csv")
    print("path file:", file_csv)
    # os.chdir("/Users/fedosha/polybox/semester4/codebase/OceanPlatformControl/")
    os.chdir("/home/seaweed/test")
    if "num_threads" in config_init:
        torch.set_num_threads(config_init.pop("num_threads"))
    yaml_file_config = "./scenarios/ocean_observer/config_real_data_GP.yaml"
    config = {k: v for k, v in config_init.items() if v is not None and not k.endswith("_2")}
    config_2 = {k[:-2]: v for k, v in config_init.items() if v is not None and k.endswith("_2")}
    filename_problems = config.pop("filename_problems", None)
    # print("dir:", directory)
    with open(yaml_file_config) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        type_kernel = config.pop("kernel")
        type_kernel_2 = config_2.pop("kernel", None)
        sigma_exp_squared = abs(config.pop("sigma_exp"))
        sigma_exp_squared_2 = abs(config_2.pop("sigma_exp", 1))
        # Not supported yet
        # if type in ["product", "sum"]:
        #     config_yaml["observer"]["model"]["gaussian_process"]["kernel"] = {
        #         "type": type,
        #         "kernel_1": ,
        #         "kernel_2":
        #     }
        scaling = config.pop("scaling", None)
        scaling_2 = config_2.pop("scaling", None)
        config_yaml["observer"]["model"]["gaussian_process"]["kernel"] = \
            {"type": type_kernel,
             "scaling": scaling,
             "sigma_exp_squared": sigma_exp_squared,
             # if matern
             # "scaling": {"longitude": config["lon_scale"], "latitude": config["lat_scale"],
             #            "time": config["time_scale"], "nu": config["nu"]},
             # if rational quadratic
             # "parameters": {"alpha": config["alpha"], "length_scale_bounds": config["length_scale_bounds"],
             #               "alpha_bounds": "fixed", "length_scale": config["length_scale"]}
             # if expsinesquared:
             "parameters": config
             }  # | search_space
        if type_kernel_2 is not None:
            config_yaml["observer"]["model"]["gaussian_process"]["kernel_2"] = \
                {"type": type_kernel_2,
                 "scaling": scaling_2,
                 "sigma_exp_squared": sigma_exp_squared_2,
                 # if matern
                 # "scaling": {"longitude": config["lon_scale"], "latitude": config["lat_scale"],
                 #            "time": config["time_scale"], "nu": config["nu"]},
                 # if rational quadratic
                 # "parameters": {"alpha": config["alpha"], "length_scale_bounds": config["length_scale_bounds"],
                 #               "alpha_bounds": "fixed", "length_scale": config["length_scale"]}
                 # if expsinesquared:
                 "parameters_2": config_2
                 }

        print("kernel:", config_yaml["observer"]["model"]["gaussian_process"]["kernel"])
        print("kernel_2:", config_yaml["observer"]["model"]["gaussian_process"].get("kernel_2", None))

        exp = ExperimentRunner(config_yaml, filename_problems=filename_problems)
        results, results_per_h, merged, _ = exp.run_all_problems()

        # Save the results in the csv file
        merged_mean = {}
        for k in merged.keys():
            if k != "time":
                merged_mean["mean_" + str(k)] = np.array(merged[k]).mean()
        merged_mean |= merged
        merged_mean = {"kernel": str(config_yaml["observer"]["model"]["gaussian_process"]["kernel"]),
                       "kernel_2": str(
                           config_yaml["observer"]["model"]["gaussian_process"].get("kernel_2", None))} | merged_mean
        if not os.path.exists(file_csv):
            write_row_csv(file_csv, merged_mean.keys())
        write_row_csv(file_csv, merged_mean.values())

        # return {"avg": np.array([r["vme_improved"] for r in results]).mean()}
        tune.report(score=np.array([r["vme_improved"] for r in results]).mean())

    # variables = config["experiment_runner"]


def run_ray_tune_GP_grid(num_samples=500, bayes=False):
    # import ray
    # ray.init(dashboard_host="0.0.0.0", dashboard_port=6379)
    # res = tune.run(train, config=search_space, num_samples=num_samples)
    # return res.get_best_config(metric="r2_avg", mode="max")
    # return res.get_best_config(metric="avg", mode="min")
    if bayes:
        bayesopt = BayesOptSearch(metric="score", mode="min")
        tune.run(train, config=search_space, num_samples=num_samples, search_alg=bayesopt)
    else:
        res = tune.run(train, config=search_space, num_samples=num_samples)
        # return res.get_best_config(metric="r2_avg", mode="max")
        return res.get_best_config(metric="avg", mode="min")


def run_experiments_and_visualize_area(number_forecasts_in_days=20):
    # idle position
    # position = ((-86.20, 29.04, datetime.datetime(2022, 4, 19)), (-84, 28.04))
    p = -85.659, 26.15
    d = datetime.datetime(2022, 4, 2, 21, 30, tzinfo=datetime.timezone.utc)
    position = ((*p, d), (-90, 30))  # (p[0] + 2, p[1] + 2))
    exp = ExperimentRunner("config_test_GP", filename_problems="all_problems_3", position=position)
    # x, y, t = [-90, -80], [24, 30], [d, d + datetime.timedelta(days=1, hours=1)]
    x, y, t = [-88, -82], [25, 29], [d, d + datetime.timedelta(days=1, hours=1)]

    exp.visualize_area(x, y, t, number_days_forecasts=number_forecasts_in_days)


def run_experiments_and_visualize_noise(number_forecasts=30):
    day = datetime.datetime(2022, 4, 5, 12, 00, tzinfo=datetime.timezone.utc)
    center_x, center_y = -90, 24
    rx, ry = 5, 4
    x = [center_x - rx, center_x + rx]
    y = [center_y - ry, center_y + ry]
    exp = ExperimentRunner("config_test_GP", filename_problems="all_problems_3",
                           position=[
                               (center_x, center_y, day),
                               (center_x, center_y + 3)],
                           dict_field_yaml_to_update={"radius_area_around_platform": 5})
    # results, results_per_h, merged, list_dates_when_new_files = exp.visualize_all_noise(x, y)
    exp.visualize_all_noise(x, y, number_forecasts=number_forecasts)
    print("noise")


def __add_line_to_csv(to_add, folder_destination, type: str):
    with open(folder_destination + f'{type}.csv', 'a+', newline='') as f_object:
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(to_add)
        # Close the file object
        f_object.close()


def run_experiments_and_collect_tiles(output_folder: str, filename_problems,
                                      folder_destination="./data_NN_DA/export/"):
    # todo: set 24 as field in config + add parameter for config
    print(f"generating output into folder: {output_folder}")
    exp = ExperimentRunner("config_GP_for_NN", filename_problems=filename_problems,
                           folder_problems="data_NN_DA/", folder_config_file="data_NN_DA/")

    results = []
    i = 0
    k = 0
    while exp.has_next_problem():
        try:
            k += 1
            print(f"starting problem {k}")
            # results.append(np.array(exp.run_next_problem(get_inputs_and_outputs=True)))
            array_fc_hc, measurement_locations, errors = exp.run_next_problem(get_inputs_and_outputs=True)
            __export_results_to_file(array_fc_hc, folder_destination)
            __add_line_to_csv(measurement_locations, folder_destination, "measurement")
            __add_line_to_csv(errors, folder_destination, "error")
            # size = sys.getsizeof(results[0]) / 1000000
            # print(f"Number results:{len(results)}, size: {len(results) * size} MB")
            # if size > max_mega_per_file:
            #     __export_list_to_file(results, folder_destination, i)
            #     i += 1
            #     results = []
        except ValueError:
            print(f"error with problem: {len(results)}")

    print("over")


def __export_results_to_file(res: list, path_dir):
    for j, s in enumerate(['x', 'y']):
        isExist = os.path.exists(path_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path_dir)
            print(f"The new directory is created: {path_dir}!")
        with NpyAppendArray(path_dir + f"data_{s}.npy") as npaa:
            npaa.append(np.ascontiguousarray(res[j].astype('float64')))


def run_experiments_and_plot(max_number_problems_to_run=None):
    """
    Run an experiment
    """
    # np.random.seed(0)
    exp = ExperimentRunner("config_test_GP", filename_problems="all_problems_3")
    results, results_per_h, merged, list_dates_when_new_files = exp.run_all_problems(
        max_number_problems_to_run=max_number_problems_to_run)
    print("final results:", results)

    '''
    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------
    PLOTTING
    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------
    '''

    # plot to see error
    metric_to_plot = "vme"
    initial, improved = metric_to_plot + "_initial", metric_to_plot + "_improved"
    problem_1 = results[-1]
    X = [datetime.fromtimestamp(t, tz=timezone.utc) for t in problem_1["time"]]
    y1 = problem_1[initial]
    y2 = problem_1[improved]
    y1_mean = np.array([r[initial] for r in results]).mean(axis=0)
    y2_mean = np.array([r[initial] for r in results]).mean(axis=0)

    plt.plot(X, y1, color='r', label=initial)
    plt.plot(X, y2, color='g', label=improved)

    plt.plot(X, y1_mean, color="b", label=initial + "_mean")
    plt.plot(X, y2_mean, "y--", label=improved + "_mean")

    for t in list_dates_when_new_files:
        plt.axvline(x=t)
    plt.xlabel("time")
    plt.ylabel("Average " + metric_to_plot)
    plt.legend()

    mean_improved_per_hour = np.array([r[improved + "_per_h"].mean(axis=0) for r in results_per_h]).mean(axis=0)
    mean_initial_per_hour = np.array([r[initial + "_per_h"].mean(axis=0) for r in results_per_h]).mean(axis=0)
    plt.figure()
    plt.plot(range(len(mean_improved_per_hour)), mean_improved_per_hour, color='g', label=(improved + "_per_h"))
    plt.plot(range(len(mean_initial_per_hour)), mean_initial_per_hour, color='r', label=(initial + "_per_h"))
    plt.xlabel("hour forecast")
    plt.ylabel("Average " + metric_to_plot)
    plt.legend()

    plt.figure()
    x = np.array([r["mean_magnitude_forecast"] for r in results]).flatten()
    y = np.array([r["rmse_initial"] for r in results]).flatten()
    y2 = np.array([r["rmse_improved"] for r in results]).flatten()
    plt.scatter(x, y, c="red", label="initial_forecast")
    plt.scatter(x, y2, c="blue", label="improved_forecast", alpha=0.5)
    plt.title("Magnitude forecast vs RMSE")
    plt.xlabel("magnitude forecast")
    plt.ylabel("RMSE")
    plt.legend()

    plt.figure()
    plt.scatter(x, y - y2, c="green", label="initial-improved rmse errors")
    plt.xlabel("magnitude forecast")
    plt.ylabel("RMSE difference")
    plt.legend()

    print("over")


if __name__ == "__main__":
    print("arguments: ", sys.argv)
    if not {"-R", "--remote"}.isdisjoint(sys.argv):
        run_ray_tune_GP_grid(num_samples=5000)
    elif not {"-V", "--visualize"}.isdisjoint(sys.argv):
        run_experiments_and_visualize_area(1)
    elif not {"-N", "--noise"}.isdisjoint(sys.argv):
        run_experiments_and_visualize_noise(number_forecasts=20)
    elif not {"-T", "--collect-tiles"}.isdisjoint(sys.argv):
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('-T', action='store_true', help='collect tiles')
        parser.add_argument('-f', type=str, help='file name')
        args = parser.parse_args()
        run_experiments_and_collect_tiles(output_folder="./data_NN_DA/", filename_problems=args.f)
    else:
        run_experiments_and_plot(max_number_problems_to_run=None)

# %%
