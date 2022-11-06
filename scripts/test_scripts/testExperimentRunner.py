import argparse
import csv
import datetime
import gc
import logging
import math
import os
import pickle
import sys
from _csv import writer

import numpy as np
import ray
import torch
import yaml
from matplotlib import pyplot as plt
from npy_append_array import NpyAppendArray
from ray import tune
from ray.air import session

# from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.search.bayesopt import BayesOptSearch

from ocean_navigation_simulator.ocean_observer.ExperimentRunner import ExperimentRunner


def conditional_parameters(str_accepted: list[str], to_return, is_kernel_1: bool = True):
    return tune.sample_from(
        lambda s: to_return
        if (s.config.kernel if is_kernel_1 else s.config.kernel_2) in str_accepted
        else None
    )


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
current_dir = os.getcwd()
search_space_bayes = {
    "sigma_exp": (0.00001, 10),
    "latitude": (1e-2, 5),
    "longitude": (1e-2, 5),  # tune.loguniform(1e-2, 5),  # tune.loguniform(1, 1e6),
    "time": (7200, 43200),
}

search_space_without_bayes = {
    "sigma_exp": 0.5,
    "latitude": 0.22,
    "longitude": 0.44,  # tune.loguniform(1e-2, 5),  # tune.loguniform(1, 1e6),
    "time": 5700,
}

search_space = {
    "folder_problems": "ablation_study/problems/",
    "folder_config": "ablation_study/configs_GP/",
    # "max_problems": 120,
    # product and sum are not supported yet
    # "kernel": tune.choice([{"product": ("matern", "rbf")}]),
    "kernel": "matern",  # tune.grid_search(["matern", "rbf", "ExpSineSquared", "RationalQuadratic"]),
    "sigma_exp": (0.00001, 10),  # tune.qrandn(1, 1, 0.0001),
    # if matern or rbf
    "scaling": {
        "latitude": (1e-2, 5),  # tune.loguniform(1e-2, 5),
        "longitude": (1e-2, 5),  # tune.loguniform(1e-2, 5),  # tune.loguniform(1, 1e6),
        "time": (7200, 43200),
    },  # tune.loguniform(7200, 43200)},
    # "lon_scale": tune.loguniform(1, 1e6),
    # "time_scale": tune.loguniform(1, 1e6),
    # if matern
    "nu": 1.5,  # tune.choice([1e-5, 1e-4, 0.001, 0.1, 0.5, 1.5]),
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
    f = open(path, "a")
    print(f"Write row to csv: {path}")
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(row)

    # close the file
    f.close()


def train(config_from_bayes=None):
    gc.collect()
    # Create a file for the log of all the results
    print("CONFIG INIT:", config_from_bayes)
    # Copy the bayes parameter to the full dict
    full_dict = search_space.copy()
    full_dict["sigma_exp"] = config_from_bayes["sigma_exp"]
    full_dict["scaling"]["longitude"] = config_from_bayes["longitude"]
    full_dict["scaling"]["latitude"] = config_from_bayes["latitude"]
    full_dict["scaling"]["time"] = config_from_bayes["time"]
    folder_yamls = full_dict.pop("folder_config")
    filename = full_dict.pop("filename_config")
    filename_problems = full_dict.pop("filename_problems", None)
    max_problems = full_dict.pop("max_problems", None)
    folder_problems = full_dict.pop("folder_problems", "scenarios/ocean_observer/")
    folder_config = full_dict.pop("folder_config", "scenarios/ocean_observer/")
    file_csv = os.path.join(
        "./ablation_study/results_grids/",
        f"results_{filename}_{filename_problems}.csv",
    )
    print("path file:", file_csv)

    os.chdir(current_dir)
    # os.chdir("/home/killian2k/seaweed/OceanPlatformControl")
    if full_dict is not None:
        if "num_threads" in full_dict:
            torch.set_num_threads(full_dict.pop("num_threads"))
        yaml_file_config = f"{folder_yamls}{filename}.yaml"
        # dict for the two kernels
        config = {k: v for k, v in full_dict.items() if v is not None and not k.endswith("_2")}
        config_2 = {k[:-2]: v for k, v in full_dict.items() if v is not None and k.endswith("_2")}

    # print("dir:", directory)
    with open(yaml_file_config) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        if full_dict is not None:
            type_kernel = config.pop("kernel")
            type_kernel_2 = config_2.pop("kernel", None)
            # sigma_exp_squared = abs(config.pop("sigma_exp"))
            sigma_exp_squared = config.pop("sigma_exp")
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
        config_yaml["observer"]["model"]["gaussian_process"]["kernel"] = {
            "type": type_kernel,
            "scaling": scaling,
            "sigma_exp_squared": sigma_exp_squared,
            # if matern
            # "scaling": {"longitude": config["lon_scale"], "latitude": config["lat_scale"],
            #            "time": config["time_scale"], "nu": config["nu"]},
            # if rational quadratic
            # "parameters": {"alpha": config["alpha"], "length_scale_bounds": config["length_scale_bounds"],
            #               "alpha_bounds": "fixed", "length_scale": config["length_scale"]}
            # if expsinesquared:
            "parameters": config,
        }  # | search_space
        if type_kernel_2 is not None:
            config_yaml["observer"]["model"]["gaussian_process"]["kernel_2"] = {
                "type": type_kernel_2,
                "scaling": scaling_2,
                "sigma_exp_squared": sigma_exp_squared_2,
                # if matern
                # "scaling": {"longitude": config["lon_scale"], "latitude": config["lat_scale"],
                #            "time": config["time_scale"], "nu": config["nu"]},
                # if rational quadratic
                # "parameters": {"alpha": config["alpha"], "length_scale_bounds": config["length_scale_bounds"],
                #               "alpha_bounds": "fixed", "length_scale": config["length_scale"]}
                # if expsinesquared:
                "parameters_2": config_2,
            }

        print("kernel:", config_yaml["observer"]["model"]["gaussian_process"]["kernel"])
        print(
            "kernel_2:",
            config_yaml["observer"]["model"]["gaussian_process"].get("kernel_2", None),
        )

        exp = ExperimentRunner(
            config_yaml,
            filename_problems=filename_problems,
            folder_problems=folder_problems,
            folder_config_file=folder_config,
        )
        results, results_per_h, merged, _ = exp.run_all_problems(
            max_number_problems_to_run=max_problems
        )
        # Save the results in the csv file
        merged_mean = {}
        for k in merged.keys():
            if k != "time":
                merged_mean["mean_" + str(k)] = np.nanmean(merged[k])
        merged_mean |= merged
        merged_mean = {
            "kernel": str(config_yaml["observer"]["model"]["gaussian_process"]["kernel"]),
            "kernel_2": str(
                config_yaml["observer"]["model"]["gaussian_process"].get("kernel_2", None)
            ),
        } | merged_mean
        if not os.path.exists(file_csv):
            write_row_csv(file_csv, merged_mean.keys())
        write_row_csv(file_csv, merged_mean.values())

        # print({"avg": np.array([r["vme_improved"] for r in results]).mean()})
        # tune.report(score=np.array([r["vme_improved"] for r in results]).mean())
        session.report(
            {
                "r2": np.nanmean(np.hstack([r["r2"] for r in results])),
                "vme_improved": np.nanmean(np.hstack([r["vme_improved"] for r in results])),
                "rmse_improved": np.nanmean(np.hstack([r["rmse_improved"] for r in results])),
                "ratio_per_tile": np.nanmean(np.hstack([r["ratio_per_tile"] for r in results])),
            }
        )

    # variables = config["experiment_runner"]
    del merged_mean
    del exp
    del results
    del results_per_h
    del full_dict
    gc.collect()


def train_without_tune(
    filename_problem,
    filename_config,
    num_samples=500,
    problems_per_sample=120,
    research_state=1,
    bayes=False,
    random_search_space=40,
):
    search_space["max_problems"] = problems_per_sample
    search_space["filename_problems"] = filename_problem
    search_space["filename_config"] = filename_config
    train(search_space_without_bayes)


def run_ray_tune_GP_grid(
    filename_problem,
    filename_config,
    num_samples=500,
    problems_per_sample=120,
    research_state=1,
    bayes=False,
    random_search_space=40,
    load=True,
):
    path_file_save_checkpoint = f"ablation_study/checkpoints/{filename_config}.pkl"
    None
    if bayes:
        search_space["max_problems"] = problems_per_sample
        search_space["filename_problems"] = filename_problem
        search_space["filename_config"] = filename_config
        ray.init(_memory=2000 * 1024 * 1024, object_store_memory=200 * 1024 * 1024)
        bayesopt = BayesOptSearch(
            space=search_space_bayes,
            metric="r2",
            mode="max",
            random_state=research_state,
            random_search_steps=random_search_space,
        )
        if load:
            bayesopt.restore(path_file_save_checkpoint)
            print("loaded bayes config: ", path_file_save_checkpoint)
        print("Creating the tuner.")
        tuner = tune.Tuner(
            train,
            tune_config=tune.TuneConfig(search_alg=bayesopt, num_samples=num_samples),
        )
        try:
            print("fitting")
            res = tuner.fit()
        finally:
            bayesopt.save(path_file_save_checkpoint)

    else:
        res = tune.run(train, config=search_space, num_samples=num_samples)
        # return res.get_best_config(metric="r2_avg", mode="max")

    return res


def run_experiments_and_visualize_area(
    number_forecasts_in_days=20,
    yaml_file_config="config_test_GP",
    folder_config_file=None,
    use_NN=False,
):
    # idle position
    # position = ((-86.20, 29.04, datetime.datetime(2022, 4, 19)), (-84, 28.04))
    p = -85.659, 27.15
    d = datetime.datetime(2022, 4, 10, 13, 30, tzinfo=datetime.timezone.utc)
    position = ((*p, d), (-90, 30))
    if folder_config_file is not None:
        exp = ExperimentRunner(
            yaml_file_config,
            filename_problems="all_problems_3",
            position=position,
            folder_config_file=folder_config_file,
        )
    else:
        exp = ExperimentRunner(
            yaml_file_config, filename_problems="all_problems_3", position=position
        )
    # x, y, t = [-90, -80], [24, 30], [d, d + datetime.timedelta(days=1, hours=1)]
    # x, y, t = [-88, -82], [25, 29], [d, d + datetime.timedelta(days=1, hours=1)]
    x, y, t = (
        [-88 + 1 / 12, -82],
        [24, 30],
        [d, d + datetime.timedelta(days=1, hours=1)],
    )

    exp.visualize_area(x, y, t, number_days_forecasts=number_forecasts_in_days, use_NN=use_NN)


def run_experiments_and_visualize_noise(number_forecasts=30):
    day = datetime.datetime(2022, 4, 5, 12, 00, tzinfo=datetime.timezone.utc)
    center_x, center_y = -90, 24
    rx, ry = 5, 4
    x = [center_x - rx, center_x + rx]
    y = [center_y - ry, center_y + ry]
    exp = ExperimentRunner(
        "config_test_GP",
        filename_problems="all_problems_3",
        position=[(center_x, center_y, day), (center_x, center_y + 3)],
        dict_field_yaml_to_update={"radius_area_around_platform": 5},
    )
    # results, results_per_h, merged, list_dates_when_new_files = exp.visualize_all_noise(x, y)
    exp.visualize_all_noise(x, y, number_forecasts=number_forecasts)
    print("noise")


def __add_line_to_csv(to_add, folder_destination, file_name: str):
    with open(folder_destination + f"{file_name}.csv", "a+", newline="") as f_object:
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(to_add)
        # Close the file object
        f_object.close()


def run_experiments_and_collect_tiles(
    output_folder: str,
    filename_problems,
    yaml_config_GP="config_GP_for_NN_validation",
    folder_problems="data_NN_DA/",
    folder_config_file="data_NN_DA/",
    filename_output_X_y: str = "problems_GP_output",
    number_max_problems=-1,
):
    # todo: set 24 as field in config + add parameter for config
    print(f"generating output into folder: {output_folder}")
    exp = ExperimentRunner(
        yaml_config_GP,
        filename_problems=filename_problems,
        folder_problems=folder_problems,
        folder_config_file=folder_config_file,
    )
    if number_max_problems < 0:
        number_max_problems = math.inf
    k = 0
    t = datetime.datetime.now()
    while exp.has_next_problem() and number_max_problems - k > 0:
        try:
            k += 1
            delt = datetime.datetime.now() - t
            print(
                f"starting problem {k}, last problem time: {delt.seconds // 60}m{delt.seconds % 60}s"
            )
            t = datetime.datetime.now()
            # results.append(np.array(exp.run_next_problem(get_inputs_and_outputs=True)))
            array_fc_hc, measurement_locations, errors = exp.run_next_problem(
                get_inputs_and_outputs=True
            )
            array_fc_hc = [np.float32(arr.swapaxes(0, 1)) for arr in array_fc_hc]
            __export_results_to_file(array_fc_hc, output_folder, filename_output_X_y)
            __add_line_to_csv(
                measurement_locations,
                output_folder,
                f"measurement_{filename_output_X_y}",
            )
            __add_line_to_csv(errors, output_folder, f"error_{filename_output_X_y}")
            # size = sys.getsizeof(results[0]) / 1000000
            # print(f"Number results:{len(results)}, size: {len(results) * size} MB")
            # if size > max_mega_per_file:
            #     __export_list_to_file(results, folder_destination, i)
            #     i += 1
            #     results = []
        except ValueError as e:
            print(f"error with problem: {k}\n{e}")

    print("over")


def __export_results_to_file(res: list, path_dir, filename: str):
    for j, s in enumerate(["x", "y"]):
        isExist = os.path.exists(path_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path_dir)
            print(f"The new directory is created: {path_dir}!")
        with NpyAppendArray(path_dir + f"{filename}_{s}.npy") as npaa:
            npaa.append(np.ascontiguousarray(res[j].astype("float64")))


def run_experiments_on_kernel():
    # yaml.load("./config_GP_for_NN.yaml")
    # train(config_init=)
    exp = ExperimentRunner(
        "config_GP_for_NN",
        filename_problems="4000_problems_1",
        folder_config_file="data_NN_DA/",
        folder_problems="data_NN_DA/",
    )
    # results, results_per_h, merged, list_dates_when_new_files = exp.visualize_all_noise(x, y)
    exp.run_all_problems(max_number_problems_to_run=12)


def run_experiments_and_plot(
    max_number_problems_to_run=None,
    plot_error_3d=False,
    name_config_yaml_file="config_test_GP",
    filename_problems="all_problems_3",
    folder_problems="scenarios/ocean_observer/",
    folder_config_file="scenarios/ocean_observer/",
):
    """
    Run an experiment
    """
    # np.random.seed(0)
    exp = ExperimentRunner(
        name_config_yaml_file,
        filename_problems=filename_problems,
        folder_problems=folder_problems,
        folder_config_file=folder_config_file,
    )
    all_results = exp.run_all_problems(
        max_number_problems_to_run=max_number_problems_to_run,
        compute_for_all_radius_and_lag=plot_error_3d,
    )
    if plot_error_3d:
        (
            results,
            results_per_h,
            merged,
            list_dates_when_new_files,
            results_grids,
        ) = all_results
        path_export = (
            "ablation_study/export_all_results_validation_set/" + name_config_yaml_file + "_export_"
        )
        print(f"exporting all the objects to: {path_export}")
        for i, obj in enumerate(
            [results, results_per_h, merged, list_dates_when_new_files, results_grids]
        ):
            filename = path_export + f"{i}.pickle"

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as handle:
                pickle.dump(results_grids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        results, results_per_h, merged, list_dates_when_new_files = all_results
    print("final results:", results)

    """
    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------
    PLOTTING
    -------------------------------------------------------------------------------
    -------------------------------------------------------------------------------
    """

    if plot_error_3d:
        # print the 3d plots
        for key in results_grids.keys():
            if key.startswith("r2"):
                to_plot = np.array(results_grids[key]).mean(axis=0)
                if key.endswith("_all_lags_and_radius"):
                    legend = "All lags and radius merged"
                    name = key[: -len("_all_lags_and_radius")]
                else:
                    legend = "each lag and radius separated"
                    name = key[: -len("per_lag_and_radius")]
                name = name.replace("_", " ")
                hf = plt.figure()
                plt.title(name + " - " + legend)

                ha = hf.add_subplot(111, projection="3d")
                ha.set_xlabel("lag")
                ha.set_ylabel("radius")
                ha.set_zlabel(name)

                X, Y = np.meshgrid(
                    range(to_plot.shape[0]), range(to_plot.shape[1])
                )  # `plot_surface` expects `x` and `y` data to be 2D
                ha.plot_surface(X.T, Y.T, to_plot)

                plt.show()

    # plot to see error
    metric_to_plot = "vme"
    initial, improved = metric_to_plot + "_initial", metric_to_plot + "_improved"
    problem_1 = results[-1]
    X = [datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc) for t in problem_1["time"]]
    y1 = problem_1[initial]
    y2 = problem_1[improved]
    y1_mean = np.array([r[initial] for r in results]).mean(axis=0)
    y2_mean = np.array([r[initial] for r in results]).mean(axis=0)

    plt.plot(X, y1, color="r", label=initial)
    plt.plot(X, y2, color="g", label=improved)

    plt.plot(X, y1_mean, color="b", label=initial + "_mean")
    plt.plot(X, y2_mean, "y--", label=improved + "_mean")

    for t in list_dates_when_new_files:
        plt.axvline(x=t)
    plt.xlabel("time")
    plt.ylabel("Average " + metric_to_plot)
    plt.legend()

    mean_improved_per_hour = np.array(
        [r[improved + "_per_h"].mean(axis=0) for r in results_per_h]
    ).mean(axis=0)
    mean_initial_per_hour = np.array(
        [r[initial + "_per_h"].mean(axis=0) for r in results_per_h]
    ).mean(axis=0)
    plt.figure()
    plt.plot(
        range(len(mean_improved_per_hour)),
        mean_improved_per_hour,
        color="g",
        label=(improved + "_per_h"),
    )
    plt.plot(
        range(len(mean_initial_per_hour)),
        mean_initial_per_hour,
        color="r",
        label=(initial + "_per_h"),
    )
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
    parser = argparse.ArgumentParser(description="Process the arguments.")
    if not {"-R", "--run-grid"}.isdisjoint(sys.argv):
        file_log = "ablation_study/results_grids/logs_best_models.log"

        parser.add_argument("-R", "--run-grid", action="store_true", help="run the grid.")
        parser.add_argument("--num-samples", type=int)
        parser.add_argument("--problems-per-sample", type=int)
        parser.add_argument("--research-state", type=int, default=1)
        parser.add_argument("--random-search-space", type=int)
        parser.add_argument("--filename-problem", type=str)
        parser.add_argument("--filename-config", type=str)
        parser.add_argument("--load", default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument("--results", default=False, action=argparse.BooleanOptionalAction)
        args = parser.parse_args()

        print("Computing 1 run and getting the results: ", args.results)
        M = 1 if args.results else 20
        for i in range(M):
            print(f"run ray iteration {i + 1}/{M}")
            results = run_ray_tune_GP_grid(
                args.filename_problem,
                args.filename_config,
                num_samples=args.num_samples,
                problems_per_sample=args.problems_per_sample,
                research_state=args.research_state,
                bayes=True,
                random_search_space=args.random_search_space,
                load=(args.load if i == 0 else True),
            )
            if i + 1 == M:
                print(
                    f"the best parameters found are: {results.get_best_result(metric='r2', mode='max')}"
                )
                print(
                    f"the best parameters found are: {results.get_best_result(metric='r2', mode='max').config}"
                )
                logging.basicConfig(
                    filename=file_log, format="%(asctime)s %(message)s", filemode="w"
                )
                logger = logging.getLogger()
                logger.info(
                    f"the best parameters found are: {results.get_best_result(metric='r2', mode='max').config}"
                )
            else:
                ray.shutdown()
    elif not {"-D", "--debug"}.isdisjoint(sys.argv):
        parser.add_argument("-D", "--debug", action="store_true", help="run the grid.")
        parser.add_argument("--num-samples", type=int)
        parser.add_argument("--problems-per-sample", type=int)
        parser.add_argument("--research-state", type=int, default=1)
        parser.add_argument("--random-search-space", type=int)
        parser.add_argument("--filename-problem", type=str)
        parser.add_argument("--filename-config", type=str)
        args = parser.parse_args()
        train_without_tune(
            args.filename_problem,
            args.filename_config,
            num_samples=args.num_samples,
            problems_per_sample=args.problems_per_sample,
            research_state=args.research_state,
            bayes=True,
            random_search_space=args.random_search_space,
        )
    elif not {"-V", "--visualize"}.isdisjoint(sys.argv):
        run_experiments_and_visualize_area(1)
    elif not {"-N", "--noise"}.isdisjoint(sys.argv):
        run_experiments_and_visualize_noise(number_forecasts=20)
    elif not {"-T", "--collect-tiles"}.isdisjoint(sys.argv):
        parser.add_argument("-T", "--collect-tiles", action="store_true", help="collect tiles")
        parser.add_argument("-f", type=str, help="file name")
        parser.add_argument("-folder-destination", type=str)
        parser.add_argument("--config-file", type=str, help="file name")
        parser.add_argument(
            "--config-folder", type=str, help="folder where the yaml file is located"
        )
        parser.add_argument(
            "--problem-folder",
            type=str,
            help="folder where the problem file is located",
        )
        parser.add_argument("--filename-destination", type=str, help="filename destination")
        parser.add_argument("--max-problems", type=int, default=-1)
        args = parser.parse_args()
        run_experiments_and_collect_tiles(
            output_folder=args.folder_destination,
            filename_problems=args.f,
            yaml_config_GP=args.config_file,
            folder_config_file=args.config_folder,
            folder_problems=args.problem_folder,
            number_max_problems=args.max_problems,
            filename_output_X_y=args.filename_destination,
        )
    elif not {"--visualize-results-NN"}.isdisjoint(sys.argv):
        parser.add_argument(
            "--visualize-results-NN",
            action="store_true",
            help="visualize results of the NN.",
        )
        parser.add_argument("--config-file", type=str, help="file name")
        parser.add_argument(
            "--config-folder", type=str, help="folder where the yaml file is located"
        )
        parser.add_argument("--number-days", type=int, default=5)
        args = parser.parse_args()
        run_experiments_and_visualize_area(
            number_forecasts_in_days=args.number_days,
            yaml_file_config=args.config_file,
            folder_config_file=args.config_folder,
            use_NN=True,
        )
    elif not {"-VR", "--vanilla-run"}.isdisjoint(sys.argv):
        run_experiments_on_kernel()
    elif not {"-3d"}.isdisjoint(sys.argv):
        parser.add_argument("-3d", action="store_true")
        parser.add_argument("--filename-problem", type=str)
        parser.add_argument("--problems-per-sample", type=int, default=200)
        parser.add_argument("--filename-config", type=str)
        args = parser.parse_args()
        run_experiments_and_plot(
            max_number_problems_to_run=args.problems_per_sample,
            plot_error_3d=True,
            filename_problems=args.filename_problem,
            name_config_yaml_file=args.filename_config,
            folder_problems="ablation_study/problems/",
            folder_config_file="ablation_study/configs_GP/",
        )
    else:
        run_experiments_and_plot(max_number_problems_to_run=2)

# %%
