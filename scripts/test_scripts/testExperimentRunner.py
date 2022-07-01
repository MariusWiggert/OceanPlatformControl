import csv
import os

import numpy as np
import yaml
from ray import tune

from ocean_navigation_simulator.ocean_observer.ExperimentRunner import ExperimentRunner


def objective(config):
    score = config["rmse_improved"]
    return {"score": score}


def conditional_parameters(str_accepted: list[str], to_return):
    return tune.sample_from(lambda s: to_return if s.config.kernel in str_accepted else None)


search_space = {
    # product and sum are not supported yet
    # "kernel": tune.choice([{"product": ("matern", "rbf")}]),
    "kernel": tune.grid_search(["matern", "rbf", "ExpSineSquared", "RationalQuadratic"]),
    "sigma_exp": tune.qrandn(1, 1, 0.0001),
    # if matern or rbf
    "scaling": conditional_parameters(["matern", "rbf"], {
        "latitude": tune.loguniform(1, 1e6),
        "longitude": tune.loguniform(1, 1e6),
        "time": tune.loguniform(1, 1e6)}),
    # "lon_scale": tune.loguniform(1, 1e6),
    # "time_scale": tune.loguniform(1, 1e6),
    # if matern
    "nu": conditional_parameters(["matern"], tune.choice([0.001, 0.1, 0.5, 1.5, 2.5, 5, 10])),
    # values not in [.5, 1.5, 2.5, inf] are 10x longer to compute

    # if rational quadratic or expsinesquared(=periodic)

    "length_scale": conditional_parameters(["RationalQuadratic", "ExpSineSquared"], tune.uniform(1, 1e6)),
    "length_scale_bounds": "fixed",
    # if rational quadratic
    "alpha": conditional_parameters(["RationalQuadratic"], tune.loguniform(1e-5, 2.5)),
    "alpha_bounds": conditional_parameters(["RationalQuadratic"], "fixed"),

    # if expSineSquared
    "periodicity": conditional_parameters(["ExpSineSquared"], tune.loguniform(0.01, 1000)),
    "periodicity_bounds": conditional_parameters(["ExpSineSquared"], "fixed"),
    "length_scale": conditional_parameters(["ExpSineSquared"], tune.loguniform(1, 1e6)),

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


# import os
# import ray
#
# runtime_env = {"working_dir": os.getcwd()}
# print("runtime_env:", runtime_env)
# ray.init(runtime_env=runtime_env)

def write_row_csv(path, row):
    # open the file in the write mode
    f = open(path, 'a')

    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(row)

    # close the file
    f.close()


def train(config):
    # Create a file for the log of all the results
    file_csv = os.path.join(os.path.dirname(os.getcwd()), "results.csv")
    os.chdir("/Users/fedosha/polybox/semester4/codebase/OceanPlatformControl/")
    yaml_file_config = "./scenarios/ocean_observer/config_real_data_GP.yaml"
    config = {k: v for k, v in config.items() if v is not None}
    print(search_space)
    # print("dir:", directory)
    with open(yaml_file_config) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        type_kernel = config.pop("kernel")
        sigma_exp_squared = abs(config.pop("sigma_exp"))
        # Not supported yet
        # if type in ["product", "sum"]:
        #     config_yaml["observer"]["model"]["gaussian_process"]["kernel"] = {
        #         "type": type,
        #         "kernel_1": ,
        #         "kernel_2":
        #     }
        scaling = config.pop("scaling", None)
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

        print("kernel:", config_yaml["observer"]["model"]["gaussian_process"]["kernel"])
        exp = ExperimentRunner(config_yaml)
        results, results_per_h, merged = exp.run_all_problems()

        # Save the results in the csv file
        merged_mean = {}
        for k in merged.keys():
            if k != "time":
                merged_mean["mean_" + str(k)] = np.array(merged[k]).mean()
        merged_mean |= merged
        merged_mean = {"kernel": str(config_yaml["observer"]["model"]["gaussian_process"]["kernel"])} | merged_mean
        if not os.path.exists(file_csv):
            write_row_csv(file_csv, merged_mean.keys())
        write_row_csv(file_csv, merged_mean.values())

        # return {"r2_avg": np.array([r["r2"] for r in results]).mean()}
        return {"rmse_avg": np.array([r["rmse_improved"] for r in results]).mean()}

    # variables = config["experiment_runner"]


def main_tune():
    res = tune.run(train, config=search_space, num_samples=25)
    # return res.get_best_config(metric="r2_avg", mode="max")
    return res.get_best_config(metric="rmse_avg", mode="min")


def main():
    """
    Run an experiment
    """
    # np.random.seed(0)
    exp = ExperimentRunner("config_real_data_GP")
    results, results_per_h, merged = exp.run_all_problems()
    print("final results:", results)


if __name__ == "__main__":
    # main()
    main_tune()

# %%
