import yaml
from ray import tune

from ocean_navigation_simulator.ocean_observer.ExperimentRunner import ExperimentRunner


def objective(config):
    score = config["rmse_improved"]
    return {"score": score}


search_space = {
    "kernel": tune.choice(["RationalQuadratic"]),  # tune.choice(["matern", "rbf", "periodic", "RationalQuadratic"]),
    # "lat_scale": tune.uniform(1, 1e6),
    # "lon_scale": tune.uniform(1, 1e6),
    # "time_scale": tune.uniform(1, 1e6),
    "length_scale": tune.uniform(1, 1e6),
    "alpha": tune.loguniform(1e-5, 2.5),
    "length_scale_bounds": "fixed",
    "alpha_bounds": "fixed"
}


# import os
# import ray
#
# runtime_env = {"working_dir": os.getcwd()}
# print("runtime_env:", runtime_env)
# ray.init(runtime_env=runtime_env)


def train(config):
    import os
    os.chdir("/Users/fedosha/polybox/semester4/codebase/OceanPlatformControl/")
    yaml_file_config = "/Users/fedosha/polybox/semester4/codebase/OceanPlatformControl/scenarios/ocean_observer/config_GP_for_failed_case.yaml"
    # print("dir:", directory)
    with open(yaml_file_config) as f:
        config_yaml = yaml.load(f, Loader=yaml.FullLoader)
        config_yaml["observer"]["model"]["gaussian_process"]["kernel"] = \
            {"type": config["kernel"],
             # "scaling": {"longitude": config["lon_scale"], "latitude": config["lat_scale"],
             #            "longitude": config["time_scale"]},

             "parameters": {"alpha": config["alpha"], "length_scale_bounds": config["length_scale_bounds"],
                            "alpha_bounds": "fixed", "length_scale": config["length_scale"]}
             }
        exp = ExperimentRunner(config_yaml)
        results = exp.run_all_problems()
        print("final results:", results)

    # variables = config["experiment_runner"]


def main_tune():
    res = tune.run(train, config=search_space, log_to_file=True)
    print("res is: ", res)
    return {"score": 10}


def main():
    """
    Run an experiment
    """
    # np.random.seed(0)
    exp = ExperimentRunner("config_GP_for_failed_case")
    results = exp.run_all_problems()
    print("final results:", results)


if __name__ == "__main__":
    main()
    # main_tune()

# %%
