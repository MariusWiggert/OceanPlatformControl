import copy

from ocean_navigation_simulator.generative_error_model.models.OceanCurrentNoiseField import OceanCurrentNoiseField
from ocean_navigation_simulator.generative_error_model.Dataset import Dataset
from ocean_navigation_simulator.generative_error_model.BuoyData import TargetedTimeRange
from ocean_navigation_simulator.generative_error_model.Problem import Problem
from ocean_navigation_simulator.generative_error_model.generative_model_metrics import get_metrics
from utils import load_config, timer, get_path_to_project, setup_logger

import pandas as pd
import numpy as np
import xarray as xr
import yaml
from typing import Dict, List, Optional
from tqdm import tqdm
import datetime
import os


class ExperimentRunner:
    """Takes a GenerativeModel, runs experiments and saved a synthetic dataset. The
    synthetic dataset is then used to construct a Variogram which is used for validation."""

    def __init__(self, yaml_file_config: str, logger=None):
        self.config = load_config(yaml_file_config)
        self.variables = self.config["experiment_runner"]
        self.project_path = get_path_to_project(os.getcwd())
        self.data_dir = self.config["data_dir"]
        self.dataset_type = self.config["dataset_type"]
        self.dataset_name = self.variables["dataset"]
        print(f"Running with {self.dataset_name} data and params.\n")

        # setup model
        model_config = self.config["model"]
        if model_config["type"] == "simplex_noise":
            self.model_config = model_config["simplex_noise"]
            parameters = model_config["simplex_noise"][self.dataset_name]
            parameters = np.load(os.path.join(self.project_path, self.data_dir, parameters), allow_pickle=True)
            harmonic_params = {"U_COMP": parameters.item().get("U_COMP"),
                               "V_COMP": parameters.item().get("V_COMP")}
            detrend_stats = parameters.item().get("detrend_metrics")
            self.model = OceanCurrentNoiseField(harmonic_params, np.array(detrend_stats))
            self.rng = np.random.default_rng(12345678)
            print(f"Using {model_config['type']} model.")
        if model_config["type"] == "gan":
            raise Exception("GAN model has not been implemented yet!")

        # read in problems and create problems list
        self.problem = self.get_problems()
        # quick way to create more problems for same area
        self.problems = [increment_time(self.problem[0], days) for days in range(40)]
        self.dataset = Dataset(self.data_dir, self.dataset_type, self.dataset_name)
        self.data = pd.DataFrame(columns={"time", "lon", "lat", "u_error", "v_error"})

    def reset(self):
        """Resets the seed of the simplex noise model. Needed to generate diverse samples.
        """
        new_seed = self.rng.choice(20000, 1)
        new_rng = np.random.default_rng(new_seed)
        self.model.reset(new_rng)

    def run_all_problems(self, error_only: bool = True):
        # save first noise field
        self.reset()
        self.model.get_noise_vec(self.problems[0]).to_netcdf(os.path.join(self.project_path, self.data_dir, "sampled_noise.nc"))
        for problem in self.problems:
            self.reset()
            print(f"Running: {problem}")
            self.run_problem(problem, error_only=error_only, sampling_method="real")

    def run_problem(self, problem: Problem, error_only: bool, sampling_method: str = "real") -> Optional[pd.DataFrame]:
        noise_field = self.model.get_noise_vec(problem)
        # folder_name = self.dataset_name
        folder_name = self.dataset_name
        self.save_noise_field(noise_field, problem, folder_name)

        # this is true if dont want to reconstruct the variogram for validation.
        if error_only:
            return None

        if sampling_method == "real":
            ground_truth = self.get_ground_truth(problem)
            synthetic_error = ExperimentRunner._get_samples_from_synthetic(noise_field, ground_truth)
            self.save_data(synthetic_error, problem, folder_name)
        elif sampling_method == "random":
            synthetic_error = ExperimentRunner._get_random_samples_from_synthetic(noise_field, 20)
            self.save_data(synthetic_error, problem, "random_samples")
        return synthetic_error

    def get_ground_truth(self, problem: Problem):
        """Loads the ground truth data for a specific problem.
        """
        ground_truth = self.dataset.get_recent_data_in_range(problem.lon_range, problem.lat_range, problem.t_range)
        ground_truth["time"] = pd.to_datetime(ground_truth["time"])
        return ground_truth

    def get_problems(self) -> List[Problem]:
        """Loads problems from yaml file and returns them as a list of object of type Problem.
        """
        problems = []
        if "problems_file" in self.variables.keys():
            with open(self.variables["problem_file"]) as f:
                yaml_problems = yaml.load(f, Loader=yaml.FullLoader)
        else:
            yaml_problems = self.variables
        for problem_dict in yaml_problems.get("problems", []):
            data_ranges = problem_dict["data_ranges"]
            targeted_time_range = TargetedTimeRange(data_ranges["t_range"])
            lon_range = data_ranges["lon_range"]
            lat_range = data_ranges["lat_range"]
            t_range = [targeted_time_range.get_start(), targeted_time_range.get_end()]
            problems.append(Problem(lon_range, lat_range, t_range))
        return problems

    @staticmethod
    def _get_samples_from_synthetic(noise_field: xr.Dataset, ground_truth: pd.DataFrame) -> pd.DataFrame:
        """Takes the generated error and takes samples where buoys are located in space and time.
        Note: Need to ensure that there is good overlap between points in ground_truth and the simplex
        noise sample. Otherwise, there will be large interpolation errors. Rows which contain at least
        one NaN value are dropped. However, if there is no overlap at all, there will be no output data.
        """
        synthetic_data = ground_truth[["time", "lon", "lat"]]
        synthetic_data["u_error"] = 0
        synthetic_data["v_error"] = 0
        n = 10
        for i in tqdm(range(0, synthetic_data.shape[0], n)):
            noise_field_interp = noise_field.interp(time=synthetic_data.iloc[i:i+n]["time"],
                                                    lon=synthetic_data.iloc[i:i+n]["lon"],
                                                    lat=synthetic_data.iloc[i:i+n]["lat"])
            synthetic_data["u_error"].iloc[i:i+n] = noise_field_interp["u_error"].values.diagonal().diagonal()
            synthetic_data["v_error"].iloc[i:i+n] = noise_field_interp["v_error"].values.diagonal().diagonal()
        print(f"Percentage of failed interp: {100*np.isnan(synthetic_data['u_error']).sum()/synthetic_data.shape[0]}%.")
        synthetic_data = synthetic_data.dropna()
        return synthetic_data

    @staticmethod
    def _get_random_samples_from_synthetic(noise_field: xr.Dataset, num_samples: int):
        """Instead of sampling the simplex noise at actual buoy points, this methods
        just uses random positions in space for each time step to sample the noise.
        """
        time_range = [noise_field["time"][0].values, noise_field["time"][-1].values]
        time_steps = len(noise_field["time"].values)
        lon_range = [noise_field["lon"].values.min(), noise_field["lon"].values.max()]
        lat_range = [noise_field["lat"].values.min(), noise_field["lat"].values.max()]

        lon_samples = np.random.uniform(lon_range[0], lon_range[1], num_samples*time_steps)
        lat_samples = np.random.uniform(lat_range[0], lat_range[1], num_samples*time_steps)
        hour_samples = np.random.choice(np.arange(time_steps), num_samples*time_steps)
        time_samples = np.array([time_range[0] + np.timedelta64(hours, 'h') for hours in hour_samples])

        synthetic_data = pd.DataFrame({"time": time_samples, "lon": lon_samples, "lat": lat_samples})
        synthetic_data["u_error"] = 0
        synthetic_data["v_error"] = 0

        n = 10
        for i in tqdm(range(0, synthetic_data.shape[0], n)):
            noise_field_interp = noise_field.interp(time=synthetic_data.iloc[i:i+n]["time"],
                                                    lon=synthetic_data.iloc[i:i+n]["lon"],
                                                    lat=synthetic_data.iloc[i:i+n]["lat"])
            synthetic_data["u_error"].iloc[i:i+n] = noise_field_interp["u_error"].values.diagonal().diagonal()
            synthetic_data["v_error"].iloc[i:i+n] = noise_field_interp["v_error"].values.diagonal().diagonal()
        print(f"Percentage of failed interp: {100*np.isnan(synthetic_data['u_error']).sum()/synthetic_data.shape[0]}%.")
        synthetic_data = synthetic_data.dropna()
        return synthetic_data

    def save_data(self, synthetic_error_samples: pd.DataFrame, problem: Problem, folder_name: str) -> None:
        """Save synthetic samples for computing the variogram.
        """
        synthetic_error_dir = os.path.join(self.project_path,
                                           self.data_dir,
                                           "dataset_synthetic_error",
                                           folder_name)
        if not os.path.exists(synthetic_error_dir):
            os.makedirs(synthetic_error_dir)

        date_format = "%Y-%m-%dT%H:%M:%SZ"
        file_name = f"synthetic_data_error_lon_[{problem.lon_range[0]},{problem.lon_range[1]}]_"\
                    f"lat_[{problem.lat_range[0]},{problem.lat_range[1]}]_"\
                    f"time_{problem.t_range[0].strftime(date_format)}__{problem.t_range[1].strftime(date_format)}.csv"
        file_path = os.path.join(synthetic_error_dir, file_name)
        if not os.path.exists(file_path):
            synthetic_error_samples.to_csv(file_path, index=False)
            print(f"Saved synthetic error dataset to: {synthetic_error_dir}.\n")

    def save_noise_field(self, noise_field: xr.Dataset, problem: Problem, folder_name: str):
        """Save the noise fields create by OceanCurrentNoiseField.
        """
        noise_field_dir = os.path.join(self.project_path, self.data_dir, "synthetic_error", folder_name)
        if not os.path.exists(noise_field_dir):
            os.makedirs(noise_field_dir)

        date_format = "%Y-%m-%dT%H:%M:%SZ"
        file_name = f"synthetic_data_error_lon_[{problem.lon_range[0]},{problem.lon_range[1]}]_"\
                    f"lat_[{problem.lat_range[0]},{problem.lat_range[1]}]_"\
                    f"time_{problem.t_range[0].strftime(date_format)}__{problem.t_range[1].strftime(date_format)}.nc"
        file_path = os.path.join(noise_field_dir, file_name)
        if not os.path.exists(file_path):
            noise_field.to_netcdf(file_path)
            print(f"Saved noise field to: {noise_field_dir}.")

    def _calculate_metrics(self, ground_truth, synthetic_error) -> Dict[str, float]:
        metrics = dict()
        wanted_metrics = self.variables.get("metrics", None)
        for s, f in get_metrics().items():
            if s in wanted_metrics:
                metrics |= f(ground_truth, synthetic_error)  # update with new key-val pair
        return metrics

    def _create_plots(self):
        return


def increment_time(problem: Problem, days: int):
    """Takes a Problem and increments the time range by one day.500
    """
    problem = copy.deepcopy(problem)
    for i in range(len(problem.t_range)):
        problem.t_range[i] = problem.t_range[i] + datetime.timedelta(days=days)
    return problem


@timer
def main():
    ex_runner = ExperimentRunner(yaml_file_config="config_buoy_data.yaml")
    ex_runner.run_all_problems(error_only=False)


if __name__ == "__main__":
    main()