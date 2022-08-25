from ocean_navigation_simulator.generative_error_model.models.OceanCurrentNoiseField import OceanCurrentNoiseField
from ocean_navigation_simulator.generative_error_model.Dataset import Dataset
from ocean_navigation_simulator.generative_error_model.BuoyData import TargetedTimeRange
from ocean_navigation_simulator.generative_error_model.Problem import Problem
from ocean_navigation_simulator.generative_error_model.generative_model_metrics import get_metrics
from utils import load_config, timer, setup_logger

import pandas as pd
import numpy as np
import xarray as xr
import yaml
from typing import Dict, Any, List


class ExperimentRunner:
    """Takes a GenerativeModel, runs experiments and reports metrics."""

    def __init__(self, logger=None):
        self.config = load_config()
        self.variables = self.config["experiment_runner"]
        self.dataset_name = self.variables["dataset"]
        print(f"Running with {self.dataset_name} data and params.\n")

        # setup model
        model_config = self.config["model"]
        if model_config["type"] == "simplex_noise":
            self.model_config = model_config["simplex_noise"]
            harmonic_params = model_config["simplex_noise"][self.dataset_name]
            self.model = OceanCurrentNoiseField(harmonic_params)
            rng = np.random.default_rng(12345678)
            self.model.reset(rng)
            print(f"Using {model_config['type']} model.")
        if model_config["type"] == "gan":
            raise Exception("GAN model has not been implemented yet!")

        # read in problems and create problems list
        self.problems = self.get_problems()
        self.dataset = Dataset(self.dataset_name)

    def reset(self):
        """Resets the seed of the simplex noise model. Needed to generate diverse samples.
        """
        new_seed = self.rng.choice(20000, 1)
        new_rng = np.random.default_rng(new_seed)
        self.model.reset(new_rng)

    def run_all_problems(self):
        results = []
        for problem in self.problems:
            results.append(self.run_problem(problem))
            print("problem results:", {name: metric for name, metric in results[-1].items()})
        return results

    def run_problem(self, problem: Problem) -> Dict[str, Any]:
        ground_truth = self.get_ground_truth(problem)
        noise_field = self.model.get_noise(problem)

        # need to multiply by original variance and add original mean
        detrend_stats = self.model_config[self.dataset_name]["detrend_stats"]
        print(f"Detrend statistics: {detrend_stats}.")
        noise_field = noise_field*detrend_stats[1] + detrend_stats[0]

        synthetic_error = self._get_samples_from_synthetic(noise_field, ground_truth)
        metrics = self._calculate_metrics(ground_truth, synthetic_error)
        return metrics

    def get_ground_truth(self, problem: Problem):
        """Loads the ground truth data for a specific problem.
        """
        ground_truth = self.dataset.get_specific_data(problem.lon_range, problem.lat_range, problem.t_range)
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

    def _get_samples_from_synthetic(self, noise_field: xr.Dataset, ground_truth: pd.DataFrame) -> pd.DataFrame:
        """Takes the generated error and takes samples where buoys are located in space and time.
        """
        synthetic_data = ground_truth[["time", "lon", "lat"]][:400]
        synthetic_data["u_error"] = 0
        synthetic_data["v_error"] = 0
        n = 10
        for i in range(0, synthetic_data.shape[0], n):
            noise_field_interp = noise_field.interp(time=synthetic_data.iloc[i:i+n]["time"],
                                                    lon=synthetic_data.iloc[i:i+n]["lon"],
                                                    lat=synthetic_data.iloc[i:i+n]["lat"])
            synthetic_data["u_error"].iloc[i:i+n] = noise_field_interp["u_error"].values.diagonal().diagonal()
            synthetic_data["v_error"].iloc[i:i+n] = noise_field_interp["v_error"].values.diagonal().diagonal()
        print(f"Percentage of failed interp: {100*np.isnan(synthetic_data['u_error']).sum()/synthetic_data.shape[0]}%.")
        return synthetic_data

    def _calculate_metrics(self, ground_truth, synthetic_error) -> Dict[str, float]:
        metrics = dict()
        wanted_metrics = self.variables.get("metrics", None)
        for s, f in get_metrics().items():
            if s in wanted_metrics:
                metrics |= f(ground_truth, synthetic_error)  # update with new key-val pair
        return metrics

    def _create_plots(self):
        return


@timer
def main():
    ex_runner = ExperimentRunner()
    ex_runner.run_all_problems()


if __name__ == "__main__":
    main()
