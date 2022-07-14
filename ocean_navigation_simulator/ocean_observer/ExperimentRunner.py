import datetime
from typing import Union, Tuple, List, Dict, Any, Optional

import dateutil
import numpy as np
import xarray
import yaml
import matplotlib.pyplot as plt

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.problem_factories.HighwayProblemFactory import HighwayProblemFactory
from ocean_navigation_simulator.problem_factories.NaiveProblemFactory import NaiveProblemFactory
from ocean_navigation_simulator.ocean_observer.Observer import Observer
from ocean_navigation_simulator.ocean_observer.PredictionsAndGroundTruthOverArea import PredictionsAndGroundTruthOverArea
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint, PlatformState
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.controllers import Controller
from ocean_navigation_simulator.controllers.NaiveToTargetController import NaiveToTargetController
from ocean_navigation_simulator.utils.units import Distance
from ocean_navigation_simulator.environment.data_sources.DataSources import DataSource
import ocean_navigation_simulator.ocean_observer.metrics.plot_metrics as plot_metrics

# TODO: think through what to do when a new forecast becomes available. The old current error measurements
# are not valid anymore (because they are samples from an old error function). So shall we throw them away
# or maybe log the observations of the currents (not the error) which should still contain information
# that's useful to improve the new forecast.


class ExperimentRunner:
    """ Class to run the experiments using a config yaml file to set up the experiment and the environment and load the ."""

    def __init__(self, yaml_file_config: str):
        """Create the ExperimentRunner object using a yaml file referenced by yaml_file_config. Used to run problems and
        get results represented by metrics

        Args:
            yaml_file_config: the name (without path or extension) of the Yaml file that will be read in the folder:
            "ocean_navigation_simulator/env/scenarios/"
        """
        with open(f'scenarios/ocean_observer/{yaml_file_config}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.variables = config["experiment_runner"]

        if self.variables.get("use_real_data", True):
            problems = []
            # Specify Problem
            for problem_dic in self.variables.get("problems", []):
                init_pos = problem_dic["initial_position"]
                x_0 = PlatformState(lon=Distance(deg=init_pos["lon_in_deg"]),
                                    lat=Distance(deg=init_pos["lat_in_deg"]),
                                    date_time=dateutil.parser.isoparse(init_pos["datetime"]))
                target_point = problem_dic["target"]
                x_t = SpatialPoint(lon=Distance(deg=target_point["lon_in_deg"]),
                                   lat=Distance(deg=target_point["lat_in_deg"]))
                problems.append(Problem(start_state=x_0, end_region=x_t, target_radius=target_point["radius_in_m"]))
            self.problem_factory = NaiveProblemFactory(problems)
        else:
            self.problem_factory = HighwayProblemFactory(
                [(SpatialPoint(Distance(meters=0), Distance(meters=0)),
                  SpatialPoint(Distance(deg=10), Distance(deg=10)))])

        self.observer = Observer(config["observer"])
        self.arena = ArenaFactory.create(scenario_name=self.variables["scenario_used"])
        self.last_observation, self.last_prediction_ground_truth = None, None

    def run_all_problems(self) -> List[Dict[str, any]]:
        """Run all the problems that were specified when then ExperimentRunner object was created consecutively and
        provide the metrics computed for each problem

        Returns: List of dictionaries where the i-th item of the list is a dict with all the metrics computed for the
        i-th problem.
        """
        results = []
        while self.problem_factory.has_problems_remaining():
            results.append(self.run_next_problem())
            self.__create_plots(results[-1])
            print("problem results:", {name: metric.mean() for name, metric in results[-1].items()})
        return results

    def run_next_problem(self) -> Dict[str, Any]:
        """ Run the next problem. It creates a NaiveToTargetController based on the problem, reset the arena and
        observer. Gather "number_burnin_steps" observations without fitting the model and then start predicting the
        model at each timestep and evaluate for that timestep the prediction compared to the hindcast.

        Returns:
            dictionary with the pairs: (metric_name, 1d array containing the output from that metric at each timestep)
        """
        if not self.problem_factory.has_problems_remaining():
            raise StopIteration()

        problem = self.problem_factory.next_problem()
        controller = NaiveToTargetController(problem)
        self.last_observation = self.arena.reset(problem.start_state)
        self.observer.reset()

        for i in range(self.variables.get("number_burnin_steps", 0)):
            self.__step_simulation(controller, fit_model=False)
            # position = arena_obs.platform_state.to_spatial_point()
        print(f"End of burnin ({self.variables.get('number_burnin_steps', 0)} steps)")
        print("start predicting")

        metrics_names = []
        metrics = []
        results = []

        # Now we run the algorithm
        for i in range(self.variables["number_steps_prediction"]):
            model_prediction = self.__step_simulation(controller, fit_model=True)
            # get ground truth
            ground_truth = self.arena.ocean_field.hindcast_data_source.get_data_over_area(
                *self.__get_lon_lat_time_intervals(ground_truth=True))

            # compute the metrics and log the results
            self.last_prediction_ground_truth = PredictionsAndGroundTruthOverArea(model_prediction, ground_truth)
            results.append(self.last_prediction_ground_truth)
            metric = self.last_prediction_ground_truth.compute_metrics(self.variables.get("metrics", None))
            if not len(metrics_names):
                metrics_names = ["time"] + list(metric.keys())

            metrics.append(np.insert(np.fromiter(metric.values(), dtype=float), 0,
                                     self.last_observation.platform_state.date_time.timestamp()))
            print(
                f"step {i + 1}/{self.variables['number_steps_prediction']}, metrics: {list(zip(metrics_names, metrics[-1]))}")

        metrics = np.array(metrics)
        return {name: metrics[:, i] for i, name in enumerate(metrics_names)}

    def __create_plots(self, last_metrics: Optional[np.ndarray] = None):
        """ Create the different plots based on the yaml file to know which one to display
        Args:
            last_metrics: the metrics to display
        """
        plots_dict = self.variables.get("plots", {})
        if plots_dict.get("metrics_to_visualize", False) and last_metrics is not None:
            for metrics_str in plots_dict.get("metrics_to_visualize"):
                fig, ax = plt.subplots()
                getattr(plot_metrics, "plot_" + metrics_str)(last_metrics, ax)
                plt.show()
        if plots_dict.get("visualize_currents", False):
            self.last_prediction_ground_truth.visualize_improvement_forecasts(self.arena.state_trajectory)
        for variable in plots_dict.get("plot_3d", []):
            self.last_prediction_ground_truth.plot_3d(variable)

    def __step_simulation(self, controller: Controller, fit_model: bool = True) -> Union['xarray', None]:
        """ Run one step of the simulation. Will return the predictions and ground truth as an xarray if we fit the
         model. We save the last observation.

        Args:
            controller: Controller that tells use which action to apply
            fit_model: Whether we should just gather observations or also predict the improved forecast around the
                        platform position

        Returns:
            The xarray dataset containing the initial and improved forecasts, the errors and also the uncertainty if
            it is provided by the observer depending on the OceanCurrentModel used.
        """
        action_to_apply = controller.get_action(self.last_observation)
        self.last_observation = self.arena.step(action_to_apply)
        self.observer.observe(self.last_observation)

        predictions = None
        if fit_model:
            self.observer.fit()
            predictions = self.observer.get_data_over_area(*self.__get_lon_lat_time_intervals())

        return predictions

    def __get_lon_lat_time_intervals(self, ground_truth: bool = False) -> Tuple[List[float],
                                                    List[float], Union[List[float], List[datetime.datetime]]]:
        """ Internal method to get the area around the platforms
        Args:
            ground_truth: If we request the area for ground truth we request a larger area so that interpolation
                          to the predicted area works and does not need to extrapolate.
        Returns:
            longitude, latitude and time intervals packed as tuples and each interval as a list of 2 elements
        """
        point = self.last_observation.platform_state.to_spatio_temporal_point()
        deg_around_x0_xT_box = self.variables.get("radius_area_around_platform", 1)
        temp_horizon_in_s = self.variables.get("time_horizon_predictions_in_sec", 86400)
        if ground_truth:
            deg_around_x0_xT_box += self.variables.get("gt_additional_area", 0.5)
            temp_horizon_in_s += self.variables.get("gt_additional_time", 3600) * 2
            point.date_time = point.date_time - datetime.timedelta(seconds=self.variables.get("gt_additional_time", 3600))
        t, lat, lon = DataSource.convert_to_x_y_time_bounds(
            x_0=point, x_T=point, deg_around_x0_xT_box=deg_around_x0_xT_box, temp_horizon_in_s=temp_horizon_in_s)
        return lon, lat, t
