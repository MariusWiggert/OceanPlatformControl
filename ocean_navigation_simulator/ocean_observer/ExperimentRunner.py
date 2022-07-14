import datetime
import math
from collections import defaultdict
from typing import Union, Tuple, List, Dict, Any, Optional

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import xarray
import yaml

import ocean_navigation_simulator.ocean_observer.metrics.plot_metrics as plot_metrics
from ocean_navigation_simulator.controllers import Controller
from ocean_navigation_simulator.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint, PlatformState
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.environment.data_sources.DataSources import DataSource
from ocean_navigation_simulator.ocean_observer.Observer import Observer
from ocean_navigation_simulator.ocean_observer.PredictionsAndGroundTruthOverArea import \
    PredictionsAndGroundTruthOverArea
from ocean_navigation_simulator.problem_factories.HighwayProblemFactory import HighwayProblemFactory
from ocean_navigation_simulator.problem_factories.NaiveProblemFactory import NaiveProblemFactory
from ocean_navigation_simulator.utils.units import Distance


def _plot_metrics_per_h(metrics: dict[str, any]) -> None:
    c = 5
    r = math.ceil(len(metrics["time"][0]) / c)
    fig, axs = plt.subplots(r, c)
    t = [datetime.datetime.fromtimestamp(time[0], tz=datetime.timezone.utc) for time in metrics["time"]]
    r2_per_h = metrics["r2_per_h"]
    for i in range(r2_per_h.shape[1]):
        axs[i // c, i % c].plot(t, r2_per_h[:, i], label="r2")
        axs[i // c, i % c].set_title(f"r2 score, index: +{i} hours")


def _plot_metrics(metrics: Dict[str, any]) -> None:
    """Plot the r2 and vector correlations over time on 4 different sub-plots.
    
    Args:
        metrics: dictionary containing the name of the metrics, and it's given value for each timestep in a ndarray
    """
    fig, axs = plt.subplots(2, 2)
    t = [datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc) for time in metrics["time"]]
    axs[0, 0].plot(t, metrics["r2"], label="r2")
    axs[0, 0].set_title("r2 score")
    axs[1, 0].plot(t, metrics["vector_correlation_ratio"], label="vector_correlation_ratio")
    axs[1, 0].set_title("vector correlation ratio")
    axs[0, 1].plot(t, metrics["vector_correlation_improved"],
                   label="vector correlation improved")
    axs[0, 1].set_title("vector correlation model")
    axs[1, 1].plot(t, metrics["vector_correlation_initial"], label="vector_correlation_ref")
    axs[1, 1].set_title("vector correlation initial")
    plt.legend()

    print(f"Mean r2_loss: {metrics['r2'].mean()}")


class ExperimentRunner:
    """ Class to run the experiments using a config yaml file to set up the experiment and the environment and load the ."""

    def __init__(self, yaml_file_config: Union[str, Dict[str, any]], filename_problems=None):
        """Create the ExperimentRunner object using a yaml file referenced by yaml_file_config. Used to run problems and
        get results represented by metrics

        Args:
            yaml_file_config: the name (without path or extension) of the Yaml file that will be read in the folder:
            "ocean_navigation_simulator/env/scenarios/"
        """
        if type(yaml_file_config) is str:
            with open(f'scenarios/ocean_observer/{yaml_file_config}.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self.variables = config["experiment_runner"]
        else:
            config = yaml_file_config
            self.variables = config["experiment_runner"]

        if filename_problems is not None:
            self.variables["problems_file"] = f'scenarios/ocean_observer/{filename_problems}.yaml'

        if self.variables.get("use_real_data", True):
            problems = []

            if "problems_file" in self.variables.keys():
                with open(self.variables["problems_file"]) as f:
                    yaml_problems = yaml.load(f, Loader=yaml.FullLoader)
            else:
                yaml_problems = self.variables
            # Specify Problem
            for problem_dic in yaml_problems.get("problems", []):
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
        self.last_file_used = None
        self.list_dates_when_new_files = []

    def run_all_problems(self, max_number_problems_to_run=None) -> Tuple[
        List[Dict[str, any]], List[Dict[str, any]], Dict[str, any]]:
        """Run all the problems that were specified when then ExperimentRunner object was created consecutively and
        provide the metrics computed for each problem

        Returns: List of dictionaries where the i-th item of the list is a dict with all the metrics computed for the
        i-th problem.
        """

        results = []
        results_per_h = []
        while self.problem_factory.has_problems_remaining() and (
                type(max_number_problems_to_run) != int or max_number_problems_to_run > 0):
            res, res_per_h = self.run_next_problem()
            results.append(res)
            results_per_h.append(res_per_h)
            self.__create_plots(results[-1], results_per_h[-1])
            if type(max_number_problems_to_run) == int:
                max_number_problems_to_run -= 1

        merged = defaultdict(list)
        for key in results[-1].keys():
            merged[key] = [r[key].mean() for r in results]
        for key in results_per_h[-1].keys():
            if key == "time":
                continue
            for r in results_per_h:
                all_hours = r[key].mean(axis=0)
                for h in range(len(all_hours)):
                    merged[key + "_" + str(h)] += [all_hours[h]]
        return results, results_per_h, merged, self.list_dates_when_new_files

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
        self.list_dates_when_new_files = []

        burnin = self.variables.get('number_burnin_steps', 0)
        if burnin > 0:
            for i in range(burnin):
                self.__step_simulation(controller, fit_model=False)
                # position = arena_obs.platform_state.to_spatial_point()
            print(f"End of burnin ({burnin} steps)")
        print("start predicting")

        metrics_names = []
        metrics_per_h_names = []
        metrics = []
        metrics_per_h = []
        results = []

        # Now we run the algorithm
        for i in range(self.variables["number_steps_prediction"]):
            model_prediction = self.__step_simulation(controller, fit_model=True)
            if not i:
                print("Shape predictions: ", dict(model_prediction.dims))
            # get ground truth
            ground_truth = self.arena.ocean_field.hindcast_data_source.get_data_over_area(
                *self.__get_lon_lat_time_intervals(ground_truth=True),
                temporal_resolution=self.variables.get("delta_between_predictions_in_sec", None))
            # print("ground_truth:", ground_truth)
            ground_truth = ground_truth.assign_coords(depth=model_prediction.depth.to_numpy().tolist())

            # compute the metrics and log the results
            self.last_prediction_ground_truth = PredictionsAndGroundTruthOverArea(model_prediction, ground_truth)
            results.append(self.last_prediction_ground_truth)
            name_metrics = self.variables["metrics"] if "metrics" in self.variables.keys() else None
            directions = self.variables.get("direction_current", ["uv"])
            metric = self.last_prediction_ground_truth.compute_metrics(name_metrics, directions=directions)
            metric_per_hour = self.last_prediction_ground_truth.compute_metrics(name_metrics, directions=directions,
                                                                                per_hour=True)
            # In case of Nan only
            if len(metric_per_hour) == 0:
                continue
            if not len(metrics_names):
                metrics_names = ["time"] + list(metric.keys())
                metrics_per_h_names = ["time"] + list(metric_per_hour.keys())

            metrics.append(np.insert(np.fromiter(metric.values(), dtype=float), 0,
                                     self.last_observation.platform_state.date_time.timestamp()))
            values_per_h = np.stack(list(metric_per_hour.values()))
            # times_per_h = np.array([self.last_observation.platform_state.date_time.timestamp() + int(datetime.timedelta(
            #    hours=i).seconds * 1e6) for i in range(values_per_h.shape[1])], ndmin=2)
            times_per_h = np.array([self.last_observation.platform_state.date_time.timestamp()] * values_per_h.shape[1],
                                   ndmin=2)
            metrics_per_h.append(np.concatenate((times_per_h, values_per_h)))
            # print(
            #    f"step {i + 1}/{self.variables['number_steps_prediction']}, time:{self.last_observation.platform_state.date_time}, metrics: {list(zip(metrics_names, metrics[-1]))}")

        metrics = np.array(metrics)
        metrics_per_h = np.array(metrics_per_h)
        return {name: metrics[:, i] for i, name in enumerate(metrics_names)}, {name: metrics_per_h[:, i, :] for i, name
                                                                               in enumerate(metrics_per_h_names)}

    def __create_plots(self, last_metrics: Optional[np.ndarray] = None,
                       last_metrics_per_h: Optional[np.ndarray] = None):
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
        if plots_dict.get("visualize_metrics", False) and last_metrics is not None:
            _plot_metrics(last_metrics)
            _plot_metrics_per_h(last_metrics_per_h)
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

        if (self.last_file_used != self.last_observation.forecast_data_source.DataArray.encoding['source']) or \
                self.last_file_used is None:
            if self.variables.get("clear_observations_when_new_file", False):
                print("clearing observations", self.last_observation.forecast_data_source.DataArray.encoding['source'],
                      "\n",
                      self.last_file_used)
                self.observer.reset()

            self.last_file_used = self.last_observation.forecast_data_source.DataArray.encoding['source']
            self.list_dates_when_new_files.append(self.last_observation.platform_state.date_time)
        self.observer.observe(self.last_observation)
        predictions = None
        if fit_model:
            self.observer.fit()
            predictions = self.observer.get_data_over_area(*self.__get_lon_lat_time_intervals(),
                                                           temporal_resolution=self.variables.get(
                                                               "delta_between_predictions_in_sec", None))
            # todo: quick fix, Find why the predictions are not always the same shape:
            n_steps = self.variables.get("number_steps_to_predict", 12)
            if len(predictions["time"]) > n_steps:
                predictions = predictions.isel(time=range(n_steps))

        return predictions

    def __get_lon_lat_time_intervals(self, ground_truth: bool = False) -> Tuple[List[float],
                                                                                List[float], Union[List[float], List[
        datetime.datetime]]]:
        """ Internal method to get the area around the platforms
        Args:
            ground_truth: If we request the area for ground truth we request a larger area so that interpolation
                          to the predicted area works and does not need to extrapolate.
        Returns:
            longitude, latitude and time intervals packed as tuples and each interval as a list of 2 elements
        """
        point = self.last_observation.platform_state.to_spatio_temporal_point()
        deg_around_x0_xT_box = self.variables.get("radius_area_around_platform", 1)
        temp_horizon_in_s = self.variables.get("number_steps_to_predict", 12) * 3600
        if ground_truth:
            deg_around_x0_xT_box += self.variables.get("gt_additional_area", 0.5)
            temp_horizon_in_s += self.variables.get("gt_additional_time", 3600) * 2
            point.date_time = point.date_time - datetime.timedelta(
                seconds=self.variables.get("gt_additional_time", 3600))
        t, lat, lon = DataSource.convert_to_x_y_time_bounds(
            x_0=point, x_T=point, deg_around_x0_xT_box=deg_around_x0_xT_box, temp_horizon_in_s=temp_horizon_in_s)
        return lon, lat, t
