import datetime
import time
from typing import Union, Tuple, List, Dict, Any

import dateutil
import numpy as np
import xarray
import yaml
from matplotlib import pyplot as plt

from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.HighwayProblemFactory import HighwayProblemFactory
from ocean_navigation_simulator.env.NaiveProblemFactory import NaiveProblemFactory
from ocean_navigation_simulator.env.Observer_rebased import Observer
from ocean_navigation_simulator.env.PlatformState import SpatialPoint, PlatformState
from ocean_navigation_simulator.env.PredictionsAndGroundTruthOverArea import PredictionsAndGroundTruthOverArea
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.controllers import Controller
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.utils.units import Distance


def _plot_metrics(dict: Dict[str, any]):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(dict["time"], dict["r2"], label="r2_loss")
    axs[0, 0].set_title("r2_loss")
    axs[1, 0].plot(dict["time"], dict["vector_correlation_ratio"], label="vector_correlation_ratio")
    axs[1, 0].set_title("vector correlation ratio")
    axs[0, 1].plot(dict["time"], dict["vector_correlation_improved"],
                   label="vector correlation improved")
    axs[0, 1].set_title("vector correlation model")
    axs[1, 1].plot(dict["time"], dict["vector_correlation_initial"], label="vector_correl_ref")
    axs[1, 1].set_title("vector correlation initial")
    plt.legend()

    print(f"Mean r2_loss: {dict['r2'].mean()}")


class ExperimentRunner:

    def __init__(self, yaml_file_config: str):
        with open(f'ocean_navigation_simulator/env/scenarios/{yaml_file_config}.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.variables = config["experiment_runner"]
            self.plots_config = self.variables["plots"]

        if self.variables.get("use_real_data", True):
            problems = []
            # Specify Problem
            for problem_dic in self.variables.get("problems", []):
                init_pos = problem_dic["initial_position"]
                x_0 = PlatformState(lon=Distance(deg=init_pos["lon_in_deg"]),
                                    lat=Distance(deg=init_pos["lat_in_deg"]),
                                    date_time=dateutil.parser.isoparse(init_pos["datetime"]))
                target_point = problem_dic["target"]
                x_T = SpatialPoint(lon=Distance(deg=target_point["lon_in_deg"]),
                                   lat=Distance(deg=target_point["lat_in_deg"]))
                problems.append(Problem(start_state=x_0, end_region=x_T, target_radius=target_point["radius_in_m"]))
            self.problem_factory = NaiveProblemFactory(problems)
        else:
            self.problem_factory = HighwayProblemFactory(
                [(SpatialPoint(Distance(meters=0), Distance(meters=0)),
                  SpatialPoint(Distance(deg=10), Distance(deg=10)))])

        self.observer = Observer(config["observer"])
        self.arena = ArenaFactory.create(scenario_name=self.variables["scenario_used"])
        self.last_observation, self.last_prediction_ground_truth = None, None

    def run_all_problems(self) -> List[Dict[str, any]]:
        results = []
        while self.problem_factory.has_problems_remaining():
            results.append(self.run_next_problem())
            _plot_metrics(results[-1])
            self.last_prediction_ground_truth.visualize_improved_error(self.arena.state_trajectory)
            plt.pause(1)
            print("problem results:", {name: metric.mean() for name, metric in results[-1].items()})
        return results

    def run_next_problem(self) -> Dict[str, Any]:
        if not self.problem_factory.has_problems_remaining():
            raise StopIteration()

        problem = self.problem_factory.next_problem()
        controller = NaiveToTargetController(problem)
        self.last_observation = self.arena.reset(problem.start_state)
        self.observer.reset()

        start = time.time()
        for i in range(self.variables.get("number_burnin_steps", 0)):
            self.step_simulation(controller, fit_model=False)
            # position = arena_obs.platform_state.to_spatial_point()
        print(f"End of burnin ({self.variables.get('number_burnin_steps', 0)} steps)")
        print("start predicting")

        metrics_names = []
        metrics = []
        results = []

        # Now we run the algorithm
        for i in range(self.variables["maximum_steps"]):
            last_prediction = self.step_simulation(controller, fit_model=True)
            results.append(self.last_prediction_ground_truth)

            print(
                f"step {i + 1}/{self.variables['maximum_steps']}:"
                f" error mean:{last_prediction['mean_error_u'].mean(dim=['lon', 'lat', 'time']).to_numpy()}, "
                f"abs error mean:{abs(last_prediction['mean_error_u']).mean(dim=['lon', 'lat', 'time']).to_numpy()}, "
                f"forecasted_std:{last_prediction['std_error_u'].mean().item() if 'std_error_u' in last_prediction.keys() else 'NA'}")

            ground_truth = self.arena.ocean_field.hindcast_data_source.get_data_over_area(
                *self._get_lon_lat_time_intervals())
            differences_predictions_ground_truth = PredictionsAndGroundTruthOverArea(last_prediction, ground_truth)
            self.last_prediction_ground_truth = differences_predictions_ground_truth
            metric = differences_predictions_ground_truth.compute_metrics()
            if not len(metrics_names):
                metrics_names = ["time"] + list(metric.keys())

            metrics.append(np.insert(np.fromiter(metric.values(), dtype=float), 0, last_prediction["time"][0]))
        metrics = np.array(metrics)
        return {name: metrics[:, i] for i, name in enumerate(metrics_names)}

    def step_simulation(self, controller: Controller, fit_model: bool = True) -> Union['xarray', None]:
        action_to_apply = controller.get_action(self.last_observation)
        self.last_observation = self.arena.step(action_to_apply)
        self.observer.observe(self.last_observation)

        # TODO: modify input to respect interface design
        predictions = None
        if fit_model:
            self.observer.fit()
            # todo: find how to get the intervals

            predictions = self.observer.get_data_over_area(*self._get_lon_lat_time_intervals())

        return predictions

    def _get_lon_lat_time_intervals(self) -> Tuple[
        List[float], List[float], Union[List[float], List[datetime.datetime]]]:
        m = self.variables.get("radius_area_around_platform", 1)
        point = self.last_observation.platform_state.to_spatio_temporal_point()
        t, lat, lon = self.last_observation.forecast_data_source. \
            convert_to_x_y_time_bounds(point, point, self.variables.get("radius_area_around_platform", 1),
                                       self.variables.get("time_horizon_predictions_in_sec", 86400))
        return lon, lat, t

    def visualize_currents(self) -> None:
        print("visualize_currents")
        x_y_intervals = self._get_lon_lat_time_intervals()[:2]
        # todo: make it adaptable by taking min max over the 3 sources
        vmin, vmax = -1, 1
        ax1 = self.arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
            self.last_observation.platform_state.date_time, *x_y_intervals, return_ax=True, vmin=vmin, vmax=vmax)
        trajectory_x, trajectory_y = self.arena.state_trajectory[:, 0], self.arena.state_trajectory[:, 1]
        x_lim, y_lim = ax1.get_xlim(), ax1.get_ylim()
        ax1.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax1.set_xlim(x_lim), ax1.set_ylim(y_lim)
        ax1.set_title("True currents")

        ax2 = self.arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
            self.last_observation.platform_state.date_time,
            *x_y_intervals,
            return_ax=True, vmin=vmin, vmax=vmax)
        x_lim, y_lim = ax2.get_xlim(), ax2.get_ylim()
        ax2.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax2.set_xlim(x_lim), ax2.set_ylim(y_lim)
        ax2.set_title("Initial forecasts")

        error_reformated = self.last_prediction_ground_truth[["mean_error_u", "mean_error_v"]].rename(
            mean_error_u="water_u",
            mean_error_v="water_v")
        ax3 = self.arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
            self.last_observation.platform_state.date_time,
            *x_y_intervals,
            error_to_incorporate=error_reformated,
            return_ax=True, vmin=vmin, vmax=vmax)
        x_lim, y_lim = ax3.get_xlim(), ax3.get_ylim()
        ax3.plot(trajectory_x, trajectory_y, color='y', marker='+')
        ax3.set_xlim(x_lim), ax3.set_ylim(y_lim)
        ax3.set_title("Improved forecasts")

        vmin = min(ax1.g)
