import datetime
import math
from collections import defaultdict
from typing import Union, Tuple, List, Dict, Any, Optional

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray
import yaml
# from fitter import Fitter
from scipy.stats import norm

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


class OutOfGridError(Exception):
    pass


POINTS_PER_DEGREE = 12


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

    def __init__(self, yaml_file_config: Union[str, Dict[str, any]], filename_problems=None,
                 position: Optional[
                     Tuple[Tuple[float, float, datetime.datetime], Tuple[float, float]]] = None,
                 dict_field_yaml_to_update: Dict[str, any] = {}, folder_problems="scenarios/ocean_observer/",
                 folder_config_file="scenarios/ocean_observer/"):
        """Create the ExperimentRunner object using a yaml file referenced by yaml_file_config. Used to run problems and
        get results represented by metrics

        Args:
            yaml_file_config: the name (without path or extension) of the Yaml file that will be read in the folder:
            "ocean_navigation_simulator/env/scenarios/"
        """
        if type(yaml_file_config) is str:
            with open(f'{folder_config_file}{yaml_file_config}.yaml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self.variables = config["experiment_runner"]
        else:
            config = yaml_file_config
            self.variables = config["experiment_runner"]

        # Modify the values from the yaml file with the values from dict_field_yaml_to_update
        for str_keys, new_value in dict_field_yaml_to_update.items():
            t = self.variables
            all_keys = str_keys.split(".")
            for key_dic in all_keys[:-1]:
                t = t[key_dic]
            t[all_keys[-1]] = new_value

        if filename_problems is not None:
            self.variables["problems_file"] = f'{folder_problems}{filename_problems}.yaml'

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
                target_point = problem_dic["target"]
                x_0 = PlatformState(lon=Distance(deg=init_pos["lon_in_deg"]),
                                    lat=Distance(deg=init_pos["lat_in_deg"]),
                                    date_time=dateutil.parser.isoparse(init_pos["datetime"]))
                x_t = SpatialPoint(lon=Distance(deg=target_point["lon_in_deg"]),
                                   lat=Distance(deg=target_point["lat_in_deg"]))
                if position is not None:
                    x_0.lon, x_0.lat = [Distance(deg=p) for p in position[0][:-1]]
                    x_0.date_time = position[0][-1]
                    x_t.lon, x_t.lat = [Distance(deg=p) for p in position[1]]
                problems.append(Problem(start_state=x_0, end_region=x_t, target_radius=target_point["radius_in_m"]))
            self.problem_factory = NaiveProblemFactory(problems)
        else:
            self.problem_factory = HighwayProblemFactory(
                [(SpatialPoint(Distance(meters=0), Distance(meters=0)),
                  SpatialPoint(Distance(deg=10), Distance(deg=10)))])

        self.observer = Observer(config["observer"])
        self.arena = ArenaFactory.create(scenario_name=self.variables["scenario_used"],
                                         folder_scenario=self.variables.get("folder_scenario", None))
        self.last_observation, self.last_prediction_ground_truth = None, None
        self.last_file_used = None
        self.list_dates_when_new_files = []

    @staticmethod
    def _remove_nan_and_flatten(x):
        x = x.flatten()
        return x[np.logical_not(np.isnan(x))]

    @staticmethod
    def _add_normal_on_plot(data, ax, dimension):
        # mu, std = norm.fit(data)
        mu, std = np.mean(data), np.std(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 400)
        p = norm.pdf(x, mu, std)
        print(f"xmin {xmin},xmax {xmax}, mu {mu}, std {std}")
        print()
        obj, = ax.plot(x, p, 'k', linewidth=2)
        obj.set_label("normal distribution")
        ax.legend()
        title = f"Dimension: {dimension}, mean: {mu:.2f}, std: {std:.2f}"
        ax.set_title(title)

    def visualize_all_noise(self, x, y, number_forecasts=30):
        if not self.problem_factory.has_problems_remaining():
            raise StopIteration()

        problem = self.problem_factory.next_problem()
        ti = problem.start_state.date_time
        controller = NaiveToTargetController(problem)
        self.last_observation = self.arena.reset(problem.start_state)
        self.observer.reset()
        self.list_dates_when_new_files = []

        list_forecast_hindcast = []
        self.__step_simulation(controller, fit_model=True)
        for i in range(number_forecasts):
            shift = datetime.timedelta(days=i)
            dim_time = ti + shift
            while self.last_observation.platform_state.date_time < dim_time:
                action_to_apply = controller.get_action(self.last_observation)
                self.last_observation = self.arena.step(action_to_apply)
            dims = self.__get_lon_lat_time_intervals(ground_truth=False)
            dims = [x, y, dims[-1]]
            # fc = self.observer.get_data_over_area(*dims,
            #                                       temporal_resolution=self.variables.get(
            #                                           "delta_between_predictions_in_sec", None))
            r = self.variables.get("radius_area_around_platform", 1)
            t = self.variables.get("number_steps_to_predict", 12) * 3600
            fc = self.observer.get_data_around_platform(
                self.last_observation.platform_state.to_spatio_temporal_point(), r, t)
            margin_area = self.variables.get("gt_additional_area", 0.5)
            margin_time = datetime.timedelta(seconds=self.variables.get("gt_additional_time", 3600))
            dims = [[x[0] - margin_area, x[1] + margin_area], [y[0] - margin_area, y[1] + margin_area],
                    [dims[-1][0] - margin_time, dims[-1][1]]]
            hc = self.arena.ocean_field.hindcast_data_source.get_data_over_area(
                *dims,
                temporal_resolution=self.variables.get("delta_between_predictions_in_sec", None))
            hc = hc.assign_coords(depth=fc.depth.to_numpy().tolist())
            obj = PredictionsAndGroundTruthOverArea(fc, hc)
            list_forecast_hindcast.append((obj.predictions_over_area, obj.ground_truth))
        list_error = [pred[0][["initial_forecast_u", "initial_forecast_v"]].rename(
            {"initial_forecast_u": "water_u", "initial_forecast_v": "water_v"}) - pred[1] for pred in
                      list_forecast_hindcast]
        list_forecast = [pred[0][["initial_forecast_u", "initial_forecast_v"]].rename(
            {"initial_forecast_u": "water_u", "initial_forecast_v": "water_v"}) for pred in
            list_forecast_hindcast]
        array_error_per_time = []
        array_forecast_per_time = []
        for t in range(len(list_error[0]["time"])):
            array_error_per_time.append(
                np.array([list_day.isel(time=t).assign(
                    magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2) ** 0.5,
                    angle=lambda x: np.arctan2(x.water_v, x.water_u)).to_array().to_numpy() for list_day in
                          list_error]))
            array_forecast_per_time.append(
                np.array([list_day.isel(time=t).assign(
                    magnitude=lambda x: (x.water_u ** 2 + x.water_v ** 2) ** 0.5).to_array().to_numpy() for list_day in
                          list_forecast]))
        array_error_per_time = np.array(array_error_per_time)
        array_forecast_per_time = np.array(array_forecast_per_time)
        # dims: lags x days x dims(u,v,magn) x lon x lat

        # -----------------------------------------
        # -----------------------------------------
        # PLOTTINGS:
        # -----------------------------------------
        # -----------------------------------------

        print("plotting")

        # -----------------------------------------
        # QQ plots
        # -----------------------------------------

        # No qq-plot on the magnitude
        n_dims = 2  # array_per_time.shape[2]
        max_lags_to_plot = 12
        # Only consider the first 24 predictions
        factor = 3
        n_col = min(len(array_error_per_time), max_lags_to_plot) // factor
        n_row = factor
        array_error_per_time = array_error_per_time[:n_col * n_row]
        array_forecast_per_time = array_forecast_per_time[:n_col * n_row]
        for dim_current in range(n_dims):
            fig, ax = plt.subplots(n_row, n_col)  # , sharey=True, sharex=True)
            for i in range(len(array_error_per_time)):
                x = self._remove_nan_and_flatten(array_error_per_time[i][:, dim_current])
                ax_now = ax[i // n_col, i % n_col]
                sm.qqplot(x, line='s', ax=ax_now)  # , fit=True)
                ax_now.set_title(f'lag:{i},dim:{["u", "v", "magn"][dim_current]}')
            plt.legend(f'current: {["u", "v", "magn"][dim_current]}')
            plt.show()

        # Plot the general qq plots
        fig, ax = plt.subplots(2, 1)
        dims_first = np.moveaxis(array_error_per_time, 2, 0)
        sm.qqplot(self._remove_nan_and_flatten(dims_first[0]), line='s', ax=ax[0])
        sm.qqplot(self._remove_nan_and_flatten(dims_first[1]), line='s', ax=ax[1])
        values_flattened = dims_first.reshape(len(dims_first), -1)

        # -----------------------------------------
        # Error wrt forecast magnitude
        # -----------------------------------------
        print("Error wrt forecast magnitude")
        dims_to_plot = ["u", "v", "magnitude"]
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
        for i, dim in enumerate(dims_to_plot):
            ax[i].scatter(array_forecast_per_time[:, :, i].flatten(),
                          array_error_per_time[:, :, i].flatten(), s=0.04)
            ax[i].set_title(f"Dimension:{dim}")
        m = max(-np.nanmin(array_forecast_per_time), np.nanmax(array_forecast_per_time))
        plt.xlim(-m, m)
        plt.ylim(np.nanmin(array_error_per_time), np.nanmax(array_error_per_time))
        plt.xlabel("forecast")
        plt.ylabel("error")

        print("Error wrt forecast magnitude same aspect")
        dims_to_plot = ["u", "v", "magnitude"]
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        for i, dim in enumerate(dims_to_plot):
            x, y = array_forecast_per_time[:, :, i].flatten(), array_error_per_time[:, :, i].flatten()
            nans = np.isnan(x + y)
            x, y = x[~nans], y[~nans]
            local_ax = ax[i % 2, i // 2]
            local_ax.scatter(x, y, s=0.04)

            # Print the regression line
            b, a = np.polyfit(x, y, deg=1)
            xseq = np.linspace(np.nanmin(x), np.nanmax(x), num=100)
            local_ax.plot(xseq, a + b * xseq, color="k", lw=2.5);

            local_ax.set_title(f"Dimension:{dim}, slope: {b}")
            local_ax.set_aspect("equal")
        m = max(-np.nanmin(array_forecast_per_time), np.nanmax(array_forecast_per_time))
        plt.xlim(-m, m)
        plt.ylim(np.nanmin(array_error_per_time), np.nanmax(array_error_per_time))
        plt.xlabel("forecast")
        plt.ylabel("error")

        # -----------------------------------------
        # Describe stats
        # -----------------------------------------
        df_describe = pd.DataFrame(values_flattened.T, columns=["u", "v", "magn", "angle"]).dropna()
        print("\n\n\n\nDETAIL about the whole dataset\n", df_describe.describe())

        # -----------------------------------------
        # Histogram
        # -----------------------------------------
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        df_describe.hist("u", ax=ax[0], bins=80, density=1)
        self._add_normal_on_plot(df_describe["u"], ax[0])
        df_describe.hist("v", ax=ax[1], bins=80, density=True)
        self._add_normal_on_plot(df_describe["v"], ax[1])
        # for each lag:
        stat_lags = []
        for i in range(len(array_error_per_time)):
            stat_lags.append(pd.DataFrame(np.array([e[i].flatten() for e in dims_first]).T,
                                          columns=["u", "v", "magn", "angle"]).describe())

        dims_to_plot_ci = ["u", "v", "magn"]
        fig, ax = plt.subplots(len(dims_to_plot_ci), 1, sharex=True)
        for i, axi in enumerate(ax):
            dim = dims_to_plot_ci[i]
            mean = np.array([s[dim]["mean"] for s in stat_lags])
            x = list(range(len(mean)))
            # 95% CI
            ci = np.array([s[dim]["std"] * 1.96 / np.sqrt(s[dim]["count"]) for s in stat_lags])
            axi.plot(mean, color="black", lw=.7)
            axi.fill_between(x, mean - ci, mean + ci, color='blue', alpha=.1)
            axi.set_title(f"Dimension: {dim}")
        plt.xlabel("Lag")
        plt.ylabel("error")
        fig.suptitle("Evolution of the dim values with respect to the lag")
        plt.plot()

        # Try to fit the best distribution
        # for i, dim in enumerate(["u", "v"]):
        #     f = Fitter(self.__remove_nan_and_flatten(dims_first[i][1]), timeout=3000, bins=50)
        #     f.fit()
        #
        #     f.summary()

        print("test")

    def print_errors(self):
        errors_magn, stds = [], []
        for i, observ in enumerate(self.observer.prediction_model.measurement_locations):
            error = self.observer.prediction_model.measured_current_errors[i]
            lon, lat, timestamp = observ
            t = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
            error_computed, std = self.observer.get_data_at_point(lon, lat, t)
            error_computed = error_computed[0]
            errors_magn.append((error_computed[0] - error[0]) ** 2 + (error_computed[1] - error[1]) ** 2)
            stds.append(std)
            print("Error:", (lon, lat, t), error_computed, error)
        print(
            f"std mean over all the observations:{np.array(stds).mean()}\n mean of error magnitude over all the overvations {np.array(errors_magn).mean()}\nall magnitudes:{np.array(errors_magn)}")

    def visualize_area(self, interval_lon, interval_lat, interval_time, number_days_forecasts=50, use_NN=False):
        if not self.problem_factory.has_problems_remaining():
            raise StopIteration()

        # Setup the problem
        problem = self.problem_factory.next_problem()
        problem.start_state.date_time = interval_time[0] - datetime.timedelta(days=1)
        controller = NaiveToTargetController(problem)
        self.last_observation = self.arena.reset(problem.start_state)
        self.observer.reset()

        list_forecast_hindcast = []
        list_datetime_when_new_forecast_files = [self.arena.platform.state.date_time]
        list_gp_output = []
        list_NN_output = []
        radius = self.variables.get("radius_area_around_platform", 1)
        lags_in_sec = self.variables.get("number_steps_to_predict", 12) * 3600
        self.__step_simulation(controller, fit_model=True)
        for i in range(number_days_forecasts * 24):
            print(i + 1, "/", number_days_forecasts * 24, self.arena.platform.state.date_time)
            intervals_lon_lat_time = interval_lon, interval_lat, [
                self.arena.platform.state.date_time + datetime.timedelta(
                    hours=1),
                self.arena.platform.state.date_time + datetime.timedelta(
                    hours=25)]
            intervals_lon_lat_time_with_margin = [interval_lon[0] - 1, interval_lon[1] + 1], \
                                                 [interval_lat[0] - 1, interval_lat[1] + 1], [
                                                     self.arena.platform.state.date_time - datetime.timedelta(hours=1),
                                                     self.arena.platform.state.date_time + datetime.timedelta(
                                                         hours=25)]

            # intervals_lon_lat_time = interval_lon, interval_lat, [ti + datetime.timedelta(hours=i) for ti in
            #                                                       interval_time]
            # while self.last_observation.platform_state.date_time < dims[-1][0]:
            #     print("skip!")
            #     action_to_apply = controller.get_action(self.last_observation)
            #     self.last_observation = self.arena.step(action_to_apply)
            file = self.last_file_used
            fc = self.__step_simulation(controller, fit_model=True, dim_lon_lat=24 if use_NN else None,
                                        lon_lat_time_intervals_to_get=intervals_lon_lat_time, use_NN=use_NN)
            if use_NN:
                fc, fc_NN = fc
                list_NN_output.append(fc_NN)
            if file != self.last_file_used:
                list_datetime_when_new_forecast_files.append(self.arena.platform.state.date_time)

            # list_gp_output.append(self.observer.get_data_over_area(*intervals_lon_lat_time,
            #                                                        spatial_resolution=0.025,
            #                                                        temporal_resolution=self.variables.get(
            #                                                            "delta_between_predictions_in_sec", None))[
            #                           ["error_u", "error_v", "std_error_u", "std_error_v"]])

            list_gp_output.append(self.observer.get_data_around_platform(self.last_observation, radius, lags_in_sec))
            hc = self.arena.ocean_field.hindcast_data_source. \
                get_data_over_area(*intervals_lon_lat_time_with_margin,
                                   temporal_resolution=self.variables.get("delta_between_predictions_in_sec", None))
            hc = hc.assign_coords(depth=fc.depth.to_numpy().tolist())
            obj = PredictionsAndGroundTruthOverArea(fc, hc)
            list_forecast_hindcast.append((obj.predictions_over_area, obj.ground_truth))
        obj.visualize_initial_error(list_forecast_hindcast,
                                    tuple_trajectory_history_new_files=(
                                        self.arena.state_trajectory[:, :3], list_datetime_when_new_forecast_files),
                                    radius_area=self.variables.get("radius_area_around_platform", None),
                                    gp_outputs=list_gp_output, NN_outputs=list_NN_output)

    def run_all_problems(self, max_number_problems_to_run=None,
                         compute_for_all_radius_and_lag=False, create_plots=False) -> Tuple[
        List[Dict[str, any]], List[Dict[str, any]], Dict[str, any], Dict[str, list]]:
        """Run all the problems that were specified when then ExperimentRunner object was created consecutively and
        provide the metrics computed for each problem

        Returns: List of dictionaries where the i-th item of the list is a dict with all the metrics computed for the
        i-th problem.
        """

        results = []
        results_per_h = []
        results_grids = defaultdict(list)
        i = 0

        # Run all the problems and gather all the data
        while self.problem_factory.has_problems_remaining() and (
                type(max_number_problems_to_run) != int or max_number_problems_to_run > i):
            try:
                print(f"Problem no:{i}/{max_number_problems_to_run}")
                i += 1
                res_tuple = self.run_next_problem(compute_for_all_radius_and_lag=compute_for_all_radius_and_lag)
                if compute_for_all_radius_and_lag:
                    res, res_per_h, res_grid_dict = res_tuple
                    for key, val in res_grid_dict.items():
                        results_grids[key].append(val)
                else:
                    res, res_per_h = res_tuple
                results.append(res)
                results_per_h.append(res_per_h)
                if create_plots:
                    self.__create_plots(results[-1], results_per_h[-1])
            except OutOfGridError:
                print(f"Problem {i} skipped, was too close of the area boundaries.")
            except ValueError as e:
                print(f"Error timestamp: {e}")

        # reformat the results
        merged = defaultdict(list)
        for key in results[-1].keys():
            merged[key] = np.array([np.nanmean(np.array(r[key])) for r in results if key in r])
        for key in results_per_h[-1].keys():
            if key == "time":
                continue
            for r in results_per_h:
                all_hours = r[key].mean(axis=0)
                for h in range(len(all_hours)):
                    merged[key + "_" + str(h)] += [all_hours[h]]

        if compute_for_all_radius_and_lag:
            return results, results_per_h, merged, self.list_dates_when_new_files, dict(results_grids)
        else:
            return results, results_per_h, merged, self.list_dates_when_new_files

    def has_next_problem(self):
        return self.problem_factory.has_problems_remaining()

    def run_next_problem(self, compute_metrics_per_h=False, get_inputs_and_outputs=False,
                         compute_for_all_radius_and_lag: Optional[bool] = False) -> Dict[str, Any]:
        """ Run the next problem. It creates a NaiveToTargetController based on the problem, reset the arena and
        observer. Gather "number_burnin_steps" observations without fitting the model and then start predicting the
        model at each timestep and evaluate for that timestep the prediction compared to the hindcast.

        Returns:
            dictionary with the pairs: (metric_name, 1d array containing the output from that metric at each timestep)
        """
        if not self.has_next_problem():
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
            print(f"End of burnin ({burnin} steps)")
        print("start predicting")

        metrics_names = []
        metrics_per_h_names = []
        metrics = []
        metrics_per_h = []
        results = []
        metric_grids = defaultdict(list)

        if get_inputs_and_outputs:
            inputs_and_outputs = ([], [], [], [])
        dim_lon_lat = None
        if get_inputs_and_outputs or compute_for_all_radius_and_lag:
            if "radius_area_around_platform" in self.variables:
                dim_lon_lat = self.variables["radius_area_around_platform"] * 2 * POINTS_PER_DEGREE + 1
            else:
                dim_lon_lat = 25

        # Now we run the simulation
        i = 0
        try:
            for i in range(self.variables["number_steps_prediction"]):
                # if i % 20 == 0:
                #    print(f"step:{i}/{self.variables['number_steps_prediction']}")
                print(f"step:{i}/{self.variables['number_steps_prediction']}")

                model_prediction = self.__step_simulation(controller, fit_model=True, dim_lon_lat=dim_lon_lat)

                # get ground truth
                ground_truth = self.arena.ocean_field.hindcast_data_source.get_data_over_area(
                    *self.__get_lon_lat_time_intervals(ground_truth=True),
                    temporal_resolution=self.variables.get("delta_between_predictions_in_sec", None))
                ground_truth = ground_truth.assign_coords(depth=model_prediction.depth.to_numpy().tolist())
                # compute the metrics and log the results
                self.last_prediction_ground_truth = PredictionsAndGroundTruthOverArea(model_prediction, ground_truth)
                if get_inputs_and_outputs:
                    inputs_and_outputs[0].append(
                        self.last_prediction_ground_truth.predictions_over_area.to_array().to_numpy())
                    inputs_and_outputs[1].append(self.last_prediction_ground_truth.ground_truth.to_array().to_numpy())
                    inputs_and_outputs[2].append(self.observer.prediction_model.measurement_locations)
                    inputs_and_outputs[3].append(self.observer.prediction_model.measured_current_errors)
                else:
                    results.append(self.last_prediction_ground_truth)
                    name_metrics = self.variables["metrics"] if "metrics" in self.variables.keys() else None
                    directions = self.variables.get("direction_current", ["uv"])
                    metric = self.last_prediction_ground_truth. \
                        compute_metrics(name_metrics, directions=directions,
                                        compute_for_all_radius_and_lag=compute_for_all_radius_and_lag)
                    if compute_metrics_per_h:
                        metric_per_hour = self.last_prediction_ground_truth.compute_metrics(name_metrics,
                                                                                            directions=directions,
                                                                                            per_hour=True)
                    if compute_for_all_radius_and_lag:
                        metric_grid = dict()
                        for key in list(metric.keys()):
                            if key.endswith("_all_lags_and_radius") or key.endswith("_per_lag_and_radius"):
                                metric_grid[key] = metric[key]
                                del metric[key]
                        for key, val in metric_grid.items():
                            metric_grids[key].append(val)
                    # for key in metric_grids.keys()
                    #     del metric

                    # In case of Nan only
                    if compute_metrics_per_h:
                        if not len(metrics_per_h_names):
                            metrics_per_h_names = ["time"] + list(metric_per_hour.keys())

                        values_per_h = np.stack(list(metric_per_hour.values()))
                        # times_per_h = np.array([self.last_observation.platform_state.date_time.timestamp() + int(datetime.timedelta(
                        #    hours=i).seconds * 1e6) for i in range(values_per_h.shape[1])], ndmin=2)
                        times_per_h = np.array(
                            [self.last_observation.platform_state.date_time.timestamp()] * values_per_h.shape[1],
                            ndmin=2)
                        metrics_per_h.append(np.concatenate((times_per_h, values_per_h)))

                    # add the metric computed to the list of the corresponding metric list from the dictionary
                    if not len(metrics_names):
                        metrics_names = ["time", "mean_magnitude_forecast"] + list(metric.keys())
                    # add the time and mean magnitude to the metrics list
                    metrics.append(np.insert(np.fromiter(metric.values(), dtype=float), 0, np.array(
                        [self.last_observation.platform_state.date_time.timestamp(),
                         (np.array(self.last_prediction_ground_truth.initial_forecast ** 2).sum(
                             axis=-1) ** 0.5).mean()])))
        except ValueError as e:
            print(f"exception caught at step: {i}. Problem stopped. \n{e}")
        except Exception as e:
            print(f"exception caught: {e}")
        finally:
            if get_inputs_and_outputs:
                return [np.concatenate(elem, axis=1) for elem in inputs_and_outputs[:2]], inputs_and_outputs[2], \
                       inputs_and_outputs[3]
            else:
                metrics = np.array(metrics)
                metrics_per_h = np.array(metrics_per_h)
                if metrics.ndim == 1:
                    res0 = {name: np.array([m[i] for m in metrics if len(m) > 2]) for i, name in
                            enumerate(metrics_names)}
                else:
                    res0 = {name: metrics[:, i] for i, name in enumerate(metrics_names)}
                res1 = {name: metrics_per_h[:, i, :] for i, name in enumerate(metrics_per_h_names)}
                if compute_for_all_radius_and_lag:
                    for key in metric_grids.keys():
                        d1 = min([a.shape[0] for a in metric_grids[key]])
                        d2 = min([a.shape[1] for a in metric_grids[key]])
                        print(d1, max([a.shape[0] for a in metric_grids[key]]), d2,
                              max([a.shape[1] for a in metric_grids[key]]))
                        metric_grids[key] = np.array([a[:d1, :d2] for a in metric_grids[key]]).mean(axis=0)
                    return res0, res1, dict(metric_grids)
                else:
                    return res0, res1

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

    def __step_simulation(self, controller: Controller, fit_model: bool = True,
                          lon_lat_time_intervals_to_get: Optional[Tuple] = None, dim_lon_lat=None, use_NN=False,
                          get_list_files_used=False) -> \
            Union['xarray', None]:
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

        # Keep track of the last file used for forecasts

        if get_list_files_used and self.last_file_used is None or (
                self.last_file_used != self.last_observation.forecast_data_source.DataArray.encoding['source']):
            # print(f"clearing observations old files:\n{self.last_file_used}\n"
            #       f"{self.last_observation.forecast_data_source.DataArray.encoding['source']}\n"
            #       f"new observation:{self.last_observation.platform_state.date_time}")

            self.last_file_used = self.last_observation.forecast_data_source.DataArray.encoding['source']
            self.list_dates_when_new_files.append(self.last_observation.platform_state.date_time)

        predictions = None
        if fit_model:
            self.observer.fit()

            radius_space = self.variables.get("radius_area_around_platform", 1)
            lags_in_second = self.variables.get("number_steps_to_predict", 12) * 3600
            predictions = self.observer.get_data_around_platform(
                self.last_observation.platform_state.to_spatio_temporal_point(), radius_space,
                lags_in_second=lags_in_second,
                temporal_resolution=self.variables.get("delta_between_predictions_in_sec", 3600))

        if use_NN:
            predictions_NN = self.observer.evaluate_neural_net(predictions)
            return predictions, predictions_NN

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
