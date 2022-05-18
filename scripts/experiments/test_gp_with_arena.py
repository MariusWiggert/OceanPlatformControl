import datetime
import math
import time
from typing import Tuple, Optional, Dict, Any

import dateutil
import matplotlib
import matplotlib.cm as cmx
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from numpy import ndarray
from xarray import DataArray

from ocean_navigation_simulator.env.Arena import ArenaObservation, Arena
from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.HighwayProblemFactory import HighwayProblemFactory
from ocean_navigation_simulator.env.Observer import Observer
from ocean_navigation_simulator.env.PlatformState import PlatformState, SpatialPoint
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.controllers.Controller import Controller
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.models.GaussianProcess import OceanCurrentGP
from ocean_navigation_simulator.env.models.OceanCurrentsModel import OceanCurrentsModel
from ocean_navigation_simulator.env.utils import units
from ocean_navigation_simulator.env.utils.units import Distance
from ocean_navigation_simulator.utils.calc_fmrc_error import calc_vector_corr_over_time

# %% Load the config file
# simulation_config = "config_simulation_GP"
simulation_config = "config_real_data_GP"
with open(f'ocean_navigation_simulator/env/scenarios/{simulation_config}.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    variables = config["testing"]
    plots_config = variables["plots"]
# %% Setup the constants
_DELTA_TIME_NEW_PREDICTION = datetime.timedelta(seconds=variables["delta_between_predictions_in_sec"])

_DURATION_SIMULATION = datetime.timedelta(seconds=variables["duration_simulation_in_sec"])
_NUMBER_STEPS = int(math.ceil(_DURATION_SIMULATION.total_seconds() / _DELTA_TIME_NEW_PREDICTION.total_seconds()))
_N_BURNIN_PTS = variables[
    "number_burnin_steps"]  # 100 # Number of minimum pts we gather from a platform to use as observations
_MAX_STEPS_PREDICTION = variables["maximum_steps"]
IGNORE_WARNINGS = variables["ignore_warnings"]

DISPLAY_INTERMEDIARY_3D_PLOTS = plots_config["display_intermediary_3d_plots"]
WAIT_KEYBOARD_INPUT_FOR_PLOT = plots_config["wait_keyboard_input_to_continue"]
DURATION_INTERACTION_WITH_PLOTS_IN_SEC = plots_config["duration_interaction_with_plots_in_sec"]
_N_STEPS_BETWEEN_PLOTS = plots_config["num_steps_between_plots"]
_MARGIN_AREA_PLOT = Distance(deg=plots_config["margin_for_area_to_plot_in_deg"])


# Todo: develop an abstract class Model.
def get_model(arena: Arena, config_file: Dict[str, Any]) -> OceanCurrentsModel:
    print(config_file)
    dic_model = config_file["model"]
    if "gaussian_process" in dic_model:
        return OceanCurrentGP(arena.ocean_field, dic_model["gaussian_process"])
    raise NotImplementedError("Only Gaussian Process supported right now")


# Todo: Not working, to fix (with real_errors)
def plot3d(expected_errors: DataArray, real_errors: Optional[DataArray] = None,
           platform_old_positions: Optional[np.ndarray] = None,
           stride: Optional[int] = 1,
           x_y_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    # create list to plot
    data, data_error = [], []
    for j in range(len(expected_errors)):
        elem = expected_errors.isel(time=j)
        x, y = np.meshgrid(elem["lon"], elem["lat"])
        z = np.sqrt(elem["water_u"].to_numpy() ** 2 + elem["water_v"].to_numpy() ** 2)
        times = elem["time"].to_numpy()
        data.append((times, {"X": x, "Y": y, "Z": z}))  # , "colors": "rgy"[j], "alpha": .25})

        if real_errors is not None:
            elem2 = real_errors.isel(time=j)
            x, y = np.meshgrid(elem["lon"], elem["lat"])
            z = np.sqrt(elem["water_u"].to_numpy() ** 2 + elem["water_v"].to_numpy() ** 2)
            times = elem["time"].to_numpy()
            data_error.append((times, {"X": x, "Y": y, "Z": z}))  # , "colors": "rgy"[j], "alpha": .25})

    def update_line(idx: int, x_y_lim: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        # Plot the wireframe of the GP
        ax.clear()
        t, dic = data[idx]
        ax.plot_wireframe(**dic)
        plt.draw()
        if len(data_error):
            t2, dic2 = data_error[idx]
            ax.plot_wireframe(**dic2, alpha=0.5, color='g')
        ax.set_title(
            f"Error prediction\nCurrent time:{np.datetime_as_string(data[0][0], unit='s')}\n" +
            f"Prediction time:{np.datetime_as_string(t, unit='s')}")
        if x_y_lim:
            ax.set_xlim(xmin=x_y_lim[0][0], xmax=x_y_lim[0][1])
            ax.set_ylim(ymin=x_y_lim[1][0], ymax=x_y_lim[1][1])
        plt.draw()

        # Plot the trajectory of the boat:
        if platform_old_positions is not None:
            map_color = 'winter'
            cm = plt.get_cmap(map_color)
            cs = platform_old_positions[::stride, 2]
            c_norm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
            scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)
            magnitude_error = np.sqrt(
                platform_old_positions[::stride, 3] ** 2 + platform_old_positions[::stride, 4] ** 2)
            ax.plot(platform_old_positions[::stride, 0], platform_old_positions[::stride, 1],
                    magnitude_error, c='black')
            ax.scatter(platform_old_positions[::stride, 0], platform_old_positions[::stride, 1],
                       magnitude_error,
                       marker=".",
                       c=scalar_map.to_rgba(cs))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    update_line(0, x_y_intervals)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_zlabel("error")

    # plot the current platform position
    # lon, lat = platformstate.to_spatial_point().lon, platformstate.to_spatial_point().lat
    # z = np.linspace(data.sel(u_v="u").min(), data.sel(u_v="u").max(), 2)
    # ax.plot([lon.deg] * 2, [lat.deg] * 2, z, 'go--', linewidth=2, markersize=12)

    # Plot the Slider
    plt.subplots_adjust(bottom=.25)
    ax_slider = plt.axes([.1, .1, .8, .05], facecolor='teal')
    slider = Slider(ax_slider, "Time", valmin=0, valmax=len(data) - 1, valinit=0, valstep=1)
    slider.on_changed(lambda j: update_line(j, x_y_intervals))

    plt.show()
    if WAIT_KEYBOARD_INPUT_FOR_PLOT:
        keyboard_click = False
        while not keyboard_click:
            print("waiting for keyboard input to continue")
            keyboard_click = plt.waitforbuttonpress()
            print("continue scenario")
    else:
        plt.pause(DURATION_INTERACTION_WITH_PLOTS_IN_SEC)


def visualize_currents(platform_state: PlatformState, arena: Arena, model_predictions: ndarray, x_y_intervals: ndarray,
                       trajectory_platform: Optional[ndarray] = None) -> None:
    print("visualize_currents")
    ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(platform_state.date_time, *x_y_intervals,
                                                                            return_ax=True)
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(trajectory_platform[:, 0], trajectory_platform[:, 1], color='y', marker='+')
    ax.set_xlim(x_lim), ax.set_ylim(y_lim)
    ax.set_title("True currents")

    ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(platform_state.date_time, *x_y_intervals,
                                                                            return_ax=True)
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(trajectory_platform[:, 0], trajectory_platform[:, 1], color='y', marker='+')
    ax.set_xlim(x_lim), ax.set_ylim(y_lim)
    ax.set_title("Initial forecasts")

    ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(platform_state.date_time, *x_y_intervals,
                                                                            return_ax=True,
                                                                            error_to_incorporate=model_predictions)
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(trajectory_platform[:, 0], trajectory_platform[:, 1], color='y', marker='+')
    ax.set_xlim(x_lim), ax.set_ylim(y_lim)
    ax.set_title("Improved forecasts")


def get_error_ocean_current_vector(forecast: OceanCurrentVector,
                                   true_current: OceanCurrentVector) -> OceanCurrentVector:
    return forecast.subtract(true_current)


def get_error_forecasts_true_currents_for_area(forecasts: np.ndarray, true_currents: np.ndarray) -> np.ndarray:
    return forecasts - true_currents


def get_prediction_currents(forecast: ndarray, error: ndarray) -> ndarray:
    return forecast - error


def get_forecast_currents(ground_truth: ndarray, error_forecast: ndarray) -> ndarray:
    return ground_truth + error_forecast


# Todo: fix if using it because it's not adapted for the shape
def compute_weighted_rmse(true_currents: ndarray, predictions: ndarray, lambda_time: float = .9,
                          lambda_space: float = .96) -> float:
    # Compute the weights for weighted rmse:
    weights_time = np.array([lambda_time ** j for j in range(len(predictions))])

    def get_weights_space(len_array):
        a = [lambda_space ** j for j in range((len_array + 1) // 2)]
        # Todo: check when the slicing should be [1:] instead (not urgent)
        return np.array((a[::-1] + a)[:len_array])

    weights_x = get_weights_space(true_currents.shape[1])
    weights_y = get_weights_space(true_currents.shape[2])
    x_y = weights_x.reshape((len(weights_x), 1)).dot(weights_y.reshape((1, len(weights_y))))
    weights = weights_time.reshape((len(weights_time), 1)).dot(x_y.reshape((1, -1))).reshape(
        (len(weights_time), *x_y.shape, 1))
    weights = np.repeat(weights, predictions.shape[-1], axis=-1)
    return np.sqrt(np.average(np.array((true_currents - predictions) ** 2), weights=weights))


def get_losses(true_currents: ndarray, improved_predictions_currents: ndarray, initial_forecast_error: ndarray,
               centered_around_platform: bool = True) -> dict[str, Any]:
    losses = dict()

    cond_not_nan = np.repeat(
        np.logical_not(np.isnan((true_currents - improved_predictions_currents).sum(axis=-1)))[..., np.newaxis],
        repeats=2, axis=-1)
    if cond_not_nan.sum() != np.prod(true_currents.shape):
        print("Nans in the area:", (1 - cond_not_nan.sum() / np.prod(true_currents.shape)) * 100, "%")

    true_currents_flattened = true_currents[cond_not_nan].reshape((-1, 2))
    predictions_currents_flattened = improved_predictions_currents[cond_not_nan].reshape((-1, 2))
    initial_forecast_error_flattened = initial_forecast_error[cond_not_nan].reshape((-1, 2))
    rmse = np.sqrt(np.mean((true_currents_flattened - predictions_currents_flattened) ** 2))
    losses["rmse"] = rmse

    # todo: MODIFY THAT
    # pearsonr_correlation = pearsonr(predictions_currents.flatten(), true_currents.flatten())
    # losses["pearson_correlation"]: pearsonr_correlation

    # Compute R2
    losses["r2"] = 1 - ((true_currents_flattened - predictions_currents_flattened) ** 2).sum() / (
            initial_forecast_error_flattened ** 2).sum()
    print("variance model:", ((true_currents_flattened - predictions_currents_flattened) ** 2).sum())
    print("variance forecast:", (initial_forecast_error_flattened ** 2).sum(), " avg:",
          (initial_forecast_error_flattened ** 2).mean())

    # Compute Vector correlation
    # Todo: check if mean() is relevant here
    losses["vector_correlation"] = calc_vector_corr_over_time(improved_predictions_currents, true_currents,
                                                              sigma_diag=0, remove_nans=True).mean()
    losses["vector_correlation_ref"] = calc_vector_corr_over_time(
        get_forecast_currents(true_currents, initial_forecast_error), true_currents,
        sigma_diag=0, remove_nans=True).mean()
    losses["vector_correlation_ratio"] = losses["vector_correlation"] / losses["vector_correlation_ref"]

    # Todo: fix compute_weighted_rmse if we want to use it
    # if centered_around_platform:
    #    losses["weighted rmse"] = compute_weighted_rmse(true_currents, predictions_currents)
    return losses


# %%
np.set_printoptions(precision=5)

if IGNORE_WARNINGS:
    import warnings

    warnings.filterwarnings('ignore')
# %% Create the problem and the arena
problemFactory = HighwayProblemFactory(
    [(SpatialPoint(Distance(meters=0), Distance(meters=0)), SpatialPoint(Distance(deg=10), Distance(deg=10)))])
arenas = []
problems = []
success = []
use_real_data = variables["use_real_data"]
if use_real_data:
    # Specify Problem
    init_pos = variables["initial_position"]
    x_0 = PlatformState(lon=units.Distance(deg=init_pos["lon_in_deg"]), lat=units.Distance(deg=init_pos["lat_in_deg"]),
                        date_time=dateutil.parser.isoparse(init_pos["datetime"]))
    target_point = variables["target"]
    x_T = SpatialPoint(lon=units.Distance(deg=target_point["lon_in_deg"]),
                       lat=units.Distance(deg=target_point["lat_in_deg"]))
    problem = Problem(start_state=x_0, end_region=x_T, target_radius=target_point["radius_in_m"])
else:
    problem = problemFactory.next_problem()
arena = ArenaFactory.create(scenario_name=variables["scenario_used"])

arenas.append(arena)
# %% Create the model, the observer, the controller
print(problem.start_state)
arena_obs = arena.reset(problem.start_state)
controller = NaiveToTargetController(problem=problem)

model = get_model(arena, config)
observer = Observer(model, arena, config["observer"], )
trajectory_platform = []
x_y_interval = arena.get_lon_lat_time_interval(end_region=problem.end_region)[:2]
print("end region:", problem.end_region, " x_y_interval:", x_y_interval)

# %% print the number of points in the area around the platform
forecast = observer.get_forecast_around_platform(arena_obs.platform_state)
print(
    f"Number of points around the platform: {len(forecast['lat'])}x{len(forecast['lon'])}={len(forecast['lat']) * len(forecast['lon'])}")

# %% visualize the all ocean
if plots_config.get("plot_initial_world_map"):
    ax = arena.ocean_field.forecast_data_source. \
        plot_data_at_time_over_area(arena_obs.platform_state.date_time,
                                    *observer.get_area_around_platform(arena_obs.platform_state, Distance(deg=0.5)),
                                    error_to_incorporate=arena.ocean_field.hindcast_data_source,
                                    return_ax=True, vmin=-1, vmax=1)

    ax.set_title("difference currents")
    plt.pause(1)


def evaluate_predictions(current_platform_state: PlatformState, observer_platform: Observer,
                         area_to_evaluate: Optional[np.ndarray] = None) -> Dict[str, Any]:
    forecasts = observer_platform.get_forecast_around_platform(current_platform_state, x_y_intervals=area_to_evaluate)
    ground_truth = observer_platform.get_ground_truth_around_platform(current_platform_state,
                                                                      x_y_intervals=area_to_evaluate)

    # put the current dimension as the last one
    # gt_np = ground_truth.transpose("time", "lon", "lat").to_array().to_numpy()
    gt_np = np.moveaxis(ground_truth.transpose("time", "lon", "lat").to_array().to_numpy(), 0, -1)
    # gt_np = gt_np.reshape(*gt_np.shape[1:], gt_np.shape[0])

    # forecasts_np = forecasts.transpose("time", "lon", "lat").to_array().to_numpy()
    forecasts_np = np.moveaxis(forecasts.transpose("time", "lon", "lat").to_array().to_numpy(), 0, -1)
    forecasts_error_predicted, _ = observer.evaluate(current_platform_state, x_y_interval=area_to_evaluate)
    forecasts_error_np = forecasts_error_predicted.to_array(dim="u_v").transpose("time", "lon", "lat", "u_v").to_numpy()
    predictions = get_prediction_currents(forecasts_np, forecasts_error_np)
    initial_error = get_error_forecasts_true_currents_for_area(forecasts_np, gt_np)

    # Plot mat:
    if plots_config.get("plot_map_when_evaluate", False):
        f = plt.figure(3)
        f.clear()
        ax = f.add_subplot()
        a = gt_np[0, :, :, 0]
        masked_array = np.ma.array(a, mask=np.isnan(a))
        cmap = matplotlib.cm.jet
        cmap.set_bad('white', 1.)
        ax.imshow(masked_array, interpolation='nearest', cmap=cmap)
        # plt.pause(1)

    return get_losses(gt_np, predictions, initial_error)


def step_simulation(controller_simulation: Controller, observer_platform: Observer, observation: ArenaObservation,
                    points_trajectory: list[np.ndarray], area_to_evaluate: Optional[np.ndarray] = None,
                    fit_model: bool = True) \
        -> Tuple[ArenaObservation, Optional[DataArray], Optional[DataArray]]:
    action_to_apply = controller_simulation.get_action(observation)
    new_observation = arena.step(action_to_apply)
    measured_current = new_observation.true_current_at_state
    forecast_current_at_platform_pos = new_observation.forecast_data_source.get_data_at_point(
        new_observation.platform_state.to_spatio_temporal_point())
    error_at_platform_position = get_error_ocean_current_vector(forecast_current_at_platform_pos, measured_current)
    observer_platform.observe(new_observation.platform_state, error_at_platform_position)

    # Give as input to observer: position and forecast-groundTruth
    # TODO: modify input to respect interface design
    mean_error, error_std = None, None
    if fit_model:
        observer.fit()
        mean_error, error_std = observer.evaluate(new_observation.platform_state, x_y_interval=area_to_evaluate)

    points_trajectory.append(
        (np.concatenate((new_observation.platform_state.to_spatio_temporal_point(), error_at_platform_position),
                        axis=0)))
    return new_observation, mean_error, error_std


# %% First we only gather observations
start = time.time()

for i in range(_N_BURNIN_PTS):
    arena_obs, _, _ = step_simulation(controller, observer, arena_obs, trajectory_platform, fit_model=False)
    position = arena_obs.platform_state.to_spatial_point()
    # print(
    #     f"Burnin step:{i + 1}/{_N_BURNIN_PTS}, position platform:{(position.lat.m, position.lon.m)}")

print("end of burnin time.")
# %%  Prediction at each step of the error
print("start predicting")
means, stds = [], []
times = []
r2_losses = []
correlation_losses = []
correlation_losses_ref = []
ratio_correlation = []
i = 0

# %% Now we run the algorithm
n_steps = min(_NUMBER_STEPS, _MAX_STEPS_PREDICTION)
for i in range(n_steps):
    arena_obs, forecasts_error, forecast_std = step_simulation(controller, observer, arena_obs, trajectory_platform,
                                                               area_to_evaluate=None)
    forecast_error, forecast_std = forecasts_error.to_array(), forecast_std.to_array()
    forecast_error_around_platform, forecast_std_around_platform = observer.evaluate(arena_obs.platform_state)

    print("step {}/{}: forecasted mean:{}, abs mean:{}, forecasted_std:{}"
          .format(i + 1,
                  n_steps,
                  forecasts_error.mean(dim=["lon", "lat", "time"]).to_numpy(),
                  abs(forecasts_error).mean(dim=["lon", "lat", "time"]).to_numpy(),
                  forecast_std.mean().item()))

    # Compute the losses
    # print("last 3 positions trajectory: ", trajectory_platform[-3:])
    # print("whole grid losses: ", evaluate_predictions(arena_obs.platform_state, observer, x_y_interval))
    res_around_platform = evaluate_predictions(arena_obs.platform_state, observer)
    print("around platform grid losses: ", res_around_platform)
    times.append(arena_obs.platform_state.date_time)
    r2_losses.append(res_around_platform["r2"])
    correlation_losses.append(res_around_platform["vector_correlation"])
    correlation_losses_ref.append(res_around_platform["vector_correlation_ref"])
    ratio_correlation.append(res_around_platform["vector_correlation_ratio"])

    # Plot the currents

    means.append(forecasts_error.to_array().to_numpy())
    stds.append(forecast_std.to_numpy())
    if DISPLAY_INTERMEDIARY_3D_PLOTS and (i % _N_STEPS_BETWEEN_PLOTS == 0 or i == n_steps - 1):
        # Plot values at other time instances
        mean, _ = observer.evaluate(arena_obs.platform_state)
        x_y_interval_platform = observer.get_area_around_platform(arena_obs.platform_state,
                                                                  margin=_MARGIN_AREA_PLOT)
        plot3d(mean, x_y_intervals=x_y_interval_platform)
        real_errors = get_error_forecasts_true_currents_for_area(
            observer.get_forecast_around_platform(arena_obs.platform_state),
            observer.get_ground_truth_around_platform(arena_obs.platform_state))
        print("real errors:", real_errors)
        plot3d(real_errors, x_y_intervals=x_y_interval_platform)

# %% Once the loop is over, we print the results.


print("duration: ", time.time() - start)
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(times, r2_losses, label="r2_loss")
axs[0, 0].set_title("r2_loss")
axs[1, 0].plot(times, ratio_correlation, label="vector_correlation_ratio")
axs[1, 0].set_title("vector correlation_ratio")
axs[0, 1].plot(times, correlation_losses,
               label="vector_correl")
axs[0, 1].set_title("vector correlation model")
axs[1, 1].plot(times, correlation_losses_ref,
               label="vector_correl_ref")
axs[1, 1].set_title("vector correlation reference")
plt.legend()

print(
    f"Mean r2_loss: {np.array(r2_losses).mean()}, mean r2_loss_without first 100 items: {np.array(r2_losses[100:]).mean()}")

# %% Visualize the currents

visualize_currents(arena_obs.platform_state, arena, forecast_error_around_platform,
                   observer.get_area_around_platform(arena_obs.platform_state),
                   trajectory_platform=np.array(trajectory_platform)[:, :2])

# %% Plot the mean prediction around
mean, _ = observer.evaluate(arena_obs.platform_state)
forecasts = observer.get_forecast_around_platform(arena_obs.platform_state)
hindcasts = observer.get_ground_truth_around_platform(arena_obs.platform_state)
real_errors = get_error_forecasts_true_currents_for_area(forecasts, hindcasts)
x_y_interval_platform = observer.get_area_around_platform(arena_obs.platform_state, margin=_MARGIN_AREA_PLOT)
plot3d(mean,  # platform_old_positions=np.array(trajectory_platform),
       x_y_intervals=x_y_interval_platform)
plt.figure()
plot3d(real_errors,  # platform_old_positions=np.array(trajectory_platform),
       x_y_intervals=x_y_interval_platform)
print("over")
