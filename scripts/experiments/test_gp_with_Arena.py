import datetime
import math
import time
from typing import Tuple, Optional, Dict, Any

import matplotlib
import matplotlib.cm as cmx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from numpy import ndarray
from scipy.stats import pearsonr
from xarray import DataArray

from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.HighwayProblemFactory import HighwayProblemFactory
from ocean_navigation_simulator.env.Observer import Observer, get_intervals_position_around_platform
from ocean_navigation_simulator.env.PlatformState import SpatialPoint, PlatformState
from ocean_navigation_simulator.env.controllers.Controller import Controller
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.utils.units import Distance
from ocean_navigation_simulator.utils.calc_fmrc_error import calc_vector_corr_over_time
from scripts.experiments.class_gp import OceanCurrentGP

# %%
# _DELTA_TIME_NEW_PREDICTION = datetime.timedelta(hours=1)
# _DURATION_SIMULATION = datetime.timedelta(days=3)
_DELTA_TIME_NEW_PREDICTION = datetime.timedelta(seconds=1)

_DURATION_SIMULATION = datetime.timedelta(seconds=72)
_NUMBER_STEPS = int(math.ceil(_DURATION_SIMULATION.total_seconds() / _DELTA_TIME_NEW_PREDICTION.total_seconds()))
# _N_DATA_PTS_MAX_USED = 100
_N_BURNIN_PTS = 30  # 100 # Number of minimum pts we gather from a platform to use as observations
_MAX_STEPS_PREDICTION = 5000
_N_STEPS_BETWEEN_PLOTS = 5
_MARGIN_AREA_PLOT = Distance(deg=.02)
EVAL_ONLY_AROUND_PLATFORM = False
IGNORE_WARNINGS = True

DISPLAY_3D_PLOTS = False
WAIT_KEYBOARD_INPUT_FOR_PLOT = True
INTERVAL_PAUSE_PLOTS = 5


def plot3d(expected_errors: DataArray, platform_old_positions: np.ndarray, stride: int = 1,
           x_y_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    # create list to plot
    data = []
    for j in range(len(expected_errors)):
        elem = expected_errors.isel(time=j)
        x, y = np.meshgrid(elem["lon"], elem["lat"])
        z = elem.sel({'u_v': "u"}).to_numpy()
        times = elem["time"].to_numpy()
        data.append((times, {"X": x, "Y": y, "Z": z}))  # , "colors": "rgy"[j], "alpha": .25})

    def update_line(idx: int, x_y_lim: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        # Plot the wireframe of the GP
        ax.clear()
        t, dic = data[idx]
        ax.plot_wireframe(**dic)
        ax.set_title(
            f"Error prediction\nCurrent time:{np.datetime_as_string(data[0][0], unit='s')}\n" +
            f"Prediction time:{np.datetime_as_string(t, unit='s')}")
        if x_y_lim:
            ax.set_xlim(xmin=x_y_lim[0][0], xmax=x_y_lim[0][1])
            ax.set_ylim(ymin=x_y_lim[1][0], ymax=x_y_lim[1][1])
        plt.draw()

        # Plot the trajectory of the boat:
        map_color = 'winter'
        cm = plt.get_cmap(map_color)
        cs = platform_old_positions[::stride, 2]
        c_norm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
        scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)
        ax.plot(platform_old_positions[::stride, 0], platform_old_positions[::stride, 1],
                platform_old_positions[::stride, 3], c='black')
        ax.scatter(platform_old_positions[::stride, 0], platform_old_positions[::stride, 1],
                   platform_old_positions[::stride, 3],
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
        plt.pause(INTERVAL_PAUSE_PLOTS)


def get_error_ocean_current_vector(forecast: OceanCurrentVector,
                                   true_current: OceanCurrentVector) -> OceanCurrentVector:
    return forecast.subtract(true_current)


def get_error_forecasts_true_currents_for_area(forecasts: np.ndarray, true_currents: np.ndarray) -> np.ndarray:
    return forecasts - true_currents


def get_prediction_currents(forecast: ndarray, error: ndarray) -> ndarray:
    return forecast - error


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


def get_losses(true_currents: ndarray, predictions_currents: ndarray, error_without_model: ndarray,
               centered_around_platform: bool = True) -> dict[str, Any]:
    losses = dict()

    rmse = np.sqrt(np.mean((true_currents - predictions_currents) ** 2))
    losses["rmse"] = rmse

    # todo: MODIFY THAT
    pearsonr_correlation = pearsonr(predictions_currents.flatten(), true_currents.flatten())
    losses["pearson_correlation"]: pearsonr_correlation

    # Compute R2
    losses["r2"] = 1 - ((true_currents - predictions_currents) ** 2).sum() / (error_without_model ** 2).sum()
    print("variance model:", ((true_currents - predictions_currents) ** 2).sum())
    print("variance forecast:", (error_without_model ** 2).sum(), " avg:", (error_without_model ** 2).mean())

    # Compute Vector correlation
    # Todo: check if mean() is relevant here
    losses["vector_correlation"] = calc_vector_corr_over_time(predictions_currents, true_currents,
                                                              sigma_diag=.00001).mean()

    if centered_around_platform:
        losses["weighted rmse"] = compute_weighted_rmse(true_currents, predictions_currents)
    return losses


# %%
np.set_printoptions(precision=5)

if IGNORE_WARNINGS:
    import warnings

    warnings.filterwarnings('ignore')
# %%

# arena, platform_state, arena_obs, end_region = ArenaFactory.create(scenario_name='current_highway_GP')
# arena, platform_state, arena_obs, end_region = ArenaFactory.create(scenario_name='current_highway_GP')
# arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='double_gyre_GP')
# arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='gulf_of_mexico')

problemFactory = HighwayProblemFactory(
    [(SpatialPoint(Distance(meters=0), Distance(meters=0)), SpatialPoint(Distance(deg=10), Distance(deg=10)))])
arenas = []
problems = []
success = []
# while problemFactory.has_problems_remaining():
#     problem = problemFactory.next_problem()
#     arena = ArenaFactory.create(scenario_name="current_highway_GP")
#     arenas.append(arena)
#     observation = arena.reset(problem.start_state)
#
#     controller = NaiveToTargetController(problem=problem)
#     is_done = False

problem = problemFactory.next_problem()
arena = ArenaFactory.create(scenario_name="current_highway_GP")
arenas.append(arena)
# %%
print(problem.start_state)
arena_obs = arena.reset(problem.start_state)
controller = NaiveToTargetController(problem=problem)
is_done = False

print("controller created")
# %%
gp = OceanCurrentGP(arena.ocean_field)
observer = Observer(gp, arena)
trajectory_platform = []
x_y_interval = arena.get_lon_lat_time_interval(end_region=problem.end_region)[:2]
print("end region:", problem.end_region, " x_y_interval:", x_y_interval)
print("gp and observer created")


# %%
def evaluate_predictions(current_platform_state: PlatformState, observer_platform: Observer,
                         area_to_evaluate: Optional[np.ndarray] = None) -> Dict[str, Any]:
    forecasts = observer_platform.get_forecast_around_platform(current_platform_state, x_y_intervals=area_to_evaluate)
    ground_truth = observer_platform.get_ground_truth_around_platform(current_platform_state,
                                                                      x_y_intervals=area_to_evaluate)
    # put the current dimension as the last one
    gt_np = ground_truth.transpose("time", "lon", "lat").to_array().to_numpy()
    gt_np = gt_np.reshape(*gt_np.shape[1:], gt_np.shape[0])
    forecasts_np = forecasts.transpose("time", "lon", "lat").to_array().to_numpy()
    forecasts_np = forecasts_np.reshape(*forecasts_np.shape[1:], forecasts_np.shape[0])
    forecasts_error_predicted, _ = observer.evaluate(current_platform_state, x_y_interval=area_to_evaluate)
    forecasts_error_np = forecasts_error_predicted.transpose("time", "lon", "lat", "u_v").to_numpy()
    predictions = get_prediction_currents(forecasts_np, forecasts_error_np)
    initial_error = get_error_forecasts_true_currents_for_area(forecasts_np, gt_np)
    return get_losses(gt_np, predictions, initial_error)


def step_simulation(controller_simulation: Controller, observer_platform: Observer, observation: ArenaObservation,
                    points_trajectory: list[np.ndarray], area_to_evaluate: Optional[np.ndarray] = None,
                    fit_model: bool = True) \
        -> Tuple[ArenaObservation, Optional[DataArray], Optional[DataArray]]:
    action_to_apply = controller_simulation.get_action(observation)
    observation = arena.step(action_to_apply)
    measured_current = observation.true_current_at_state
    forecast_current_at_platform_pos = observation.forecast_data_source.get_data_at_point(
        observation.platform_state.to_spatio_temporal_point())
    error_at_platform_position = get_error_ocean_current_vector(forecast_current_at_platform_pos, measured_current)
    observer_platform.observe(observation.platform_state, error_at_platform_position)

    # Give as input to observer: position and forecast-groundTruth
    # TODO: modify input to respect interface design
    mean_error, error_std = None, None
    if fit_model:
        observer.fit()
        mean_error, error_std = observer.evaluate(observation.platform_state, x_y_interval=area_to_evaluate)

    points_trajectory.append(
        (np.concatenate((observation.platform_state.to_spatio_temporal_point(), error_at_platform_position), axis=0)))
    return observation, mean_error, error_std


# %% First we only gather observations
start = time.time()

for i in range(_N_BURNIN_PTS):
    arena_obs, _, _ = step_simulation(controller, observer, arena_obs, trajectory_platform, fit_model=False)
    position = arena_obs.platform_state.to_spatial_point()
    print(
        f"Burnin step:{i + 1}/{_N_BURNIN_PTS}, position platform:{(position.lat.m, position.lon.m)}")

print("end of burnin time.")
# %%  Prediction at each step of the error
print("start predicting")
means, stds = [], []
times = []
r2_losses = []
correlation_losses = []
i = 0

# %% Now we run the algorithm
n_steps = min(_NUMBER_STEPS, _MAX_STEPS_PREDICTION)
for i in range(n_steps):
    arena_obs, forecasts_error, forecast_std = step_simulation(controller, observer, arena_obs, trajectory_platform,
                                                               area_to_evaluate=None)
    forecast_around_platform_error, forecast_around_platform_std = observer.evaluate(arena_obs.platform_state)

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

    means.append(forecasts_error.to_numpy())
    stds.append(forecast_std.to_numpy())
    if DISPLAY_3D_PLOTS and (i % _N_STEPS_BETWEEN_PLOTS == 0 or i == n_steps - 1):
        # Plot values at other time instances
        mean, _ = observer.evaluate(arena_obs.platform_state, None)
        x_y_interval_platform = get_intervals_position_around_platform(arena_obs.platform_state,
                                                                       margin=_MARGIN_AREA_PLOT)
        plot3d(mean, np.array(trajectory_platform),
               x_y_intervals=x_y_interval_platform)

# %% Once the loop is over, we print the averages.
print("duration: ", time.time() - start)
l1 = plt.plot(times, r2_losses, label="r2 losses")
l2 = plt.plot(times, correlation_losses, label="vector correlation averages")
plt.legend()

mean, _ = observer.evaluate(arena_obs.platform_state, None)
x_y_interval_platform = get_intervals_position_around_platform(arena_obs.platform_state,
                                                               margin=_MARGIN_AREA_PLOT)
plot3d(mean, np.array(trajectory_platform),
       x_y_intervals=x_y_interval_platform)
# if len(means):
#    print(f"average mean:{np.array(means).mean(axis=(0, 1))}, average stds:{np.array(stds).mean(axis=(0, 1))}")

# %%
print("over")
