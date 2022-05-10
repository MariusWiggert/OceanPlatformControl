import datetime
import math
import time
from typing import Tuple, Optional, Dict, Any, List

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

WAIT_KEYBOARD_INPUT_FOR_PLOT = True
INTERVAL_PAUSE_PLOTS = 5


def plot3d(expected_errors: DataArray, platform_old_positions: np.ndarray, stride: int = 1,
           x_y_intervals: Optional[Tuple[List, List]] = None):
    # create list to plot
    data = []
    for j in range(len(expected_errors)):
        elem = expected_errors.isel(time=j)
        x, y, z = *np.meshgrid(elem["lon"], elem["lat"]), elem.sel({'u_v': "u"}).to_numpy()
        times = elem["time"].to_numpy()
        data.append((times, {"X": x, "Y": y, "Z": z}))  # , "colors": "rgy"[j], "alpha": .25})

    def update_line(idx):
        # Plot the wirefram of the GP
        ax.clear()
        time, dic = data[idx]
        ax.plot_wireframe(**dic)
        ax.set_title(
            f"Error prediction\nCurrent time:{np.datetime_as_string(data[0][0], unit='s')}\n Prediction time:{np.datetime_as_string(time, unit='s')}")
        plt.draw()

        # Plot the trajectory of the boat:
        map = 'winter'
        cm = plt.get_cmap(map)
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
    update_line(0)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_zlabel("error")

    if x_y_intervals:
        plt.xlim(x_y_intervals[0])
        plt.ylim(x_y_intervals[1])

    # plot the current platform position
    # lon, lat = platformstate.to_spatial_point().lon, platformstate.to_spatial_point().lat
    # z = np.linspace(data.sel(u_v="u").min(), data.sel(u_v="u").max(), 2)
    # ax.plot([lon.deg] * 2, [lat.deg] * 2, z, 'go--', linewidth=2, markersize=12)

    # Plot the Slider
    plt.subplots_adjust(bottom=.25)
    ax_slider = plt.axes([.1, .1, .8, .05], facecolor='teal')
    slider = Slider(ax_slider, "Time", valmin=0, valmax=len(data) - 1, valinit=0, valstep=1)
    slider.on_changed(update_line)

    plt.show()
    if WAIT_KEYBOARD_INPUT_FOR_PLOT:
        keyboard_click = False
        while not keyboard_click:
            print("wainting for keyboard input to continue")
            keyboard_click = plt.waitforbuttonpress()
            print("continue scenario")
    else:
        plt.pause(INTERVAL_PAUSE_PLOTS)


def get_error(forecast: OceanCurrentVector, true_current: OceanCurrentVector) -> OceanCurrentVector:
    return forecast.subtract(true_current)


def get_prediction_currents(forecast: ndarray, error: ndarray) -> ndarray:
    return forecast - error


# todo: compute on true currents estimation or on error???
def get_losses(true_currents: ndarray, predictions: ndarray, centered_around_platform: bool = True) -> dict[str, Any]:
    losses = dict()

    rmse = np.sqrt(np.mean((true_currents - predictions) ** 2))
    losses["rmse"] = rmse

    # todo: MODIFY THAT
    pearsonr_correlation = pearsonr(predictions.flatten(), true_currents.flatten())
    losses["pearson_correl"]: pearsonr_correlation

    if centered_around_platform:
        # Compute the weights for weighted rmse:
        lambda_time = .9
        weights_time = np.array([lambda_time ** j for j in range(len(predictions))])
        lambda_space = .96

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
        losses["weighted rmse"] = np.sqrt(np.average(np.array((true_currents - predictions) ** 2), weights=weights))
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
    if area_to_evaluate is None:
        print("\n\n\nplatform position:", current_platform_state.to_spatial_point())
        print("positions forecasts:", forecasts["lon"], forecasts["lat"])
        print(
            f"averages: lon{(forecasts['lon'][0] + forecasts['lon'][-1]) / 2}, lat:{(forecasts['lon'][0] + forecasts['lon'][-1]) / 2}")
    forecasts_error_predicted, _ = observer.evaluate(current_platform_state, x_y_interval=area_to_evaluate)
    forecasts_error_np = forecasts_error_predicted.transpose("time", "lon", "lat", "u_v").to_numpy()
    predictions = get_prediction_currents(forecasts_np, forecasts_error_np)
    return get_losses(gt_np, predictions)


def step_simulation(controller_simulation: Controller, observer_platform: Observer, observation: ArenaObservation,
                    points_trajectory: list[np.ndarray], area_to_evaluate: Optional[np.ndarray] = None,
                    fit_model: bool = True) \
        -> Tuple[ArenaObservation, Optional[DataArray], Optional[DataArray]]:
    action_to_apply = controller_simulation.get_action(observation)
    observation = arena.step(action_to_apply)
    measured_current = observation.true_current_at_state
    forecast_current_at_platform_pos = observation.forecast_data_source.get_data_at_point(
        observation.platform_state.to_spatio_temporal_point())
    error_at_platform_position = get_error(forecast_current_at_platform_pos, measured_current)
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
    # print("around platform grid losses: ", evaluate_predictions(arena_obs.platform_state, observer))

    means.append(forecasts_error)
    stds.append(forecast_std)
    if i % _N_STEPS_BETWEEN_PLOTS == 0 or i == n_steps - 1:
        # Plot values at other time instances
        mean, _ = observer.evaluate(arena_obs.platform_state, None)

        x_y_interval_platform = get_intervals_position_around_platform(arena_obs.platform_state,
                                                                       margin=_MARGIN_AREA_PLOT)
        plot3d(mean, np.array(trajectory_platform),
               x_y_intervals=x_y_interval_platform)

# %% Once the loop is over, we print the averages.
print("duration: ", time.time() - start)
if len(means):
    print(f"average mean:{np.array(means).mean(axis=(0, 1))}, average stds:{np.array(stds).mean(axis=(0, 1))}")

# %%
print("over")
