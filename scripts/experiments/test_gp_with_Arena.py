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
from ocean_navigation_simulator.env.Observer import Observer
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
_N_STEPS_BETWEEN_PLOTS = 15
EVAL_ONLY_AROUND_PLATFORM = False
IGNORE_WARNINGS = False

WAIT_KEYBOARD_INPUT_FOR_PLOT = False
INTERVAL_PAUSE_PLOTS = 5


def plot3d(expected_errors: DataArray, platform_old_positions: np.ndarray, stride: int = 1):
    # create list to plot
    data = []
    for j in range(len(expected_errors)):
        elem = expected_errors.isel(time=j)
        x, y, z = *np.meshgrid(elem["lat"], elem["lon"]), elem.sel({'u_v': "u"}).to_numpy()
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
    # ax.plot_wireframe(*np.meshgrid(data["lat"], data["lon"]), data.sel({'u_v': "u"}).to_numpy())
    update_line(0)
    # for plot in other_plots:
    #     ax.plot_wireframe(**plot)
    ax.set_xlabel("lat")
    ax.set_ylabel("lon")
    ax.set_zlabel("error")

    # plot the current platform position
    # lon, lat = platformstate.to_spatial_point().lon, platformstate.to_spatial_point().lat
    # z = np.linspace(data.sel(u_v="u").min(), data.sel(u_v="u").max(), 2)
    # ax.plot([lon.deg] * 2, [lat.deg] * 2, z, 'go--', linewidth=2, markersize=12)

    # Plot the Slider
    plt.subplots_adjust(bottom=.25)
    ax_slider = plt.axes([.1, .1, .8, .05], facecolor='teal')
    slider = Slider(ax_slider, "Time", valmin=0, valmax=len(data) - 1, valinit=0, valstep=1)
    slider.on_changed(update_line)

    print("show plot")
    plt.show()
    if WAIT_KEYBOARD_INPUT_FOR_PLOT:
        keyboard_click = False
        while not keyboard_click:
            keyboard_click = plt.waitforbuttonpress()
    else:
        plt.pause(INTERVAL_PAUSE_PLOTS)


def get_error(forecast: OceanCurrentVector, true_current: OceanCurrentVector) -> OceanCurrentVector:
    return forecast.subtract(true_current)


def get_prediction_currents(forecast: ndarray, error: ndarray) -> ndarray:
    return forecast - error


# todo: compute on true currents estimation or on error???
def get_losses(true_currents: ndarray, predictions: ndarray) -> dict[str, Any]:
    rmse = np.sqrt(np.mean((true_currents - predictions) ** 2))
    pearsonr_correlation = pearsonr(predictions.flatten(), true_currents.flatten())
    return {"rmse": rmse, "correlation": pearsonr_correlation}


# %%
np.set_printoptions(precision=2)

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
print("gp and observer created")


# %%
def evaluate_predictions(platform_state: PlatformState, area: Optional[np.ndarray] = None) -> Dict[str, Any]:
    forecasts = observer.get_forecast_around_platform(platform_state, x_y_intervals=area)
    ground_truth = observer.get_ground_truth_around_platform(platform_state, x_y_intervals=area)
    # put the current dimension as the last one
    gt_np = ground_truth.transpose("time", "lon", "lat").to_array().to_numpy()
    gt_np = gt_np.reshape(*gt_np.shape[1:], gt_np.shape[0])
    forecasts_np = forecasts.transpose("time", "lon", "lat").to_array().to_numpy()
    forecasts_np = forecasts_np.reshape(*forecasts_np.shape[1:], forecasts_np.shape[0])
    forecasts_error_np = forecasts_error.transpose("time", "lon", "lat", "u_v").to_numpy()
    predictions = get_prediction_currents(forecasts_np, forecasts_error_np)
    return get_losses(gt_np, predictions)


def step_simulation(controller_simulation: Controller, observation: ArenaObservation,
                    points_trajectory: list[np.ndarray], area: np.ndarray, fit_model: bool = True) \
        -> Tuple[ArenaObservation, Optional[DataArray], Optional[DataArray]]:
    action_to_apply = controller_simulation.get_action(observation)
    observation = arena.step(action_to_apply)
    measured_current = observation.true_current_at_state
    forecast_current_at_platform_pos = observation.forecast_data_source.get_data_at_point(
        observation.platform_state.to_spatio_temporal_point())
    error_at_platform_position = get_error(forecast_current_at_platform_pos, measured_current)
    observer.observe(observation.platform_state, error_at_platform_position)

    # Give as input to observer: position and forecast-groundTruth
    # TODO: modify input to respect interface design
    mean_error, error_std = None, None
    if fit_model:
        observer.fit()
        mean_error, error_std = observer.evaluate(observation.platform_state, x_y_interval=area)

    points_trajectory.append(
        (np.concatenate((observation.platform_state.to_spatio_temporal_point(), error_at_platform_position), axis=0)))
    return observation, mean_error, error_std


# %% First we only gather observations
start = time.time()

for i in range(_N_BURNIN_PTS):
    arena_obs, _, _ = step_simulation(controller, arena_obs, trajectory_platform, x_y_interval, fit_model=False)
    platform_state = arena_obs.platform_state
    print(
        f"Burnin step:{i + 1}/{_N_BURNIN_PTS}," +
        f"position platform:{(platform_state.to_spatial_point().lat.m, platform_state.to_spatial_point().lon.m)}")

print("end of burnin time.")
# %%  Prediction at each step of the error
print("start predicting")
# means_no_noise, stds_no_noise = [], []
# means_noise, stds_noise = [], []
means, stds = [], []
i = 0

# %%
n_steps = min(_NUMBER_STEPS, _MAX_STEPS_PREDICTION)
for i in range(n_steps):
    arena_obs, forecasts_error, forecast_std = step_simulation(controller, arena_obs, trajectory_platform,
                                                               x_y_interval)

    print("step {}/{}: forecasted mean:{}, abs mean:{}, forecasted_std:{}"
          .format(i + 1,
                  n_steps,
                  forecasts_error.mean(dim=["lon", "lat", "time"]).to_numpy(),
                  abs(forecasts_error).mean(dim=["lon", "lat", "time"]).to_numpy(),
                  forecast_std.mean().item()))

    # Compute the losses
    print("whole grid losses: ", evaluate_predictions(platform_state, x_y_interval))

    if i % _N_STEPS_BETWEEN_PLOTS == 0 or i == n_steps - 1:
        # Plot values at other time instances
        mean, _ = observer.evaluate(arena_obs.platform_state, x_y_interval)
        plot3d(mean, np.array(trajectory_platform))
    # if error == OceanCurrentVector(u=0, v=0):
    #     means_no_noise.append(forecasts_error)
    #     stds_no_noise.append(forecast_std)
    # else:
    #     means_noise.append(forecasts_error)
    #     stds_noise.append(forecast_std)
    means.append(forecasts_error)
    stds.append(forecast_std)

# %% Once the loop is over, we print the averages.
print("duration: ", time.time() - start)
# plot3d(forecasts_error[])
# if len(means_no_noise):
#     print(
#         f"average No noise: mean:{np.array(means_no_noise).mean(axis=(0, 1))}, average stds:{np.array(stds_no_noise).mean(axis=(0, 1))}")
# if len(means_noise):
#     print(
#        f"average with noise: mean:{np.array(means_noise).mean(axis=(0, 1))}, average stds:{np.array(stds_noise).mean(axis=(0, 1))}")
if len(means):
    print(f"average mean:{np.array(means).mean(axis=(0, 1))}, average stds:{np.array(stds).mean(axis=(0, 1))}")

# %%
print("over")
