import datetime

import matplotlib
import numpy as np
import time
import math
import xarray as xr
from matplotlib import pyplot as plt, cm
import matplotlib.cm as cmx

from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.Observer import Observer
from ocean_navigation_simulator.env.Platform import Platform, PlatformState
from ocean_navigation_simulator.env.PlatformState import SpatialPoint
from ocean_navigation_simulator.env.controllers.UnmotorizedController import UnmotorizedController
from ocean_navigation_simulator.env.Problem import Problem

from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.data_sources.OceanCurrentSource.OceanCurrentVector import OceanCurrentVector
from ocean_navigation_simulator.env.utils import units
from scripts.experiments.class_gp import OceanCurrentGP

# %%
# _DELTA_TIME_NEW_PREDICTION = datetime.timedelta(hours=1)
# _DURATION_SIMULATION = datetime.timedelta(days=3)
_DELTA_TIME_NEW_PREDICTION = datetime.timedelta(seconds=1)
_DURATION_SIMULATION = datetime.timedelta(seconds=72)
_NUMBER_STEPS = int(math.ceil(_DURATION_SIMULATION.total_seconds() / _DELTA_TIME_NEW_PREDICTION.total_seconds()))
#_N_DATA_PTS_MAX_USED = 100
_N_BURNIN_PTS = 30  # 100 # Number of minimum pts we gather from a platform to use as observations
_MAX_STEPS_PREDICTION = 5000
EVAL_ONLY_AROUND_PLATFORM = False
IGNORE_WARNINGS = False


def plot3d(data: xr.DataArray, past_predictions, stride=1, other_plots=None):
    plt.figure()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(*np.meshgrid(data["lat"], data["lon"]), data.sel({'u_v': "u"}).to_numpy())
    for plot in other_plots:
        ax.plot_wireframe(**plot)
    ax.set_xlabel("lat")
    ax.set_ylabel("lon")
    ax.set_zlabel("error")

    # plot the current platform position
    #lon, lat = platformstate.to_spatial_point().lon, platformstate.to_spatial_point().lat
    #z = np.linspace(data.sel(u_v="u").min(), data.sel(u_v="u").max(), 2)
    #ax.plot([lon.deg] * 2, [lat.deg] * 2, z, 'go--', linewidth=2, markersize=12)

    MAP = 'winter'
    cm = plt.get_cmap(MAP)
    cs = past_predictions[::stride, 2]
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    ax.plot(past_predictions[::stride, 0], past_predictions[::stride, 1], past_predictions[::stride, 3],c='black')
    ax.scatter(past_predictions[::stride, 0], past_predictions[::stride, 1], past_predictions[::stride, 3], marker=".",
            c=scalarMap.to_rgba(cs))


    print("show plot")
    plt.show()
    plt.pause(10)


def get_error(forecast: OceanCurrentVector, true_current: OceanCurrentVector) -> OceanCurrentVector:
    return forecast.subtract(true_current)


# %%

np.set_printoptions(precision=2)

if IGNORE_WARNINGS:
    import warnings

    warnings.filterwarnings('ignore')
# %%

arena, platform_state, arena_obs, end_region = ArenaFactory.create(scenario_name='current_highway_GP')
# arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='double_gyre_GP')
# arena, platform_state, observation, end_region = ArenaFactory.create(scenario_name='gulf_of_mexico')


controller = NaiveToTargetController(problem=Problem(
    start_state=platform_state,
    end_region=end_region
))
print("controller created")
# %%
gp = OceanCurrentGP(arena.ocean_field)
observer = Observer(gp, arena)
past_pred = []
print("gp and observer created")
# %% First we only gather observations
start = time.time()

for i in range(_N_BURNIN_PTS):
    action = controller.get_action(arena_obs)
    arena_obs = arena.step(action)
    true_current, forecast_current = arena_obs.true_current_at_state, arena_obs.forecasted_current_at_state
    observer.observe(arena_obs.platform_state, get_error(forecast_current, true_current))
    past_pred.append((np.concatenate((arena_obs.platform_state.to_spatio_temporal_point(), get_error(forecast_current, true_current)), axis=0)))
    print("Burnin step:{}/{}, position platform:{}, difference_observation:{}"
          .format(i + 1, _N_BURNIN_PTS, arena_obs.platform_state.to_spatial_point(),
                  get_error(forecast_current, true_current)))
print("end of burnin time.")
#arena.quick_plot(end_region)
# %%  Prediction at each step of the error
print("start predicting")
means_no_noise, stds_no_noise = [], []
means_noise, stds_noise = [], []
i=0
#%%
for i in range(min(_NUMBER_STEPS, _MAX_STEPS_PREDICTION)):
    action = controller.get_action(arena_obs)
    arena_obs = arena.step(action)
    true_current, forecast_current = arena_obs.true_current_at_state, arena_obs.forecasted_current_at_state
    error = get_error(forecast_current, true_current)
    print("error{}, forecast:{}, true_current:{}".format(error, forecast_current, true_current))
    # Give as input to observer: position and forecast-groundTruth
    observer.observe(arena_obs.platform_state, error)
    forecasts_error, forecast_std = observer.fit_and_evaluate(arena_obs.platform_state,
                                                              x_y_intervals=arena.get_lon_lat_interval(
                                                                  end_region=end_region))
    past_pred.append((np.concatenate((arena_obs.platform_state.to_spatio_temporal_point(), error), axis=0)))
    print("\n\nstep {}/{}: noise:{}, forecasted mean:{}, abs mean:{}, forecasted_std:{}"
          .format(i + 1,
                  min(_NUMBER_STEPS, _MAX_STEPS_PREDICTION),
                  error,
                  forecasts_error.mean(dim=["lat", "lon", "time"]).to_numpy(),
                  abs(forecasts_error).mean(dim=["lat", "lon", "time"]).to_numpy(),
                  forecast_std.mean().item()))
    print("query observed position", arena_obs.platform_state.to_spatial_point(),
          (arena_obs.platform_state.date_time - datetime.datetime(1970, 1, 1,
                                                                  tzinfo=datetime.timezone.utc)).total_seconds(),
          " should be:", error,
          " and is: ", observer.evaluate(arena_obs.platform_state.to_spatio_temporal_point()))

    if i % 10 == 0:
        # Plot values at other time instances
        mean, _ = observer.fit_and_evaluate(arena_obs.platform_state,
                                           x_y_intervals=arena.get_lon_lat_interval(
                                               end_region=end_region),
                                           delta=datetime.timedelta(seconds=-20))
        #Add older plots
        other_plots = []
        for j in range(2):
            x, y, z = *np.meshgrid(mean.isel(time=j)["lat"], mean.isel(time=j)["lon"]), mean.isel(time=j).sel({'u_v': "u"}).to_numpy()
            other_plots.append({"X": x,"Y": y, "Z": z, "colors": "rgy"[j], "alpha": .25})

        #Initial plot
        plot3d(forecasts_error.isel(time=0), np.array(past_pred), other_plots=other_plots)
    if error == OceanCurrentVector(u=0, v=0):
        means_no_noise.append(forecasts_error)
        stds_no_noise.append(forecast_std)
    else:
        means_noise.append(forecasts_error)
        stds_noise.append(forecast_std)
    # get_forecasts(observation.platform_state)
    print("end iteration")

#%% Once the loop is over, we print the averages.
print("duration: ", time.time() - start)
# plot3d(forecasts_error[])
if len(means_no_noise):
    print(
        f"average No noise: mean:{np.array(means_no_noise).mean(axis=(0, 1))}, average stds:{np.array(stds_no_noise).mean(axis=(0, 1))}")
if len(means_noise):
    print(
        f"average with noise: mean:{np.array(means_noise).mean(axis=(0, 1))}, average stds:{np.array(stds_noise).mean(axis=(0, 1))}")
# Testing if solar caching or not-caching makes much of a difference
# For 240 steps: without caching 0.056s > with caching: 0.037.
# %%
# arena.do_nice_plot(x_T=np.array([controller.problem.end_region.lon.deg, controller.problem.end_region.lat.deg]))
arena.quick_plot(end_region=end_region)

# %%
print("over")
