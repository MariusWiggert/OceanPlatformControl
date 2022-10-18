import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.Problem import Problem
from ocean_navigation_simulator.utils import units

# # %%
# import xarray as xr
# path = "data/Copernicus_Hindcast/2021_11_20-30_Copernicus.nc"
# folder = "data/Copernicus_Hindcast/"
# from ocean_navigation_simulator.environment.data_sources.OceanCurrentSource.OceanCurrentSource import get_file_dicts
# files_dicts = get_file_dicts(folder)
#%%
# # Step 2: open the respective file as multi dataset
# DataArray = xr.open_mfdataset([h_dict['file'] for h_dict in files_dicts]).isel(depth=0)
# DataArray["time"] = DataArray["time"].dt.round("H")
#
# %%


arena = ArenaFactory.create(scenario_name="multi_reach_fails_test")
#
# % Plot to check if loading worked
t_0 = datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=4)]
x_interval = [-82, -80]
y_interval = [24, 26]
# x_0 = PlatformState(lon=units.Distance(deg=-81.5), lat=units.Distance(deg=23.5), date_time=t_0)
# x_T = SpatialPoint(lon=units.Distance(deg=-80), lat=units.Distance(deg=24.2))
# Plot Hindcast
# arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(time=t_0 + datetime.timedelta(days=2),
#                                                                    x_interval=x_interval, y_interval=y_interval)
# # Plot Forecast at same time
# xarray_out = arena.ocean_field.forecast_data_source.get_data_over_area(t_interval=t_interval, x_interval=x_interval, y_interval=y_interval)
# ax = arena.ocean_field.forecast_data_source.plot_data_from_xarray(time_idx=49, xarray=xarray_out)
# plt.show()
# forecast_data_source = arena.ocean_field.forecast_data_source
# % Specify Problem
x_0 = PlatformState(
    lon=units.Distance(deg=-81.749),
    lat=units.Distance(deg=18.839),
    date_time=datetime.datetime(2021, 11, 24, 12, 10, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-83.178), lat=units.Distance(deg=18.946))
problem = Problem(start_state=x_0, end_region=x_T, target_radius=0.1)
# % Plot the problem function -> To create

# %Instantiate the HJ Planner
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)

specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": 3600 * 5,
    "direction": "multi-time-reach-back",
    "n_time_vector": 100,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 5,
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    "initial_set_radii": [
        0.1,
        0.1,
    ],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    "grid_res": 0.02,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    # 'EVM_threshold': 0.3 # in m/s error when floating in forecasted vs sensed currents
    # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
    "platform_dict": arena.platform.platform_dict,
}
planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
# % Run forward reachability
observation = arena.reset(platform_state=x_0)
# #%% Plot the problem on the map
# problem.plot(data_source=arena.ocean_field.hindcast_data_source)
# plt.show()
# #%% Various plotting of the reachability computations
# planner.plot_reachability_snapshot(rel_time_in_seconds=0, granularity_in_h=5,
#                                    alpha_color=1, time_to_reach=True, fig_size_inches=(12, 12), plot_in_h=True)
# planner.plot_reachability_snapshot_over_currents(rel_time_in_seconds=0, granularity_in_h=5, time_to_reach=True)
# planner.plot_reachability_animation(time_to_reach=True)
# ##%% run closed-loop simulation
# for i in tqdm(range(int(3600*24*7/600))):  # 720*10 min = 5 days
#     action = planner.get_action(observation=observation)
#     observation = arena.step(action)
##%% New loop with the observer inside
from ocean_navigation_simulator.ocean_observer.Observer import Observer

# create observer
observer_config = {
    "life_span_observations_in_sec": 86400,  # 24 * 3600
    "model": {
        "gaussian_process": {
            "sigma_noise_squared": 0.000005,
            # 3.6 ** 2 = 12.96
            "sigma_exp_squared": 100,  # 12.96
            "kernel": {
                "scaling": {"latitude": 1, "longitude": 1, "time": 10000},  # [m]  # [m]  # [s]
                "type": "matern",
                "parameters": {"length_scale_bounds": "fixed"},
            },
            "time_horizon_predictions_in_sec": 3600,
        }
    },
}
observer = Observer(observer_config)
action = planner.get_action(observation=observation)
# planner.plot_reachability_animation(time_to_reach=True, granularity_in_h=5, filename="new_true.mp4")
# %% Now use the observer for planning instead of the
for i in tqdm(range(int(3600 * 24 * 5 / 600))):  # 720*10 min = 5 days
    # modify the observation, now coming from the observer for data to the planner
    observer.observe(observation)
    observation.forecast_data_source = observer
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
#%% Check if the piping works!
# query forecast for a specific point
# Step 1: get the x,y,t bounds for current position, goal position and settings.
from ocean_navigation_simulator.environment.data_sources.DataSources import (
    DataSource,
)

t_interval, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
    x_0=observation.platform_state.to_spatio_temporal_point(),
    x_T=planner.problem.end_region,
    deg_around_x0_xT_box=planner.specific_settings["deg_around_xt_xT_box"],
    temp_horizon_in_s=planner.specific_settings["T_goal_in_seconds"],
)

# get the data subset from the file
data_xarray_fmrc = observation.forecast_data_source.get_data_over_area(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
    spatial_resolution=planner.specific_settings["grid_res"],
)

#%% Same thing from the observer
observer.observe(observation)
observation.forecast_data_source = observer

#%%
t_interval, y_interval, x_interval = DataSource.convert_to_x_y_time_bounds(
    x_0=observation.platform_state.to_spatio_temporal_point(),
    x_T=planner.problem.end_region,
    deg_around_x0_xT_box=planner.specific_settings["deg_around_xt_xT_box"],
    temp_horizon_in_s=planner.specific_settings["T_goal_in_seconds"],
)

# get the data subset from the file
data_xarray_observer = observation.forecast_data_source.get_data_over_area(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
    spatial_resolution=planner.specific_settings["grid_res"],
)

#%%
data_xarray_true = arena.ocean_field.hindcast_data_source.get_data_over_area(
    x_interval=x_interval,
    y_interval=y_interval,
    t_interval=t_interval,
    spatial_resolution=planner.specific_settings["grid_res"],
)
#%%
data_xarray_true_interp = data_xarray_true.interp_like(
    data_xarray_fmrc, method="linear", assume_sorted=False, kwargs=None
)
#%%
(data_xarray_observer["initial_forecast_u"] - data_xarray_true_interp["water_u"]).max()
import yaml

# %% Major TODOS:
# 1) Modify Forecast Source to take in the fmrc index queried for accessing the area and for a point
# 2) Finally set-up a logger and logging mechanisms! Otherwise this is just a mess...
# %% Re-init Arena
from ocean_navigation_simulator.environment.data_sources.OceanCurrentField import (
    OceanCurrentField,
)

with open("scenarios/multi_reach_fails_test.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

arena.ocean_field = OceanCurrentField(
    casadi_cache_dict=config["casadi_cache_dict"],
    hindcast_source_dict=config["ocean_dict"]["hindcast"],
    forecast_source_dict=config["ocean_dict"]["forecast"],
    use_geographic_coordinate_system=True,
)
# %% Plot the trajectory
x_int, y_int, t_interval = arena.get_lon_lat_time_interval(end_region=problem.end_region, margin=1)
ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=t_interval[0], x_interval=x_int, y_interval=y_int, return_ax=True
)
arena.plot_state_trajectory_on_map(ax=ax)
arena.plot_control_trajectory_on_map(ax=ax)
problem.plot(ax=ax, data_source=None)
plt.show()
# %%
# Looks like with GP is almost the same as without. That's weird.
# Need to investigate why! First ideas: check if the xarray is properly created in observer.
# For the critical parts (where it actuates down vs up: investigate why that happens looking at the HJ function)...
