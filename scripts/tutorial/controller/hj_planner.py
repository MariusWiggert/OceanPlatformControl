import datetime
import logging

import matplotlib.pyplot as plt
from tqdm import tqdm

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import set_arena_loggers

set_arena_loggers(logging.DEBUG)

# Initialize the Arena (holds all data sources and the platform, everything except controller)
arena = ArenaFactory.create(
    scenario_name="gulf_of_mexico_HYCOM_hindcast",
    # Note: uncomment this to download the hindcast data if not already locally available
    t_interval=[
        datetime.datetime(2022, 8, 23, 12, 0, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 8, 25, 12, 0, tzinfo=datetime.timezone.utc)]
)
# we can also download the respective files directly to a temp folder, then t_interval needs to be set
# % Specify Navigation Problem
x_0 = PlatformState(
    lon=units.Distance(deg=-82.5),
    lat=units.Distance(deg=18.6),
    date_time=datetime.datetime(2022, 8, 24, 15, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-85.3), lat=units.Distance(deg=19.9))

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.1,
)
#%%
observation = arena.reset(platform_state=problem.start_state)
# %% Plot the problem on the map
t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(), x_T=x_T, deg_around_x0_xT_box=1, temp_horizon_in_s=3600
)

ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()
# %% Instantiate and runm the HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "multi-time-reach-back",
    "n_time_vector": 200,
    "closed_loop": True,  # to run closed-loop or open-loop
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

# % Run reachability planner
observation = arena.reset(platform_state=x_0)
action = planner.get_action(observation=observation)
#%% let's test saving and re-loading!
type(planner.reach_times)
# %% Various plotting of the reachability computations
planner.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    plot_in_h=True,
)
# planner.plot_reachability_snapshot_over_currents(rel_time_in_seconds=0, granularity_in_h=5, time_to_reach=False)
# planner.plot_reachability_animation(time_to_reach=False, granularity_in_h=5, filename="test_reach_animation.mp4")
# planner.plot_reachability_animation(time_to_reach=True, granularity_in_h=5, with_opt_ctrl=True,
#                                     filename="test_reach_animation_w_ctrl.mp4", forward_time=True)
#%% save planner state and reload it
# Save it to a folder
planner.save_planner_state("saved_planner/")
#%% Testing access to azure blob storage from outside
import xarray as xr
# ds_gcs = xr.open_dataset(
#     "azure://dev-rcone/fs/seaweed-control/devseaweedrc1/download_speed_test//logs/test_array.nc",
#     # "gcs://<bucket-name>/path.zarr",
#     backend_kwargs={
#         "storage_options": {"project": "<project-name>", "token": None}
#     },
#     engine="zarr",
# )
url = 'https://devrconestore01.blob.core.windows.net/dev-rcone/fs/seaweed-control/devseaweedrc1/download_speed_test//logs/test_array.nc?sig=qkwAEpCYeA1frI%2FyeRzN0sms85Fs6pqf6jIUA59j4NQ%3D&se=2023-01-09T22%3A27%3A22Z&sv=2018-11-09&sp=r&sr=b'
xr.open_dataset(url)#, engine='h5netcdf')
#%% save to a zarr file
import numpy as np
import xarray as xr
ds = xr.Dataset(
    {"foo": (("x", "y", "t"), np.ones((200,200,100)))},
    coords={
        "x": np.arange(200),
        "y": np.arange(200),
        "t": np.arange(100)
    },
)
ds.to_zarr("array_test.zarr")
#%%
from netCDF4 import Dataset
dataset = Dataset(url)

#%% Saving the planner state to pickle files and reloading from there
import time
start = time.time()
import datetime
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.utils import units

print("import takes:", time.time() - start)
mid = time.time()
# Load it from the folder
loaded_planner = HJReach2DPlanner.from_saved_planner_state(folder="saved_planner/", problem=None)
# loaded_planner.last_data_source = arena.ocean_field.hindcast_data_source
# observation = arena.reset(platform_state=x_0)
# loaded_planner._update_current_data(observation=observation)
# planner = loaded_planner
print("load planner takes:", time.time() - mid)
mid = time.time()
x_0 = PlatformState(
    lon=units.Distance(deg=-82.5),
    lat=units.Distance(deg=23.7),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
print("init platform state takes:", time.time() - mid)
mid = time.time()
action = loaded_planner._get_action_from_plan(state=x_0)
print("_get_action_from_plan takes:", time.time() - mid)
print("took:", time.time() - start)
print("action:", action)
# Takes 5.7s (mostly imports 3.4s and get_action 2s)
#%% Prep to save as posix
folder="saved_planner/"
import pickle
with open(folder + "reach_times_posix.pickle", "wb") as file:
    pickle.dump(loaded_planner.reach_times + loaded_planner.current_data_t_0, file)
#%% Option 2: saving it in the type.
import time
start = time.time()
import pickle
from scipy.interpolate import interp1d
from hj_reachability.finite_differences import upwind_first
import jax.numpy as jnp
import jax.lax
print("import takes:", time.time() - start)
mid = time.time()
folder="saved_planner/"
with open(folder + "all_values.pickle", "rb") as file:
    all_values = pickle.load(file)
with open(folder + "reach_times_posix.pickle", "rb") as file:
    reach_times_posix = pickle.load(file)
with open(folder + "grid.pickle", "rb") as file:
    grid = pickle.load(file)
print("load takes:", time.time() - mid)
mid = time.time()
# Issue now is, times is relative times, need to have posix times...
state = [-82.5, 23.7, 1637755200.0]
# now get actions
# Note: this can probably be done more efficiently e.g. by initializing the function once?
val_at_t = interp1d(reach_times_posix, all_values, axis=0, kind='linear')(state[2]).squeeze()
print("1D takes:", time.time() - mid)
mid = time.time()
# Step 1: get center approximation of gradient at current point x
left_grad_values, right_grad_values = grid.upwind_grad_values(upwind_first.WENO3, values=val_at_t)
grad_at_x_cur = grid.interpolate(values=0.5 * (left_grad_values + right_grad_values), state=state[:2])
print("grad computation takes:", time.time() - mid)
mid = time.time()
# get optimal action
alpha = jax.lax.atan2(grad_at_x_cur[1], grad_at_x_cur[0]) + jnp.pi
print("atan takes:", time.time() - mid)
print("took:", time.time() - start)
print(jnp.array([1.0, alpha]))
#%% Try doing everything with numpy
# Grad at x is DeviceArray([-0.7753288 ,  0.28574902], dtype=float32)
#%% Option 2: saving it in the type.
import time
start = time.time()
import pickle
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import numpy as np
# from hj_reachability.finite_differences import upwind_first
# import jax.numpy as jnp
# import jax.lax
print("import takes:", time.time() - start)
mid = time.time()
folder="saved_planner/"
with open(folder + "all_values.pickle", "rb") as file:
    all_values = pickle.load(file)
with open(folder + "reach_times_posix.pickle", "rb") as file:
    reach_times_posix = pickle.load(file)
with open(folder + "grid.pickle", "rb") as file:
    grid = pickle.load(file)
print("load takes:", time.time() - mid)
mid = time.time()
# Issue now is, times is relative times, need to have posix times...
state = [-82.5, 23.7, 1637755200.0]
# now get actions
# Note: this can probably be done more efficiently e.g. by initializing the function once?
val_at_t = interp1d(reach_times_posix, all_values, axis=0, kind='linear')(state[2]).squeeze()
print("1D takes:", time.time() - mid)
mid = time.time()
# Step 1: get center approximation of gradient at current point x
# left_grad_values, right_grad_values = grid.upwind_grad_values(upwind_first.WENO3, values=val_at_t)
# grad_at_x_cur = grid.interpolate(values=0.5 * (left_grad_values + right_grad_values), state=state[:2])
# get grad with numpy
grad = np.gradient(val_at_t, 0.02, 0.02) #grid.spacings()
# x_grad = interp1d(reach_times_posix, all_values, axis=0, kind='linear')(state[2]).squeeze()
x_grad = RegularGridInterpolator((grid.coordinate_vectors[0], grid.coordinate_vectors[1]), grad[0])(state[:2])
y_grad = RegularGridInterpolator((grid.coordinate_vectors[0], grid.coordinate_vectors[1]), grad[1])(state[:2])
# print("x_grad ", x_interp(state[:2]))
# print("y_grad ", y_interp(state[:2]))
print("grad computation takes:", time.time() - mid)
mid = time.time()
# get optimal action
# alpha = jax.lax.atan2(grad_at_x_cur[1], grad_at_x_cur[0]) + jnp.pi
alpha = np.arctan2(y_grad, x_grad) + np.pi
print("atan takes:", time.time() - mid)
print("took:", time.time() - start)
print([1.0, alpha])
#%%
# # Step 2: get u_opt and d_opt
# def optimal_control(self, state, time, grad_value):
#     """Computes the optimal control realized by the HJ PDE Hamiltonian."""
#     uOpt = jnp.array(1.0)
#     # angle of px, py vector of gradient
#     alpha = jax.lax.atan2(grad_value[1], grad_value[0])
#     # if min, go against the gradient direction
#     if self.control_mode == "min":
#         alpha = alpha + jnp.pi
#     return jnp.array([uOpt, alpha])
#%% the get_action_loop







# %% Let the controller run closed-loop within the arena (the simulation loop)
# Note this runs the sim for a fixed time horizon (does not terminate when reaching the goal)
for i in tqdm(range(int(3600 * 24 * 5 / 600))):  # 5 days
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem)
#%% Animate the trajectory
arena.animate_trajectory(problem=problem, temporal_resolution=7200)
