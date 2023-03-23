#%% Run imports
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

import wandb
from ocean_navigation_simulator.controllers.hj_planners.HJBSeaweed2DPlanner import (
    HJBSeaweed2DPlanner,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.SeaweedProblem import (
    SeaweedProblem,
)
from ocean_navigation_simulator.ocean_observer.NoObserver import NoObserver
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import get_c3, set_arena_loggers

%load_ext autoreload
%autoreload 2
# % Initialize
# os.chdir(
#     "/Users/matthiaskiller/Library/Mobile Documents/com~apple~CloudDocs/Studium/Master RCI/Masters Thesis/Code/OceanPlatformControl"
# )
set_arena_loggers(logging.INFO)

scenario_name = "Region_M_HC_as_forecast_daily+monthly_averages_Copernicus_HC_solar_seaweed"  # "Region_3_Copernicus_HC_solar_seaweed"

# Initialize the Arena (holds all data sources and the platform, everything except controller)
# we can also download the respective files directly to a temp folder, then t_interval needs to be set
# % Specify Navigation Problem
# x_0 = PlatformState(
#     lon=units.Distance(deg=-150),
#     lat=units.Distance(deg=30),
#     date_time=datetime.datetime(2022, 7, 24, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-120), lat=units.Distance(deg=70))

#% init weights & biases
with open(f"config/arena/{scenario_name}.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# wandb.init(project="Long Horizon Seaweed Maximization", entity="ocean-platform-control", config=config)


#%
# x_0 = PlatformState(
#     lon=units.Distance(deg=-83.1),
#     lat=units.Distance(deg=23.2),
#     date_time=datetime.datetime(2021, 11, 23, 12, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-83.1), lat=units.Distance(deg=23.2))
x_0 = PlatformState(
    lon=units.Distance(deg=-81.06714984983319),
    lat=units.Distance(deg=-28.61748961935156),
    date_time=datetime.datetime(2022, 3, 5, 1, 22, 15, tzinfo=datetime.timezone.utc),
)
# %%
arena = ArenaFactory.create(
    scenario_name=scenario_name,
    # t_interval=[
    #     x_0.date_time - datetime.timedelta(days=2),
    #     x_0.date_time + datetime.timedelta(days=32),
    # ],
    # points=[x_0.to_spatial_point()],
)

problem = SeaweedProblem(
    start_state=x_0,
    platform_dict=arena.platform.platform_dict,
)

# %% arena

# %% Plot the problem on the map
t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(),
    x_T=x_0.to_spatial_point(),
    deg_around_x0_xT_box=12,
    temp_horizon_in_s=3600,
)
lat_bnds = [-40,0]
lon_bnds = [-130,-70]
# %%
ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time,
    x_interval=lon_bnds,
    y_interval=lat_bnds,
    return_ax=True,
    quiver_spatial_res=0.6,
    quiver_scale=20,
)
problem.plot(ax=ax)
plt.show()
# %%
ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()
# %%
ax = arena.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=[-130,-70], y_interval=[-40,0], return_ax=True
)
#%%
fig = ax.get_figure()
fig = plt.figure(fig)

plt.savefig("seaweed_trajectory_on_map.svg")
problem.plot(ax=ax)
plt.show()


# %% Instantiate the HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "backward",
    "n_time_vector": 24 * 30,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 10,  # area over which to run HJ_reachability
    "deg_around_xt_xT_box_average": 10,  # area over which to run HJ_reachability for average data
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 2,
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    "grid_res": 0.0833,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "grid_res_average": 0.166,  # Grid res for average data  Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    "calc_opt_traj_after_planning": False,
    "x_interval_seaweed": [-130, -70],
    "y_interval_seaweed": [-40, 0],
    "dirichlet_boundry_constant": 1,
    "discount_factor_tau": False, #500, #50 #False, #10,
    "affine_dynamics": True,
}
# wandb.config.update({"planner_settings": specific_settings})
#%
planner = HJBSeaweed2DPlanner(arena=arena, problem=problem, specific_settings=specific_settings)
observer = NoObserver()
# % Run reachability planner
observation = arena.reset(platform_state=x_0)
observer.observe(observation)
observation.forecast_data_source = observer
action = planner.get_action(observation=observation)
#%% get value function #
planner.animate_value_func_3D()
# %% Let the controller run closed-loop within the arena (the simulation loop)
# observation = arena.reset(platform_state=x_0)
dt_in_s = arena.platform.platform_dict["dt_in_s"]
print(arena.platform.state.seaweed_mass.kg)

for i in tqdm(range(int((specific_settings["T_goal_in_seconds"]) / dt_in_s))):
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
    observer.observe(observation)
    observation.forecast_data_source = observer

#%%
## Seaweed growth curve
print(arena.platform.state.seaweed_mass.kg)
fig, ax = plt.subplots()
ax = arena.plot_seaweed_trajectory_on_timeaxis(ax=ax)
fig.canvas.draw()
ax.draw(fig.canvas.renderer)

arena.plot_all_on_map(
    problem=problem, background="current", quiver_spatial_res=0.155, quiver_scale=9
)
arena.plot_all_on_map(problem=problem, background="seaweed")
# %%
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=14400,  # 7200,
    background="current",
    quiver_spatial_res=0.155,
    quiver_scale=9,
    # margin=8,
    # x_interval=[-98,-72],
    # y_interval=[0,-9],
    output="trajectory_currents.mp4",
)
# # wandb.log(
# #   {"video": wandb.Video("generated_media/trajectory_currents.mp4", fps=25, format="mp4")})
# # %%
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=14400,  # 7200,
    background="seaweed",
    # margin=8,
    # x_interval=[-98,-72],
    # y_interval=[0,-9],
    output="trajectory_seaweed.mp4",
)
