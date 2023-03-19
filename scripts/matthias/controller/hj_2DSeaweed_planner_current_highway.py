#%% Run imports
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

import wandb
from ocean_navigation_simulator.controllers.hj_planners.HJBSeaweed2DPlanner import (
    HJBSeaweed2DPlanner,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.ocean_observer.NoObserver import NoObserver
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.SeaweedProblem import (
    SeaweedProblem,
)
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import set_arena_loggers
from ocean_navigation_simulator.data_sources.SeaweedGrowth.SeaweedGrowthSource import SeaweedGrowthCircles

# %load_ext autoreload
# %autoreload 2

# % Initialize
set_arena_loggers(logging.INFO)

scenario_name = "current_highway_toy_seaweed"  # "Region_3_Copernicus_HC_solar_seaweed"

# Initialize the Arena (holds all data sources and the platform, everything except controller)
arena = ArenaFactory.create(scenario_name=scenario_name)
#%
x_0 = PlatformState(
    lon=units.Distance(deg=5),
    lat=units.Distance(deg=0.1),
    date_time=datetime.datetime.utcfromtimestamp(0).astimezone(tz=datetime.timezone.utc),
    seaweed_mass=units.Mass(kg=100),
)


problem = SeaweedProblem(
    start_state=x_0,
    platform_dict=arena.platform.platform_dict,
)


# %% Plot the problem on the map
t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(),
    x_T=x_0.to_spatio_temporal_point(),
    deg_around_x0_xT_box=3,
    temp_horizon_in_s=3600,
)

#%%
ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True, quiver_spatial_res=0.17, quiver_scale=6)
problem.plot(ax=ax)
plt.show()
plt.tight_layout()

#%%
ax = arena.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()

# %% Instantiate the HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "backward",
    "n_time_vector": 100,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box_global": 3.5,  # area over which to run HJ_reachability on the first global run
    "deg_around_xt_xT_box": 10,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 100,
    "use_geographic_coordinate_system": False,
    "progress_bar": True,
    "grid_res_global": 0.1,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "grid_res": 0.1,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    "platform_dict": arena.platform.platform_dict,
    "calc_opt_traj_after_planning": False,
    ## discount Factor stuff
    # The higher the discount factor, the MORE the planner cares about the future.
    # This is the opposite of the usual RL discount factor!
    # E.g. if we set it to 100, that means at each time-step the value function is reduced by 1/100th*dt of it's value.
    'discount_factor_tau': 50 #False, #10,
}

planner = HJBSeaweed2DPlanner(arena=arena, problem=problem, specific_settings=specific_settings)
observer = NoObserver()
# %Run reachability planner
observation = arena.reset(platform_state=x_0)
observer.observe(arena_observation=observation)
observation.forecast_data_source = observer
action = planner.get_action(observation=observation)
#%%
planner.plot_value_fct_snapshot(
    rel_time_in_seconds=0,
)
#%% get value function #
planner.animate_value_func_3D()
# %% Let the controller run closed-loop within the arena (the simulation loop)
# observation = arena.reset(platform_state=x_0)
dt_in_s = arena.platform.platform_dict["dt_in_s"]
print(arena.platform.state.seaweed_mass.kg)
# 3600 * 24 * 10
for i in tqdm(range(int(50 / dt_in_s))):
    action = planner.get_action(observation=observation)
    observation = arena.step(action)

#%%
## Seaweed growth curve
print(arena.platform.state.seaweed_mass.kg)
fig, ax = plt.subplots()
ax = arena.plot_seaweed_trajectory_on_timeaxis(ax=ax)
fig.canvas.draw()
ax.draw(fig.canvas.renderer)
plt.show()

#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, background="seaweed")
arena.plot_all_on_map(problem=problem, background="current", quiver_spatial_res=0.165, quiver_scale=7)


# %%
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=1,  # 7200,
    background="current",
    output="trajectory_currents.mp4",
    quiver_spatial_res=0.165, 
    quiver_scale=7,
)
# wandb.log(
#   {"video": wandb.Video("generated_media/trajectory_currents.mp4", fps=25, format="mp4")})

arena.animate_trajectory(
    problem=problem,
    temporal_resolution=1,  # 7200,
    background="seaweed",
    output="trajectory_seaweed.mp4",
)
