#%%
import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm

from ocean_navigation_simulator.controllers.multi_agent_planner import (
   MultiAgentPlanner,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState, PlatformStateSet
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units
import os
import numpy as np

os.chdir("/home/nicolas/documents/Master_Thesis_repo/OceanPlatformControl")
#%%
# Initialize the Arena (holds all data sources and the platform, everything except controller)
arena = ArenaFactory.create(scenario_name="gulf_of_mexico_HYCOM_hindcast_local")

# Initialize two platforms
x_0 = PlatformState(
    lon=units.Distance(deg=-82.5),
    lat=units.Distance(deg=23.7),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_1 = PlatformState(
    lon=units.Distance(deg=-82.25),
    lat=units.Distance(deg=23.7),
    date_time=datetime.datetime(2021, 11, 25, 12, 0, tzinfo=datetime.timezone.utc),
)
# create a platformSet object
x_set = PlatformStateSet([x_0, x_1])
# try if methods returns are correct
print("lon array: ", x_set.lon, ", lat array: ", x_set.lat)
# target region is the same for now
x_T = SpatialPoint(lon=units.Distance(deg=-80.3), lat=units.Distance(deg=24.6))

#%% create a navigation problem
problem = NavigationProblem(
    start_state=x_set,
    end_region=x_T,
    target_radius=0.1,
    timeout=datetime.timedelta(days=2),
    platform_dict=arena.platform.platform_dict,
)

# # %% Plot the problem on the map
# t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
#     x_start=x_set.to_spatio_temporal_point_set(), x_T=x_T, deg_around_x0_xT_box=1, temp_horizon_in_s=3600
# )
# ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
# )
# problem.plot(ax=ax)
# plt.show()

# %% Instantiate the HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "backward",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 2,
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
multi_agent_setting = {
    "planner": "hj_planner"
}
planner_set = MultiAgentPlanner(problem=problem, multi_agent_setting = multi_agent_setting, 
                                  specific_settings= specific_settings)

observation = arena.reset(platform_set=x_set)
action = planner_set.get_action(observation=observation)

# %%
new_observation = arena.step(action)