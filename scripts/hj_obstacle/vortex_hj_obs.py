#%%

import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)
from ocean_navigation_simulator.data_sources.OceanCurrentField import (
    OceanCurrentField,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import (
    PlatformState,
    SpatialPoint,
)
from ocean_navigation_simulator.utils import units

%load_ext autoreload
%autoreload 2
#%
scenario_name = "vortex_current"
arena = ArenaFactory.create(scenario_name=scenario_name)

#% Get the ranges of the vortex field dynamics
x_range = arena.ocean_field.hindcast_data_source.grid_dict['x_range']
y_range = arena.ocean_field.hindcast_data_source.grid_dict['y_range']
t_range = arena.ocean_field.hindcast_data_source.grid_dict['t_range']
print("x_range: ", x_range)
print("y_range: ", y_range)
print("t_range: ", t_range) #=> operates in posix time
#% Define the problem
# Start at left middle
x_0 = PlatformState(
    lon=units.Distance(deg=0.0),
    lat=units.Distance(deg=0.25),
    date_time=datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
)
# After cylinder in the middle
x_T = SpatialPoint(lon=units.Distance(deg=0.7),lat=units.Distance(deg=0.25),)


problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.03,
)

#%% viz the current field
# Regarding magnitude of currents, we can leave it or scale it up/down...
# -> can be added hacky as multiplication when loading the DataArray in the OceanDataSouce

ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time,
    x_interval=x_range,
    y_interval=y_range,
    # plot_type='streamline',
    plot_type="quiver",
    return_ax=True,
    # vmax=0.7,
    quiver_spatial_res=0.05,
    # quiver_scale=15,
)
problem.plot(ax=ax)
plt.show()
#%% Animate the full currents for intuition
arena.ocean_field.hindcast_data_source.animate_data(
    x_interval=x_range,
    y_interval=y_range,
    t_interval=t_range,
    plot_type="quiver",
    quiver_spatial_res=0.05,
    temporal_resolution=1,
    output="vortex_current_animation.mp4",
)

#%% apply the HJ planner
specific_settings = {
    "direction": "multi-time-reach-back",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1,  # area over which to run HJ_reachability
    "T_goal_in_seconds": 3,
    "calc_opt_traj_after_planning": True, # this takes a bit so set False for faster debugging
    "use_geographic_coordinate_system": False,
    "progress_bar": True,
    "grid_res": 0.02,
    "platform_dict": arena.platform.platform_dict,
    # "obstacle_dict": {
    #     "path_to_obstacle_file":filename_obstacles, #"ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_-150.nc",
    #     "obstacle_value": 1,
    #     "safe_distance_to_obstacle": 0,
    # },
}

planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
observation = arena.reset(platform_state=x_0)
action = planner.get_action(observation=observation)

#%%
# Note hours is now the non-dimensional time units when setting plot_in_h=False
planner.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=0.1,
    plot_in_h=False,
    alpha_color=1,
    time_to_reach=True,
)
#%%
# Note: to make sure it does not mask out too much, reduce grid_resolution...
planner.plot_reachability_snapshot_over_currents(
    rel_time_in_seconds=0,
    granularity_in_h=0.1,
    plot_in_h=False,
    alpha_color=0.5, # of the TTR function overlay -> Maybe we just plot contourlines?
    time_to_reach=True,
    background_args={
        'plot_type':"quiver",
        "alpha": 0.2,# of the background flow...
        'quiver_spatial_res': 0.05}
)
#%%
# Note
# planner.plot_reachability_snapshot_over_currents(rel_time_in_seconds=0, granularity_in_h=5, time_to_reach=False)
planner.plot_reachability_animation(time_to_reach=True,
                                    plot_in_h=False,
                                    granularity_in_h=0.2,
                                    temporal_resolution=0.5,
                                    with_opt_ctrl=True,
                                    forward_time=True,
                                    filename="test_reach_animation.mp4",
                                    background_animation_args={
                                        'plot_type':"quiver",
                                        'quiver_spatial_res': 0.05,

                                    }
                                    )
# planner.plot_reachability_animation(time_to_reach=True, granularity_in_h=5, with_opt_ctrl=True,
#                                     filename="test_reach_animation_w_ctrl.mp4", forward_time=True)