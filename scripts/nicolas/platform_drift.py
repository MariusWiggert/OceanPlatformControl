# %%
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
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

os.chdir("/home/nicolas/documents/Master_Thesis_repo/OceanPlatformControl")
print(os.getcwd())

# %% Initialize the Arena (holds all data sources and the platform, everything except controller)
arena = ArenaFactory.create(scenario_name="gulf_of_mexico_HYCOM_hindcast_local", nb_platforms=2)
# we can also download the respective files directly to a temp folder, then t_interval needs to be set
# % Specify Navigation Problem
x_0_1 = PlatformState(
    lon=units.Distance(deg=-82.5),
    lat=units.Distance(deg=23.7),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
    id=int(0),
)
x_0_2 = PlatformState(
    lon=units.Distance(deg=-82.6),
    lat=units.Distance(deg=23.8),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
    id=int(1),
)
x_T = SpatialPoint(lon=units.Distance(deg=-80.3), lat=units.Distance(deg=24.6))

problem_1 = NavigationProblem(
    start_state=[x_0_1, x_0_2],
    # start_state= [x_0_1],
    end_region=x_T,
    target_radius=0.1,
    timeout=datetime.timedelta(days=2),
    platform_dict=arena.platform[0].platform_dict,
)

# %% Plot the problem on the map
t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=[x_0_1.to_spatio_temporal_point(), x_0_2.to_spatio_temporal_point()],
    # x_0 = [x_0_1.to_spatio_temporal_point()],
    x_T=x_T,
    deg_around_x0_xT_box=1,
    temp_horizon_in_s=3600,
)

ax1 = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=t_interval[0], x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem_1.plot(ax=ax1)
plt.show()


# %% Instantiate the HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "multi-time-reach-back",
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
    "platform_dict": arena.platform[0].platform_dict,
}
planner = HJReach2DPlanner(problem=problem_1, specific_settings=specific_settings)

# % Run reachability planner
observation = arena.reset(platform_state_list=[x_0_1, x_0_2])
action = planner.get_action(observation_list=observation)
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
# planner.plot_reachability_animation(time_to_reach=True, granularity_in_h=5, filename="test_reach_animation.mp4")
# planner.plot_reachability_animation(time_to_reach=True, granularity_in_h=5, with_opt_ctrl=True,
#                                     filename="test_reach_animation_w_ctrl.mp4", forward_time=True)
#%% save planner state and reload it
# Save it to a folder
# planner.save_planner_state("saved_planner/")
# Load it from the folder
# loaded_planner = HJReach2DPlanner.from_saved_planner_state(folder="saved_planner/", problem=problem)
# loaded_planner.last_data_source = arena.ocean_field.hindcast_data_source
# observation = arena.reset(platform_state=x_0)
# loaded_planner._update_current_data(observation=observation)
# planner = loaded_planner
# %% Let the controller run closed-loop within the arena (the simulation loop)
for i in tqdm(range(int(3600 * 24 * 1 / 600))):  # 3 days
    action = planner.get_action(observation=observation)
    observation = arena.step(action)

#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem_1)
#%% Animate the trajectory
arena.animate_trajectory(problem=problem_1, temporal_resolution=7200)