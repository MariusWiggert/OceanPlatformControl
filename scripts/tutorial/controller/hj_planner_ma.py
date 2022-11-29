#%%
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

os.chdir("/home/nicolas/documents/Master_Thesis_repo/OceanPlatformControl")
# os.chdir("/home/nicolas/codeRepo/OceanPlatformControl")
from ocean_navigation_simulator.controllers.multi_agent_planner import (
    MultiAgentPlanner,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import (
    PlatformState,
    PlatformStateSet,
)
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units

#%%
# Initialize the Arena (holds all data sources and the platform, everything except controller)
arena = ArenaFactory.create(scenario_name="gulf_of_mexico_HYCOM_hindcast_local")

# Initialize two platforms
x_0 = PlatformState(
    lon=units.Distance(deg=-83.4),
    lat=units.Distance(deg=23.2),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_1 = PlatformState(
    lon=units.Distance(deg=-83.4),
    lat=units.Distance(deg=23.25),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_2 = PlatformState(
    lon=units.Distance(deg=-83.35),
    lat=units.Distance(deg=23.2),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_3 = PlatformState(
    lon=units.Distance(deg=-83.35),
    lat=units.Distance(deg=23.25),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
# create a platformSet object
x_set = PlatformStateSet([x_0, x_1, x_2, x_3])
# try if methods returns are correct
print("lon array: ", x_set.lon, ", lat array: ", x_set.lat)
# target region is the same for now
x_T = SpatialPoint(lon=units.Distance(deg=-82.3), lat=units.Distance(deg=24.35))

#%% create a navigation problem
problem = NavigationProblem(
    start_state=x_set,
    end_region=x_T,
    target_radius=0.12,
    timeout=datetime.timedelta(days=2),
    platform_dict=arena.platform.platform_dict,
)
problem.passed_seconds(x_0)
# %% Plot the problem on the map
t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_set.to_spatio_temporal_point(), x_T=x_T, deg_around_x0_xT_box=1, temp_horizon_in_s=3600
)
ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.savefig("generated_media/problemSetting.png")

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
    "platform_dict": arena.platform.platform_dict,
}
multi_agent_settings = {"planner": "hj_planner"}
planner_set = MultiAgentPlanner(
    problem=problem, multi_agent_settings=multi_agent_settings, specific_settings=specific_settings
)

observation = arena.reset(platform_set=x_set)
action = planner_set.get_action_set(observation=observation)

# %% Reachability snapshot plot
plt.clf()
planner_set.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    plot_in_h=True,
    return_ax=True,
)
plt.savefig("reachabilitySnap.png")

update_rate_s = arena.platform.platform_dict["dt_in_s"]  # 10 mins
day_sim = 1
for i in tqdm(range(int(3600 * 24 * day_sim / update_rate_s))):  # 1 day
    action = planner_set.get_action_set(observation=observation)
    observation = arena.step(action)

arena.animate_graph_net_trajectory(
    temporal_resolution=7200,
    # collision_communication_thrslds=(10, 50), (not specified take defaut one)
    plot_ax_ticks=True,
    output="network_graph_anim_1days_new.mp4",
)
arena.plot_graph_isolated_platforms()
plt.savefig("isolatedPlatforms.png")
arena.plot_distance_evolution_between_platforms()
plt.savefig("distanceEvolution6.png")
arena.plot_platform_nb_collisions()
plt.savefig("collisions.png")
#%% Plot the arena trajectory on the map
# ax = arena.plot_all_on_map(problem=problem, return_ax=True)
# ax = problem.plot(ax=ax)
# plt.savefig("traj_3_days.png", dpi=300)
# # #%% Animate the trajectory
arena.animate_trajectory(
    margin=0.25, problem=problem, temporal_resolution=7200, output="traj_1days_anim_new_new.mp4"
)
# ax = arena.plot_distance_evolution_between_neighbors(neighbors_list_to_plot= [(0, 1), (0, 2), (1, 3)], figsize=(9, 6))
# plt.savefig("distance_evolution_3days", dpi=300)
