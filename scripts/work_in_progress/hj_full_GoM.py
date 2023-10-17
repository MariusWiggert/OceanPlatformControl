import datetime

import matplotlib.pyplot as plt
import numpy as np

from ocean_navigation_simulator.controllers.NaiveController import NaiveController

# plt.style.use("dark_background")
import logging
from functools import partial

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
from ocean_navigation_simulator.utils.plotting_utils import set_palatino_font\

# #%%
# import xarray as xr
# xr.open_dataset("data/hycom_hc_2022/hindcasts_2022.nc4")
# #%%

set_palatino_font()
# logging.basicConfig(level=logging.DEBUG)
#%
# Full GoM through (HYCOM Setting)
# [datetime.datetime(2022, 10, 1, 12, 0, tzinfo=datetime.timezone.utc),
#   datetime.datetime(2022, 12, 6, 10, 0, tzinfo=datetime.timezone.utc)]
x_0_list = [-94.5, 25, datetime.datetime(2021, 11, 20, 12, 0, 0, tzinfo=datetime.timezone.utc)]
# x_0_list = [-95.4, 22.7, datetime.datetime(2021, 11, 21, 10, 0, 0, tzinfo=datetime.timezone.utc)]
# x_T_list = [-83, 25]
x_T_list = [-79, 24]
# x_T_list = [-78, 30]
# Full GoM through (COP Setting)
# x_0_list = [-94.5, 25, datetime.datetime(2023, 2, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)]
# x_T_list = [-78, 30]
# x_0_list = [-96, 22, datetime.datetime(2023, 2, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)]
# x_T_list = [-78, 30]
x_0 = PlatformState(
    lon=units.Distance(deg=x_0_list[0]),
    lat=units.Distance(deg=x_0_list[1]),
    date_time=x_0_list[2],
)
x_T = SpatialPoint(lon=units.Distance(deg=x_T_list[0]), lat=units.Distance(deg=x_T_list[1]))

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.5,
)

vmax = 1.4
quiver_spatial_res = 0.3
quiver_scale = 30

# Initialize the Arena (holds all data sources and the platform, everything except controller)
arena = ArenaFactory.create(
    scenario_name="gulf_of_mexico_HYCOM_hindcast_full",
    # t_interval=[
    #     datetime.datetime(2021, 11, 20, 12, 0, tzinfo=datetime.timezone.utc),
    #     datetime.datetime(2021, 12, 5, 12, 0, tzinfo=datetime.timezone.utc)]
)
#%%
arena.ocean_field.forecast_data_source.grid_dict['t_range']
#%%
import xarray as xr
xr.open_dataset("data/hycom_hc_2021/GOMu0_expt_90 (3).nc4")['time']
# %% Plot the problem on the map (Hindcast data)
plt.rcParams.update({"font.size": 16})

ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time + datetime.timedelta(days=0),
    x_interval=[-98, -77],
    y_interval=[18, 32],
    return_ax=True,
    figsize=(18, 9),
    quiver_scale=30,
    quiver_spatial_res=0.3,
    vmax=1.4,
)
problem.plot(ax=ax)
plt.show()
#%%
# % Instantiate the HJ Planner
# probably need to change: u_max, grid_res, might have to do forward/backward here... more stable?
# smaller red-dot, lables on axis!
specific_settings = {
    "replan_on_new_fmrc": False,
    "replan_every_X_seconds": False,
    "direction": "multi-time-reach-back",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 4,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 13,
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    "calc_opt_traj_after_planning": False,  # TODO change it!
    # "num_traj_disc": 50,
    "dt_in_sec": 600,
    # 'x_interval': [-82, -78.5],
    # 'y_interval': [23, 25.5],
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    "grid_res": 0.05,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
    "platform_dict": arena.platform.platform_dict,
}
planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
# planner = NaiveController(problem=problem)
# % Run reachability planner
observation = arena.reset(platform_state=x_0)

#% under FC
# 65% under 0.05 res, 96% with 0.1 res
action = planner.get_action(observation=observation)
#%% simulate closed-loop
problem_status = 0
action = planner.get_action(observation=observation)
# % Plot planning for each of the forecasts
while problem_status == 0:
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
    # update problem status
    problem_status = arena.problem_status(problem=problem)
print(arena.problem_status_text(problem_status))
#%% debug a bit!
planner.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=10,
    alpha_color=1,
    time_to_reach=True,
    plot_in_h=True,
    # add_drawing=add_planned_traj,
)
#%%
# compute the time it tookl
in_seconds = arena.state_trajectory.time[-1] - arena.state_trajectory.time[0]
print(f"Time to reach target: {in_seconds.total_seconds() / 3600} hours")
#%%
ax = arena.plot_all_on_map(problem=problem,
                    index=10,
                    control_stride = 300,
                    control_vec_scale = 17,
                    vmax=1.4, vmin=0,
                    x_interval=[-98, -77],
                    y_interval=[18, 32],
                    return_ax=True,
                    figsize=(18, 9),
                    spatial_resolution=0.3,
                    traj_linewidth=6,
                      )
# ax = arena.plot_state_trajectory_on_map(color='white')
# problem.plot(ax=ax)
# plt.show()
#%
# add trajectories
trajs = [passive_traj]#, naive_to_target_traj]
colors = ["white"]#, "darkred"]
linestyles = ["--"]#, "--"]
stride = 2
linewidth = 5
for traj, color, linestyle in zip(trajs, colors, linestyles):
    ax.plot(traj[::stride, 0],
            traj[::stride, 1],
            linestyle=linestyle,
            # marker=".",
            # markersize=1,
            color=color,
            linewidth=linewidth,
            zorder=1,
            # label="State Trajectory",
            )
# ax.legend()
# problem.plot(ax=ax)
plt.show()
#%%
def add_traj(ax, time):
    trajs = [passive_traj, naive_to_target_traj]
    colors = ["white", "darkred"]
    linestyles = ["-", "-"]
    stride=2
    linewidth=6
    for traj, color, linestyle in zip(trajs, colors, linestyles):
        ax.plot(traj[::stride, 0],
                traj[::stride, 1],
                linestyle=linestyle,
                alpha=0.5,
                # marker=".",
                # markersize=1,
                color=color,
                linewidth=linewidth,
                # label="State Trajectory",
                )
#%%
plt.rcParams.update({"font.size": 16})
arena.animate_trajectory(
    # margin=4,
    problem=problem,
    set_title=True,
    temporal_resolution=3600*4,
    full_traj=True,
    fps=15,
    output="full_GoM_longest_white.mp4",
    traj_linestyle="-",
    traj_color="black",
    traj_linewidth=6,
    figsize=(18, 9),
    vmax=1.4,
    vmin=0,
    quiver_spatial_res=0.3,
    quiver_scale=30,
    # x_interval=specific_settings['x_interval'],
    # y_interval=specific_settings['y_interval'],
    x_interval=[-98, -77],
    y_interval=[18, 32],
    x_t_marker_color="m",
    x_t_marker_size=40,
    # add_ax_func_ext=add_traj
)





#%% Plot others on the map!
# save the trajectory in arena.state_trajectory as pickle file
import pickle
# with open('state_trajectory.pickle', 'wb') as f:
#     pickle.dump(arena.state_trajectory, f)
# load it again
with open('state_trajectory.pickle', 'rb') as f:
    naive_to_target_traj = pickle.load(f)
#%%
import pickle
# with open('state_trajectory_passive.pickle', 'wb') as f:
#     pickle.dump(arena_new.state_trajectory, f)
# load it again
with open('state_trajectory_passive.pickle', 'rb') as f:
    passive_traj = pickle.load(f)
#%%
import pickle
# with open('ctrl_traj.pickle', 'wb') as f:
#     pickle.dump(arena.state_trajectory, f)
# load it again
with open('ctrl_traj.pickle', 'rb') as f:
    ctrl_traj = pickle.load(f)
#%%
plt.rcParams.update({"font.size": 16})

ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time + datetime.timedelta(days=0),
    x_interval=[-98, -77],
    y_interval=[18, 32],
    return_ax=True,
    figsize=(18, 9),
    quiver_scale=30,
    quiver_spatial_res=0.3,
    vmax=1.4,
    vmin=0,
)

# add trajectories

trajs = [passive_traj, naive_to_target_traj, ctrl_traj]
colors = ["white", "darkred", "black"]
linestyles = ["-", "-", "-"]
stride = 2
linewidth = 6
for traj, color, linestyle in zip(trajs, colors, linestyles):
    ax.plot(traj[::stride, 0],
            traj[::stride, 1],
            linestyle=linestyle,
            # marker=".",
            # markersize=1,
            color=color,
            linewidth=linewidth,
            # label="State Trajectory",
            )
# ax.legend()
problem.plot(ax=ax)
plt.show()

#%%
planner.plot_reachability_animation(
    time_to_reach=True,
    granularity_in_h=5,
    with_opt_ctrl=True,
    filename="test_reach_animation_w_ctrl.mp4",
    forward_time=True,
)
#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem)
#%% Animate the trajectory FINAL
set_palatino_font("ocean_navigation_simulator/package_data/font/Palatino_thin.ttf")
arena.animate_trajectory(
    problem=problem,
    set_title=True,
    temporal_resolution=3600,
    full_traj=False,
    output="not_full_traj.mp4",
    traj_linestyle="-",
    traj_color="whitesmoke",
    traj_linewidth=5,
)
#%% Plot Hindcast in that area of a while
arena.ocean_field.hindcast_data_source.animate_data(
    x_interval=[-87, -79],
    y_interval=[22, 27],
    t_interval=[
        datetime.datetime(2021, 11, 21, 12, 0, tzinfo=datetime.timezone.utc),
        datetime.datetime(2021, 12, 5, 0, 0, tzinfo=datetime.timezone.utc),
    ],
    temporal_resolution=3600 * 5,
    spatial_resolution=0.1,
    output="HYCOM_GT_animation.mp4",
)
#%% Plot Forecast in that area of a while
arena.ocean_field.forecast_data_source.animate_data(
    x_interval=[-87, -79],
    y_interval=[22, 27],
    t_interval=[
        datetime.datetime(2021, 11, 21, 12, 0, tzinfo=datetime.timezone.utc),
        datetime.datetime(2021, 12, 5, 0, 0, tzinfo=datetime.timezone.utc),
    ],
    temporal_resolution=3600 * 5,
    spatial_resolution=0.1,
    output="HYCOM_FMRC_animation.mp4",
)
#%%
arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    x_interval=[-87, -79],
    y_interval=[23, 25],
    time=datetime.datetime(2021, 11, 21, 12, 0, tzinfo=datetime.timezone.utc),
)
