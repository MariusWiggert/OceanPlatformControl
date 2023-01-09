import datetime

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")
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
from ocean_navigation_simulator.utils.plotting_utils import set_palatino_font

set_palatino_font("ocean_navigation_simulator/package_data/font/Palatino_thin.ttf")
logging.basicConfig(level=logging.DEBUG)
# Full GoM through (HYCOM Setting)
# x_0_list = [-94.5, 25, datetime.datetime(2021, 11, 22, 12, 10, 10, tzinfo=datetime.timezone.utc)]
# x_T_list = [-83, 25]
# x_T_list = [-78, 30]
# Full GoM through (COP Setting)
x_0_list = [-94.5, 25, datetime.datetime(2021, 10, 2, 0, 0, 0, tzinfo=datetime.timezone.utc)]
x_T_list = [-78, 30]
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

# % Instantiate the HJ Planner
# probably need to change: u_max, grid_res, might have to do forward/backward here... more stable?
# smaller red-dot, lables on axis!
specific_settings = {
    "replan_on_new_fmrc": False,
    "replan_every_X_seconds": False,
    "direction": "backward",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 4,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 50,
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    "calc_opt_traj_after_planning": False,  # TODO change it!
    # "num_traj_disc": 50,
    # "dt_in_sec": 1800,
    # 'x_interval': [-82, -78.5],
    # 'y_interval': [23, 25.5],
    "initial_set_radii": [
        0.1,
        0.1,
    ],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    "grid_res": 0.12,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
    "platform_dict": arena.platform.platform_dict,
}
planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
# % Run reachability planner
observation = arena.reset(platform_state=x_0)
# %% Plot the problem on the map (Hindcast data)
ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time + datetime.timedelta(days=2),
    x_interval=[-98, -77],
    y_interval=[18, 32],
    return_ax=True,
    # figsize=(12, 7),
    quiver_scale=30,
    quiver_spatial_res=0.3,
    vmax=1.4,
)
problem.plot(ax=ax)
plt.show()
#%% under FC
# 65% under 0.05 res, 96% with 0.1 res
action = planner.get_action(observation=observation)


def add_planned_traj(ax, time, x_traj=planner.x_traj):
    ax.plot(
        x_traj[0, :],
        x_traj[1, :],
        color="black",
        linewidth=2,
        linestyle="--",
        label="State Trajectory",
    )


planner.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=50,
    alpha_color=1,
    time_to_reach=False,
    fig_size_inches=(12, 7),
    plot_in_h=True,
    # add_drawing=add_planned_traj,
    # add_drawing=plan_from_x_t_plot
)
#%%
planner.plot_reachability_animation(
    time_to_reach=False,
    granularity_in_h=10,
    temporal_resolution=3600 * 3,
    fps=20,
    with_opt_ctrl=False,
    filename="backward_new.mp4",
    with_background=True,
    figsize=(12, 7),
    background_animation_args={
        "quiver_scale": quiver_scale,
        "quiver_spatial_res": quiver_spatial_res,
        "vmax": vmax,
        "figsize": (12, 7),
    },
)
#%% forward
planner.plot_reachability_animation(
    time_to_reach=True,
    granularity_in_h=5,
    temporal_resolution=1800,
    fps=20,
    with_opt_ctrl=True,
    filename="forward_new_fast_new.mp4",
    with_background=True,
    forward_time=True,
    figsize=(12, 7),
    background_animation_args={
        "quiver_scale": quiver_scale,
        "quiver_spatial_res": quiver_spatial_res,
        "vmax": vmax,
        "figsize": (12, 7),
    },
)
#%% simulate closed-loop
problem_status = 0
action = planner.get_action(observation=observation)
current_fmrc = planner.last_fmrc_time_planned_with
# # % Plot planning for each of the forecasts
while problem_status == 0:
    action = planner.get_action(observation=observation)
    # check if new forecast, then plot
    new_fmrc = planner.last_fmrc_time_planned_with
    observation = arena.step(action)
    # update problem status
    problem_status = arena.problem_status(problem=problem)
# arena.plot_all_on_map(problem=problem)
arena.animate_trajectory(
    problem=problem,
    set_title=True,
    temporal_resolution=1800,
    full_traj=False,
    fps=15,
    output="full_GoM.mp4",
    traj_linestyle="-",
    traj_color="black",
    traj_linewidth=6,
    figsize=(18, 9),
    vmax=1.4,
    quiver_spatial_res=0.3,
    quiver_scale=30,
    # x_interval=specific_settings['x_interval'],
    # y_interval=specific_settings['y_interval'],
    x_t_marker_color="m",
    x_t_marker_size=40,
)

#%% create an add_ax_function: plan_traj_from_x_t_and_plot


def plan_traj_from_x_t_and_plot(
    ax, rel_time, arena_traj=arena.state_trajectory, num_traj_disc=None, dt_in_sec=1800, **kwargs
):
    # get x_t from arena trajectory
    idx_x_t = min(
        np.searchsorted(a=arena_traj[:, 2] - planner.current_data_t_0, v=rel_time),
        len(arena_traj) - 1,
    )
    # plot trajectory so far in white
    ax.plot(
        arena_traj[:idx_x_t, 0],
        arena_traj[:idx_x_t, 1],
        color=kwargs.get("traj_color", "whitesmoke"),
        linewidth=kwargs.get("traj_linewidth", 5),
        linestyle=kwargs.get("traj_linestyle", "-"),
        label="State Trajectory",
    )
    # plot planned trajectory in the future
    x_t = arena_traj[idx_x_t, :]
    times, x_traj, contr_seq, _ = planner._extract_trajectory(
        x_start=planner.get_x_from_full_state(x_t),
        t_start=x_t[2],
        num_traj_disc=num_traj_disc,
        dt_in_sec=dt_in_sec,
    )
    # get the planned idx of current time
    select_idx = 0  # np.searchsorted(a=trajectory[2,:], v=x_t.date_time.timestamp())
    ax.plot(x_traj[0, :], x_traj[1, :], linewidth=2, color="whitesmoke", linestyle="--")
    # plot the control arrow for the specific time
    ax.scatter(x_traj[0, select_idx], x_traj[1, select_idx], c="magenta", marker="o", s=20)
    ax.quiver(
        x_traj[0, select_idx],
        x_traj[1, select_idx],
        contr_seq[0, min(select_idx, len(times) - 1)]
        * np.cos(contr_seq[1, min(select_idx, len(times) - 1)]),  # u_vector
        contr_seq[0, min(select_idx, len(times) - 1)]
        * np.sin(contr_seq[1, min(select_idx, len(times) - 1)]),  # v_vector
        color="magenta",
        scale=10,
        label="Next Control",
        zorder=4,
    )
    plot_plan_from_idx = 5
    ax.scatter(
        x_traj[0, plot_plan_from_idx:-1:4],
        x_traj[1, plot_plan_from_idx:-1:4],
        c="m",
        marker="o",
        s=20,
    )
    ax.quiver(
        x_traj[0, plot_plan_from_idx:-1:4],
        x_traj[1, plot_plan_from_idx:-1:4],
        contr_seq[0, plot_plan_from_idx - 1 : -1 : 4]
        * np.cos(contr_seq[1, plot_plan_from_idx - 1 : -1 : 4]),  # u_vector
        contr_seq[0, plot_plan_from_idx - 1 : -1 : 4]
        * np.sin(contr_seq[1, plot_plan_from_idx - 1 : -1 : 4]),  # v_vector
        color="orchid",
        alpha=0.5,
        scale=10,
        linestyle="-",
        label="Future Planned Controls",
        zorder=4,
    )


#%%
plan_from_x_t_plot = partial(plan_traj_from_x_t_and_plot, arena.state_trajectory, None, 1800)
planner.plot_reachability_animation(
    time_to_reach=True,
    granularity_in_h=5,
    with_opt_ctrl=False,
    temporal_resolution=3600,
    fps=15,
    filename="replanning_x_t_fmrc_idx_{}.mp4".format(current_fmrc.timestamp()),
    forward_time=True,
    add_drawing=plan_from_x_t_plot,
    with_background=False,
    figsize=(12, 7),
    t_end=observation.platform_state.date_time,
)
#%% Test plotting
plan_from_x_t_plot = partial(plan_traj_from_x_t_and_plot, arena.state_trajectory, None, 1800)
planner.plot_reachability_snapshot(
    rel_time_in_seconds=3600 * 20,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    plot_in_h=True,
    add_drawing=plan_from_x_t_plot,
)
#%%
arena.plot_all_on_map(problem=problem)
#%%
# partial pack the function with most current arena traj
plan_from_x_t_plot = partial(plan_traj_from_x_t_and_plot, arena.state_trajectory, None, 3600)
planner.plot_reachability_animation(
    time_to_reach=True,
    granularity_in_h=5,
    with_opt_ctrl=False,
    temporal_resolution=3600,
    filename="replanning_x_t_fmrc_idx_{}.mp4".format(current_fmrc.timestamp()),
    forward_time=True,
    add_drawing=plan_from_x_t_plot,
    t_end=observation.platform_state.date_time + datetime.timedelta(hours=5),
)
#%% Other to-do's left!
# in the replanning always display the first planned path!
#%%
#% Then plot all of it together
set_palatino_font("ocean_navigation_simulator/package_data/font/Palatino_thin.ttf")
arena.animate_trajectory(
    problem=problem,
    set_title=True,
    temporal_resolution=3600,
    full_traj=False,
    output="full_traj_iteratively.mp4",
    traj_linestyle="-",
    traj_color="whitesmoke",
    traj_linewidth=5,
)
#%% Plot reachability planning for each of the forecasts
planner.plot_reachability_snapshot_over_currents(
    rel_time_in_seconds=0, granularity_in_h=5, time_to_reach=False
)
#%% Plot planning for each of the forecasts
set_palatino_font("ocean_navigation_simulator/package_data/font/Palatino_thin.ttf")
planner.plot_reachability_animation(
    time_to_reach=True, granularity_in_h=5, filename="fmrc_idx_{}.mp4".format(0)
)
#%%
planner.plot_reachability_snapshot_over_currents(
    rel_time_in_seconds=3600 * 50,
    return_ax=False,
    time_to_reach=True,
    add_drawing=plan_traj_from_x_t_and_plot,
)
#%% Now make an animation with that..
planner.plot_reachability_animation(
    time_to_reach=True,
    granularity_in_h=5,
    with_opt_ctrl=False,
    temporal_resolution=3600 * 10,
    filename="test_reach_animation_w_ctrl.mp4",
    forward_time=True,
    add_drawing=plan_traj_from_x_t_and_plot,
)
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
