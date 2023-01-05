import datetime
import logging
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import yaml

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

logging.basicConfig(level=logging.DEBUG)

# plotting parameters
vmax = 1.3
quiver_spatial_res = 0.1
quiver_scale = 18
figsize = (12, 7)
set_palatino_font("ocean_navigation_simulator/package_data/font/Palatino_thin.ttf")
plt.style.use("dark_background")

# Notes to replicate CDC presentation animations (not necessary, pipeline works without it)
# - As FC the HYCOM forecasts (can download with settings below) as HC I used a single FC
#   'GOMu0.04_901m000_FMRC_RUN_2021-11-23T12_00_00Z-2021-11-23T12_00_00Z-2021-11-28T12_00_00Z'
#   -> need to manually delete the downloaded hindcasts in "data/cdc22_animation_fc_as_hindcast/"
#      and add the specific FMRC_RUN file from "data/cdc22_animation_fc" to this folder (after downloading HYCOM FC)
#      Only then move to next cell, otherwise won't replicate it (but pipeline will still work)
# - to plot without the x, y axis labels, change in DataSource.set_up_geographic_ax() draw_labels=True to False
# - need to make a small hack to get the noise needed for the animation. Seed is 151 and need to change sign of water_u
#   Go into OceanCurrentSource in GroundTruthFromNoise.get_data_over_area, in Step 2 add this line
#   additive_noise = additive_noise.assign(water_u=lambda x: -x.water_u)

# plot over area
plot_x_interval = [-83, -79]
plot_y_interval = [23, 25.3]

#%% Download the HYCOM Files if needed
download_data = False
if download_data:
    # load yaml file
    with open(os.path.join("config/arena/", "gulf_of_mexico_CDC22_animations.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # modify config to enable downloading of forecasts
    config["ocean_dict"]["hindcast"] = {
        "source": "hindcast_files",
        "source_settings": {
            "local": False,
            "folder": "data/cdc22_animation_fc_as_hindcast/",
            "source": "HYCOM",
            "type": "hindcast",
        },
    }
    # initialize arena just to trigger downloading of FC and HC files
    _ = ArenaFactory.create(
        scenario_config=config,
        t_interval=[
            datetime.datetime(2021, 11, 20, 12, 0, tzinfo=datetime.timezone.utc),
            datetime.datetime(2021, 12, 5, 12, 0, tzinfo=datetime.timezone.utc),
        ],
    )
#%% Only go here when the data is downloaded in the right folders!
# CDC Presentation Case with Noise Seed 151
x_0_list = [-81, 23.7, datetime.datetime(2021, 11, 23, 12, 00, 00, tzinfo=datetime.timezone.utc)]
x_T_list = [-80.4, 24.2]

# Initialize the Problem
x_0 = PlatformState(
    lon=units.Distance(deg=x_0_list[0]), lat=units.Distance(deg=x_0_list[1]), date_time=x_0_list[2]
)
x_T = SpatialPoint(lon=units.Distance(deg=x_T_list[0]), lat=units.Distance(deg=x_T_list[1]))
problem = NavigationProblem(start_state=x_0, end_region=x_T, target_radius=0.1)

# Initialize the Arena
arena = ArenaFactory.create(scenario_name="gulf_of_mexico_CDC22_animations")
observation = arena.reset(platform_state=x_0)

# % Instantiate the HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "multi-time-reach-back",
    "n_time_vector": 200,
    "closed_loop": True,  # to run closed-loop or open-loop
    "deg_around_xt_xT_box": 1.5,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 110,
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    # Settings for the trajectory calculation after each replan
    "calc_opt_traj_after_planning": False,
    # "num_traj_disc": 50,
    "dt_in_sec": 1800,  # discretization at which to backtrack the trajectory
    "x_interval": [-83, -78.5],  # fixed interval for planning (for consistent plotting)
    "y_interval": [23, 25.3],  # fixed interval for planning (for consistent plotting)
    "grid_res": 0.02,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "platform_dict": arena.platform.platform_dict,
}
planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
#%% Plot Problem on Map over Forecast Data Source
ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time,
    x_interval=plot_x_interval,
    y_interval=plot_y_interval,
    return_ax=True,
    figsize=figsize,
    quiver_scale=quiver_scale,
    quiver_spatial_res=quiver_spatial_res,
    vmax=vmax,
)
problem.plot(ax=ax)
plt.show()
#%% Animate the Hindcast Currents with the Problem over temp_horizon_in_s
t_interval, _, _ = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(),
    x_T=x_T,
    deg_around_x0_xT_box=1,
    temp_horizon_in_s=3600 * 24 * 5,
)


def add_problem(ax, time):
    problem.plot(ax=ax)


# Note: to visualize the first forecast, just replace hindcast_data_source with forecast_data_source
arena.ocean_field.hindcast_data_source.animate_data(
    x_interval=plot_x_interval,
    y_interval=plot_y_interval,
    t_interval=t_interval,
    temporal_resolution=3600,
    fps=15,
    quiver_scale=quiver_scale,
    quiver_spatial_res=quiver_spatial_res,
    vmax=vmax,
    figsize=figsize,
    add_ax_func=add_problem,
    # Setting 1 if we want just the noise
    # output="noise_animation_15fps.mp4",
    # noise_only=True,
    # Setting 2 for the true currents (fc + noise)
    output="true_black_animation_15fps.mp4",
)
#%% Code snippet to quickly visualize different noise seeds (CDC is 151) and their effects
# get ranges for the plot
t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(), x_T=x_T, deg_around_x0_xT_box=1, temp_horizon_in_s=3600
)
# loop over possible seed_integers and plot all
for i in range(151, 152):
    print(i)
    arena.ocean_field.hindcast_data_source.set_noise_seed(seed_integer=i)
    arena.ocean_field.hindcast_data_source.plot_noise_at_time_over_area(
        time=x_0.date_time + datetime.timedelta(hours=10), x_interval=lon_bnds, y_interval=lat_bnds
    )
#%% plot the FC that the planner sees
ax = arena.ocean_field.forecast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time,
    x_interval=lon_bnds,
    y_interval=lat_bnds,
    return_ax=True,
)
problem.plot(ax=ax)
plt.show()
#%% Plot Reachability Value Function (TTR) when planning with either Forecast or True currents
# Purpose: Check if the optimal trajectory is fundamentally different between FC and True currents
# Setting
with_true_currents = False

# change the observations if planning with true currents
if with_true_currents:
    observation.forecast_data_source = arena.ocean_field.hindcast_data_source

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
    granularity_in_h=10,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=figsize,
    plot_in_h=True,
    # add_drawing=add_planned_traj,
)
#%% Define and Set-up various functions for easy in the loop plotting
# This function is needed to trigger the plotting with replanning just before the planner plans on the new FC again


def check_if_new_fmrc(planner, observation):
    most_current_fmrc_time = observation.forecast_data_source.check_for_most_recent_fmrc_dataframe(
        time=observation.platform_state.date_time
    )
    # Check if this is after our last planned one
    if most_current_fmrc_time != planner.last_fmrc_time_planned_with:
        return True
    else:
        return False


# This function performs back-tracking of the trajectory for a specific rel_time to visualize the re-planning
def plan_traj_from_x_t_and_plot(arena_traj, num_traj_disc, dt_in_sec, ax, rel_time, **kwargs):
    # get x_t from arena trajectory
    idx_x_t = min(
        np.searchsorted(a=arena_traj[:, 2] - planner.current_data_t_0, v=rel_time),
        len(arena_traj) - 1,
    )
    # plot trajectory so far in white
    ax.plot(
        arena_traj[:idx_x_t, 0],
        arena_traj[:idx_x_t, 1],
        color=kwargs.get("traj_color", "black"),
        linewidth=kwargs.get("traj_linewidth", 5),
        linestyle=kwargs.get("traj_linestyle", "-"),
        label="trajectory until x_t",
    )
    # plot trajectory that was planned when forecast came out
    ax.plot(
        planner.planned_trajs[-1]["traj"][0, :],
        planner.planned_trajs[-1]["traj"][1, :],
        linewidth=2,
        color="red",
        linestyle="--",
        label="plan at forecast time",
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
    ax.plot(
        x_traj[0, :],
        x_traj[1, :],
        linewidth=4,
        color="black",
        linestyle="--",
        label="plan at x_t",
    )
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
        label="control at x_t",
        zorder=4,
    )
    plot_plan_from_idx = 5
    # ax.scatter(x_traj[0, plot_plan_from_idx:-1:4], x_traj[1, plot_plan_from_idx:-1:4], c="m", marker="o", s=20)
    ax.quiver(
        x_traj[0, plot_plan_from_idx:-1:24],
        x_traj[1, plot_plan_from_idx:-1:24],
        contr_seq[0, plot_plan_from_idx - 1 : -1 : 24]
        * np.cos(contr_seq[1, plot_plan_from_idx - 1 : -1 : 24]),  # u_vector
        contr_seq[0, plot_plan_from_idx - 1 : -1 : 24]
        * np.sin(contr_seq[1, plot_plan_from_idx - 1 : -1 : 24]),  # v_vector
        color="orchid",
        alpha=0.5,
        scale=10,
        linestyle="-",
        label="planned controls at x_t",
        zorder=4,
    )
    ax.legend()


#%% Closed-Loop Controller Simulation while rendeting a) the TTR planning at every new FC, b) the closed-loop replanning.
# Note: this takes 1h to run, mostly because the backtracking at every point (to visualize replanning) takes long

# prep the loop
problem_status = 0
observation = arena.reset(platform_state=x_0)
_ = planner.get_action(observation=observation)
current_fmrc = None

# Start the Loop (until problem terminates)
while problem_status == 0:
    # Core Loop
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
    problem_status = arena.problem_status(problem=problem)

    # If new FC triggered a re-planning -> plot the TTR function planning
    new_fmrc = planner.last_fmrc_time_planned_with
    if current_fmrc != new_fmrc:
        print("new FRMC after replanning")
        current_fmrc = planner.last_fmrc_time_planned_with
        # Plot planning for each of the forecasts
        planner.plot_reachability_animation(
            time_to_reach=True,
            granularity_in_h=5,
            temporal_resolution=3600,
            fps=15,
            with_opt_ctrl=False,
            figsize=figsize,
            filename="fmrc_idx_{}_with_currents.mp4".format(current_fmrc.timestamp()),
            with_background=True,
            background_animation_args={
                "quiver_scale": quiver_scale,
                "quiver_spatial_res": quiver_spatial_res,
                "vmax": vmax,
            },
        )

    # If new FC becomes available next round or terminating sim -> render re-planning over the previous FC time period
    if check_if_new_fmrc(planner, observation) or problem_status != 0:
        print("new FRMC pre replanning")
        # partial pack the function with most current arena traj
        plan_from_x_t_plot = partial(plan_traj_from_x_t_and_plot, arena.state_trajectory, None, 600)
        planner.plot_reachability_animation(
            time_to_reach=True,
            granularity_in_h=5,
            with_opt_ctrl=False,
            temporal_resolution=1800,
            fps=15,
            filename="replanning_x_t_fmrc_idx_{}_slow.mp4".format(current_fmrc.timestamp()),
            forward_time=True,
            add_drawing=plan_from_x_t_plot,
            with_background=False,
            figsize=figsize,
            t_end=observation.platform_state.date_time,
        )

# Animate the full closed-loop trajectory iteratively
arena.animate_trajectory(
    problem=problem,
    set_title=True,
    temporal_resolution=1800,
    full_traj=False,
    fps=15,
    output="full_traj_iteratively.mp4",
    traj_linestyle="-",
    traj_color="black",
    traj_linewidth=6,
    figsize=figsize,
    vmax=vmax,
    quiver_spatial_res=quiver_spatial_res,
    quiver_scale=quiver_scale,
    x_interval=specific_settings["x_interval"],
    y_interval=specific_settings["y_interval"],
    x_t_marker_color="m",
    x_t_marker_size=40,
)
#%% For debugging: Plot all on a map
arena.plot_all_on_map(problem=problem)
