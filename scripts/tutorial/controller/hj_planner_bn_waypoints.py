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
from ocean_navigation_simulator.environment.NavigationWaypointsProblem import (
    NavigationWaypointsProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.utils.misc import set_arena_loggers

set_arena_loggers(logging.DEBUG)

#in form lon,lat
# list_of_waypoints = [(-95.85,19.4),(-89.8,24.5),(-80.3,24.6),(-79.9,25.46)] # Miami to Veracruz
list_of_waypoints = [(-82.5,23.7),(-81.6,24.1),(-80.3,24.6)] # Default with waypoint
# list_of_waypoints = [(-82.5,23.7),(-80.3,24.6)] #Default

total_time = 0

# Initialize the Arena (holds all data sources and the platform, everything except controller)
arena_overall = ArenaFactory.create(
    scenario_name="gulf_of_mexico_HYCOM_hindcast_BN",
    # Note: uncomment this to download the hindcast data if not already locally available
    # t_interval=[
    #     datetime.datetime(2021, 11, 23, 12, 0, tzinfo=datetime.timezone.utc),
    #     datetime.datetime(2021, 11, 30, 12, 0, tzinfo=datetime.timezone.utc)]
)
# we can also download the respective files directly to a temp folder, then t_interval needs to be set
# % Specify Navigation Problem
x_0_overall = PlatformState(
    lon=units.Distance(deg=list_of_waypoints[0][0]),
    lat=units.Distance(deg=list_of_waypoints[0][1]),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_T_overall = SpatialPoint(lon=units.Distance(deg=list_of_waypoints[-1][0]), lat=units.Distance(deg=list_of_waypoints[-1][1]))

problem_overall = NavigationWaypointsProblem(
    start_state=x_0_overall,
    end_region=x_T_overall,
    target_radius=0.1,
)

problem_overall.add_waypoints(list_of_waypoints[1:-1])

# %% Plot the problem on the map
t_interval, lat_bnds, lon_bnds = arena_overall.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0_overall.to_spatio_temporal_point(), x_T=x_T_overall, deg_around_x0_xT_box=1, temp_horizon_in_s=3600
)

ax_overall = arena_overall.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0_overall.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem_overall.plot(ax=ax_overall)
plt.show()


x = 0
for first, second in zip(list_of_waypoints, list_of_waypoints[1:]):

    # Initialize the Arena (holds all data sources and the platform, everything except controller)
    arena = ArenaFactory.create(
        scenario_name="gulf_of_mexico_HYCOM_hindcast_BN",
        # Note: uncomment this to download the hindcast data if not already locally available
        # t_interval=[
        #     datetime.datetime(2021, 11, 23, 12, 0, tzinfo=datetime.timezone.utc),
        #     datetime.datetime(2021, 11, 30, 12, 0, tzinfo=datetime.timezone.utc)]
    )
    # we can also download the respective files directly to a temp folder, then t_interval needs to be set
    # % Specify Navigation Problem
    x_0 = PlatformState(
        lon=units.Distance(deg=first[0]),
        lat=units.Distance(deg=first[1]),
        date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
    )
    x_T = SpatialPoint(lon=units.Distance(deg=second[0]), lat=units.Distance(deg=second[1]))

    problem = NavigationProblem(
        start_state=x_0,
        end_region=x_T,
        target_radius=0.1,
    )

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
    # planner.save_planner_state("saved_planner/")
    # Load it from the folder
    # loaded_planner = HJReach2DPlanner.from_saved_planner_state(folder="saved_planner/", problem=problem)
    # loaded_planner.last_data_source = arena.ocean_field.hindcast_data_source
    # observation = arena.reset(platform_state=x_0)
    # loaded_planner._update_current_data(observation=observation)
    # planner = loaded_planner
    # %% Let the controller run closed-loop within the arena (the simulation loop)
    # Note this runs the sim for a fixed time horizon (does not terminate when reaching the goal)
    #BN: I added this in; it's a flag saying whether the platform has arrived in the target area yet
    unsolved = 1

    for i in tqdm(range(int(3600 * 24 * 5 / 600))):  # 5 days
        action = planner.get_action(observation=observation)
        observation = arena.step(action)

        if (problem.is_done(observation.platform_state) ) and (unsolved):
            print("Waypoint " + str(x) + " reached:")
            print(datetime.timedelta(seconds=i*600))
            unsolved = 0
            total_time += i
    arena.plot_all_on_map(problem=problem)
    x += 1

print("Final destination reached:")
print(total_time)
print(datetime.timedelta(seconds=total_time*600))
    # #%% Plot the arena trajectory on the map
    # arena.plot_all_on_map(problem=problem)
    # #%% Animate the trajectory
    # arena.animate_trajectory(problem=problem, temporal_resolution=7200)