import datetime
from tqdm import tqdm

from ocean_navigation_simulator.controllers.SwitchingController import (
    SwitchingController,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units
import matplotlib.pyplot as plt

specific_settings_navigation = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "backward",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 1,
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
    # TODO: This is from the config. So we can perhaps pass the whole config and then pick what we need
    "platform_dict": None,
}

specific_settings_safety = {
    "filepath_distance_map": {
        "bathymetry": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_global_res_0.083_0.083_max_elevation_-150.nc",
        "garbage": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/garbage_patch_distance_res_0.083_0.083_max.nc",
    }
}
specific_settings_switching = dict(
    safety_condition={"base_setting": "on", "area_type": ["bathymetry", "garbage"]},
    safe_distance={"bathymetry": 10, "garbage": 10},  # TODO: change to units,
    safety_controller="ocean_navigation_simulator.controllers.NaiveSafetyController.NaiveSafetyController(problem, self.specific_settings_safety)",
    navigation_controller="ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner(problem, self.specific_settings_navigation)",
)


arena = ArenaFactory.create(
    scenario_name="safety_region1_Copernicus_forecast_Copernicus_hindcast_local"
)


x_0 = PlatformState(
    lon=units.Distance(deg=-158.5),  # -118.5
    lat=units.Distance(deg=28.4),  # 32.8
    date_time=datetime.datetime(2022, 10, 10, 14, 0, tzinfo=datetime.timezone.utc),
)

x_T = SpatialPoint(lon=units.Distance(deg=-158.1), lat=units.Distance(deg=27.9))  # -118, 33

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.1,
)
#%%
# t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
#     x_0=x_0.to_spatio_temporal_point(),
#     x_T=x_T,
#     deg_around_x0_xT_box=1,
#     temp_horizon_in_s=3600 * 24 * 2,
# )

# ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
# )
# ax = arena.garbage_source.plot_mask_from_xarray(
#     xarray=arena.garbage_source.get_data_over_area(x_interval=lon_bnds, y_interval=lat_bnds),
#     var_to_plot="garbage",
#     contour=True,
#     hatches="///",
#     overlay=False,
#     ax=ax,
# )
# problem.plot(ax=ax)
# plt.show()

# ax = arena.garbage_source.plot_data_over_area(
#     x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
# )
# problem.plot(ax=ax)
# plt.show()
#%%

specific_settings_navigation["platform_dict"] = arena.platform.platform_dict

planner = SwitchingController(
    problem, specific_settings_switching, specific_settings_navigation, specific_settings_safety
)


# % Run reachability planner
observation = arena.reset(platform_state=x_0)
# TODO: this needs to be done to calculate the HJ dynamics at least once
# In case that the safety is on
plaction = planner.get_action(observation=observation)
# Calculate reachability
action_nav = planner.navigation_controller.get_action(observation=observation)


planner.navigation_controller.plot_reachability_snapshot(
    rel_time_in_seconds=0,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=False,
    fig_size_inches=(12, 12),
    plot_in_h=True,
)
#%% Let controller run close-loop within the arena
for i in tqdm(range(int(3600 * 22 / 600))):  # 5 days
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
    # problem_status = arena.problem_status(problem=problem)
    # if problem_status != 0:
    #     print(problem_status)
    # if problem_status == 1:
    # TODO: fake it by going if i == 4, break, check what goes wrong
    # TODO: error in plotting_utils.py in add_traj_and_ctrl_at_time:
    #         ctrl_trajectory[0, idx] * np.cos(ctrl_trajectory[1, idx]),  # u_vector
    # IndexError: index 657 is out of bounds for axis 1 with size 657
    # break
    # print("reached")

#%%
garbage_traj = arena.plot_garbage_trajectory_on_timeaxis()
plt.show()
#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, background="garbage")
#%% Animate the trajectory
arena.animate_trajectory(problem=problem, temporal_resolution=7200)
