import datetime
from tqdm import tqdm


# from ocean_navigation_simulator.controllers.PassiveFloatingController import PassiveFloatController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units
from ocean_navigation_simulator.controllers.SwitchingController import SwitchingController


# Initialize the Arena (holds all data sources and the platform, everything except controller)
arena = ArenaFactory.create(scenario_name="safety_gulf_of_mexico_HYCOM_hindcast_local")

# -87.5, 29, should go directly down, did it and lead to more or less canceling out the currents
# -83, 28.5, near florida west coast, should go to middle, with lots of actuation it does, with 0.1m/s it is not moving much
# -83.11, 29.0 current directly onshore, goes away from shore and saves the day. Alternative is to use PassiveFloating Controller to see stranding.
x_0 = PlatformState(
    lon=units.Distance(deg=-83),
    lat=units.Distance(deg=28.95),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-83), lat=units.Distance(deg=28.7))

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.1,
)

t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(), x_T=x_T, deg_around_x0_xT_box=1, temp_horizon_in_s=3600
)

ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
# problem.plot(ax=ax)
# plt.show()


specific_settings_navigation = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "backward",
    "n_time_vector": 200,
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
    # TODO: This is from the config. So we can perhaps pass the whole config and then pick what we need
    "platform_dict": arena.platform.platform_dict,
}

specific_settings_safety = dict(
    filepath_distance_map="data/bathymetry/bathymetry_distance_res_0.083_0.083_max.nc",
)
specific_settings_switching = dict(
    safety_condition="distance",
    safe_distance_to_land=20,  # TODO: change to units,
    safety_controller="ocean_navigation_simulator.controllers.NaiveSafetyController.NaiveSafetyController(problem, self.specific_settings_safety)",
    navigation_controller="ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner.HJReach2DPlanner(problem, self.specific_settings_navigation)",
)

planner = SwitchingController(
    problem, specific_settings_switching, specific_settings_navigation, specific_settings_safety
)


# % Run reachability planner
observation = arena.reset(platform_state=x_0)
# TODO: this needs to be done to calculate the HJ dynamics at least once
# In case that the safety is on
plaction = planner.get_action(observation=observation)
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
for i in tqdm(range(int(3600 * 24 * 3 / 600))):  # 5 days
    action = planner.get_action(observation=observation)
    observation = arena.step(action)

#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem)
print("Need to do to save images...")
#%% Animate the trajectory
arena.animate_trajectory(problem=problem, temporal_resolution=7200)

print("Done")