import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)

scenario_config = {
    "casadi_cache_dict": {"deg_around_x_t": 0.5, "time_around_x_t": 36000.0},
    "platform_dict": {
        "battery_cap_in_wh": 400.0,
        "u_max_in_mps": 0.1,
        "motor_efficiency": 1.0,
        "solar_panel_size": 0.5,
        "solar_efficiency": 0.2,
        "drag_factor": 675.0,
        "dt_in_s": 600.0,
    },
    "use_geographic_coordinate_system": True,
    "spatial_boundary": None,
    "timeout": 259200,
    "ocean_dict": {
        "region": "Region 1",
        "hindcast": {
            "field": "OceanCurrents",
            "source": "hindcast_files",
            "source_settings": {
                "local": True,
                "folder": "data/tests/test_GarbagePatchSource/",
                "source": "HYCOM",
                "type": "hindcast",
            },
        },
        "forecast": {
            "field": "OceanCurrents",
            "source": "forecast_files",
            "source_settings": {
                "local": True,
                "folder": "data/tests/test_GarbagePatchSource/forecast/",
                "source": "copernicus",
                "type": "forecast",
            },
        },
    },
    "bathymetry_dict": {
        "field": "Bathymetry",
        "source": "gebco",
        "source_settings": {
            "filepath": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_global_res_0.083_0.083_max.nc"
        },
        "casadi_cache_settings": {"deg_around_x_t": 20},
        "use_geographic_coordinate_system": True,
    },
    "garbage_dict": None,
    # "garbage_dict": {
    #     "field": "Garbage",
    #     "source": "Lebreton",
    #     "source_settings": {
    #         "filepath": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/garbage_patch_global_res_0.083_0.083.nc"
    #     },
    #     "casadi_cache_settings": {"deg_around_x_t": 10},
    #     "use_geographic_coordinate_system": True,
    # },
    "solar_dict": {"hindcast": None, "forecast": None},
    "seaweed_dict": {"hindcast": None, "forecast": None},
}
t_interval = [datetime.datetime(2022, 10, 4, 0, 0, 0), datetime.datetime(2022, 10, 7, 0, 0, 0)]
# Download files if not there yet
# ArenaFactory.download_required_files(
#     archive_source="HYCOM",
#     archive_type="hindcast",
#     region="Region 1",
#     download_folder=scenario_config["ocean_dict"]["hindcast"]["source_settings"]["folder"],
#     t_interval=t_interval,
# )
# ArenaFactory.download_required_files(
#     archive_source="copernicus",
#     archive_type="forecast",
#     region="Region 1",
#     download_folder=scenario_config["ocean_dict"]["forecast"]["source_settings"]["folder"],
#     t_interval=t_interval,
# )

arena = ArenaFactory.create(scenario_config=scenario_config)


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
    # TODO: add this to config somewhere, change path, add type (dynamic, static, ...)
    "platform_dict": arena.platform.platform_dict,
    "obstacle_dict": {
        "path_to_obstacle_file": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_-150.nc",
        "obstacle_value": 10,
    },
}

x_0 = PlatformState(
    lon=units.Distance(deg=-155.65),
    lat=units.Distance(deg=20.56),
    date_time=datetime.datetime(2022, 10, 4, 0, 0, tzinfo=datetime.timezone.utc),
)

x_T = SpatialPoint(lon=units.Distance(deg=-156.461), lat=units.Distance(deg=20.22))

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.1,
)
#%% Not necessary, but good to visualize
t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(),
    x_T=x_T,
    deg_around_x0_xT_box=1,
    temp_horizon_in_s=3600 * 24 * 1,
)

# ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
#     time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
# )
# ax = arena.bathymetry_source.plot_mask_from_xarray(
#     xarray=arena.bathymetry_source.get_data_over_area(x_interval=lon_bnds, y_interval=lat_bnds),
#     ax=ax,
#     **{"masking_val": -150}
# )
# problem.plot(ax=ax)
# plt.show()

# ax = arena.bathymetry_source.plot_data_over_area(
#     x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True,
# )
# problem.plot(ax=ax)
# plt.show()

#%% Prepare ontroller and run reachability
planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
observation = arena.reset(platform_state=x_0)
# TODO: this needs to be done to calculate the HJ dynamics at least once
# In case that the safety is on
plaction = planner.get_action(observation=observation)
# Calculate reachability
action = planner.get_action(observation=observation)

ax = planner.plot_reachability_snapshot(
    rel_time_in_seconds=3600 * 24 * 2,
    granularity_in_h=5,
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    plot_in_h=True,
    return_ax=True,
)  # ttr True if multitime
planner.plot_reachability_animation()
# Many more options to customize the visualization
ax = arena.bathymetry_source.plot_mask_from_xarray(
    xarray=arena.bathymetry_source.get_data_over_area(x_interval=lon_bnds, y_interval=lat_bnds),
    ax=ax,
    **{"masking_val": -150}
)
plt.show()
#%% Let controller run close-loop within the arena
for i in tqdm(range(int(3600 * 24 * 3 / 600))):  # 1 day
    action = planner.get_action(observation=observation)
    observation = arena.step(action)

#%%
# garbage_traj = arena.plot_garbage_trajectory_on_timeaxis()
# plt.show()
# #%% Plot the arena trajectory on the map
# arena.plot_all_on_map(problem=problem, background="garbage")
#%% Animate the trajectory
arena.animate_trajectory(
    add_ax_func_ext=arena.add_ax_func_ext_overlay,
    problem=problem,
    temporal_resolution=7200,
    **{"masking_val_bathymetry": -150}
)
