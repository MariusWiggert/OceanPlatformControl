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

timeout_in_days = 15
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
    "timeout": timeout_in_days * 24 * 3600 * 1.2,
    "ocean_dict": {
        "region": "Region 1",
        "hindcast": {
            "field": "OceanCurrents",
            "source": "hindcast_files",
            "source_settings": {
                "local": False,
                "folder": "data/tests/miss_gen_fake_hindcast_3/",
                "source": "Copernicus",
                "type": "hindcast",
            },
        },
        "forecast": {
            "field": "OceanCurrents",
            "source": "hindcast_as_forecast_files",
            "source_settings": {
                "folder": "data/miss_gen_fake_forecast_3/",
                "local": False,
                "source": "HYCOM",
                "type": "hindcast",
                "currents": "total",
            },
            "forecast_length_in_days": 15,
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
t_interval = [datetime.datetime(2022, 10, 4, 0, 0, 0), datetime.datetime(2022, 10, 20, 0, 0, 0)]
# Download files if not there yet
ArenaFactory.download_required_files(
    archive_source="HYCOM",
    archive_type="hindcast",
    region="Region 1",
    download_folder=scenario_config["ocean_dict"]["hindcast"]["source_settings"]["folder"],
    t_interval=t_interval,
    remove_files_if_corrupted=False,
)
ArenaFactory.download_required_files(
    archive_source=scenario_config["ocean_dict"]["forecast"]["source_settings"]["source"],
    archive_type=scenario_config["ocean_dict"]["forecast"]["source_settings"]["type"],
    region="Region 1",
    download_folder=scenario_config["ocean_dict"]["forecast"]["source_settings"]["folder"],
    t_interval=t_interval,
    remove_files_if_corrupted=True,
)

arena = ArenaFactory.create(scenario_config=scenario_config, t_interval=t_interval)


specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "multi-time-reach-back",  # "multi-time-reach-back",
    "n_time_vector": 200,
    "closed_loop": True,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1.5,  # area over which to run HJ_reachability
    "accuracy": "low",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * timeout_in_days,
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    "initial_set_radii": [
        0.01,  # 0.1
        0.01,
    ],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    "grid_res": 0.04,  # 0.02  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    # 'EVM_threshold': 0.3 # in m/s error when floating in forecasted vs sensed currents
    # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
    "platform_dict": arena.platform.platform_dict,
    "obstacle_dict": {
        "path_to_obstacle_file": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_-150.nc",
        "obstacle_value": 1,
        "safe_distance_to_obstacle": 10,
    },
}

# # LA, random regions at edge of obstacles have lower values than target
# x_0 = PlatformState(
#     lon=units.Distance(deg=-119.6),  # -119.7
#     lat=units.Distance(deg=33),
#     date_time=datetime.datetime(2022, 10, 4, 0, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-119.15), lat=units.Distance(deg=33))  # 32.5, 18.7


# # Island
# x_0 = PlatformState(
#     lon=units.Distance(deg=-118.5),
#     lat=units.Distance(deg=29.1),
#     date_time=datetime.datetime(2022, 10, 4, 0, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-118.1), lat=units.Distance(deg=29.2))


# Hawaii, reachable, very nice plot
# x_0 = PlatformState(
#     lon=units.Distance(deg=-155.7),
#     lat=units.Distance(deg=20.5),
#     date_time=datetime.datetime(2022, 10, 4, 0, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-157), lat=units.Distance(deg=20.3))

# Hawaii, on notion, before marius suggestions
# x_0 = PlatformState(
#     lon=units.Distance(deg=-155.7),
#     lat=units.Distance(deg=20.5),
#     date_time=datetime.datetime(2022, 10, 4, 0, 0, tzinfo=datetime.timezone.utc),
# )
# x_T = SpatialPoint(lon=units.Distance(deg=-157.2), lat=units.Distance(deg=20.6))


x_0 = PlatformState(
    lon=units.Distance(deg=-155),
    lat=units.Distance(deg=20.3),
    date_time=datetime.datetime(2022, 10, 4, 0, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-158), lat=units.Distance(deg=19.3))


x_0 = PlatformState(
    lon=units.Distance(deg=-155),
    lat=units.Distance(deg=20.3),
    date_time=datetime.datetime(2022, 10, 4, 0, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-156.5), lat=units.Distance(deg=19.5))


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
    temp_horizon_in_s=specific_settings["T_goal_in_seconds"],
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
#     x_interval=lon_bnds,
#     y_interval=lat_bnds,
#     return_ax=True,
# )
# problem.plot(ax=ax)
# plt.show()

#%% Prepare ontroller and run reachability
planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
observation = arena.reset(platform_state=x_0)
action = planner.get_action(observation=observation)

ax = planner.plot_reachability_snapshot_over_currents(
    rel_time_in_seconds=0,
    granularity_in_h=5,  # 12
    alpha_color=1,
    time_to_reach=True,
    fig_size_inches=(12, 12),
    # plot_in_h=True,
    return_ax=True,
)
# # Many more options to customize the visualization
# ax = arena.bathymetry_source.plot_mask_from_xarray(
#     xarray=arena.bathymetry_source.get_data_over_area(x_interval=lon_bnds, y_interval=lat_bnds),
#     ax=ax,
#     **{"masking_val": -150}
# )
plt.show()

# planner.animate_value_func_3D()
#%% Let controller run close-loop within the arena
for i in tqdm(
    range(int(specific_settings["T_goal_in_seconds"] / scenario_config["platform_dict"]["dt_in_s"]))
):
    action = planner.get_action(observation=observation)
    observation = arena.step(action)


# #%% Plot the arena trajectory on the map
# planner.animate_value_func_3D()
# planner.plot_reachability_animation()

#%% Animate the trajectory
arena.animate_trajectory(
    add_ax_func_ext=arena.add_ax_func_ext_overlay,
    problem=problem,
    temporal_resolution=7200,
    **{"masking_val_bathymetry": -150}
)
