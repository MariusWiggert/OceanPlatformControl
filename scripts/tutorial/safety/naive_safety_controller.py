import datetime

import matplotlib.pyplot as plt
from tqdm import tqdm

from ocean_navigation_simulator.controllers.NaiveSafetyController import (
    NaiveSafetyController,
)
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units

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
        "region": "GOM",
        "hindcast": {
            "field": "OceanCurrents",
            "source": "hindcast_files",
            "source_settings": {
                "local": True,
                "folder": "data/tests/test_BathymetrySource/",
                "source": "HYCOM",
                "type": "hindcast",
            },
        },
        "forecast": None,
    },
    "bathymetry_dict": {
        "field": "Bathymetry",
        "source": "gebco",
        "source_settings": {
            "filepath": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_global_res_0.083_0.083_max.nc"
        },
        "distance": {
            "safe_distance": 10,
            "filepath": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_-150.nc",
        },
        "casadi_cache_settings": {"deg_around_x_t": 20},
        "use_geographic_coordinate_system": True,
    },
    "garbage_dict": None,
    "solar_dict": {"hindcast": None, "forecast": None},
    "seaweed_dict": {"hindcast": None, "forecast": None},
}
t_interval = [datetime.datetime(2021, 11, 24, 0, 0, 0), datetime.datetime(2021, 12, 5, 0, 0, 0)]
# download files if not there yet
ArenaFactory.download_required_files(
    archive_source="HYCOM",
    archive_type="hindcast",  # should be hindcast once that works on C3
    region="GOM",
    download_folder=scenario_config["ocean_dict"]["hindcast"]["source_settings"]["folder"],
    t_interval=t_interval,
)
arena = ArenaFactory.create(scenario_config=scenario_config)

# -87.5, 29, should go directly down, did it and lead to more or less canceling out the currents
# -83, 28.5, near florida west coast, should go to middle, with lots of actuation it does, with 0.1m/s it is not moving much
# -83.11, 29.0 current directly onshore, goes away from shore and saves the day. Alternative is to use PassiveFloating Controller to see stranding.
x_0 = PlatformState(
    lon=units.Distance(deg=-83),
    lat=units.Distance(deg=28.5),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-80.3), lat=units.Distance(deg=24.6))

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.1,
)

#%% Additional plots for better visualization
t_interval, lat_bnds, lon_bnds = arena.ocean_field.hindcast_data_source.convert_to_x_y_time_bounds(
    x_0=x_0.to_spatio_temporal_point(), x_T=x_T, deg_around_x0_xT_box=1, temp_horizon_in_s=3600
)

ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
# Many more options to customize the visualization
ax = arena.bathymetry_source.plot_mask_from_xarray(
    xarray=arena.bathymetry_source.get_data_over_area(x_interval=lon_bnds, y_interval=lat_bnds),
    ax=ax,
)
problem.plot(ax=ax)
plt.show()
#%% Prepare controller
specific_settings_safety = {
    "filepath_distance_map": {
        "bathymetry": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_0.nc",
        "garbage": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/garbage_patch_distance_res_0.083_0.083_max.nc",
    }
}
planner = NaiveSafetyController(problem, specific_settings_safety)
# % Run controller
observation = arena.reset(platform_state=x_0)
action = planner.get_action(observation=observation)
#%% Let controller run close-loop within the arena
for i in tqdm(range(int(3600 * 24 * 0.5 / 600))):  # 5 days
    action = planner.get_action(observation=observation, area_type="bathymetry")
    observation = arena.step(action)
    problem_status = arena.problem_status(problem=problem)
    print(problem_status)


#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, background="bathymetry")
#%% Animate the trajectory with bathymetry overlay
arena.animate_trajectory(
    add_ax_func_ext=arena.add_ax_func_ext_overlay,
    problem=problem,
    temporal_resolution=7200,
    # output="traj_animation.mp4",
    **{"masking_val_bathymetry": 0}
)
