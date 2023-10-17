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
    "bathymetry_dict": None,
    "garbage_dict": {
        "field": "Garbage",
        "source": "Lebreton",
        "source_settings": {
            "filepath": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/garbage_patch_global_res_0.083_0.083.nc"
        },
        "casadi_cache_settings": {"deg_around_x_t": 10},
        "use_geographic_coordinate_system": True,
    },
    "solar_dict": {"hindcast": None, "forecast": None},
    "seaweed_dict": {"hindcast": None, "forecast": None},
}
t_interval = [datetime.datetime(2022, 10, 4, 0, 0, 0), datetime.datetime(2022, 10, 7, 0, 0, 0)]
# Download files if not there yet
ArenaFactory.download_required_files(
    archive_source="HYCOM",
    archive_type="hindcast",
    region="Region 1",
    download_folder=scenario_config["ocean_dict"]["hindcast"]["source_settings"]["folder"],
    t_interval=t_interval,
)
ArenaFactory.download_required_files(
    archive_source="copernicus",
    archive_type="forecast",
    region="Region 1",
    download_folder=scenario_config["ocean_dict"]["forecast"]["source_settings"]["folder"],
    t_interval=t_interval,
)

arena = ArenaFactory.create(scenario_config=scenario_config)

x_0 = PlatformState(
    lon=units.Distance(deg=-158.5),
    lat=units.Distance(deg=28.4),
    date_time=datetime.datetime(2022, 10, 4, 0, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-158.1), lat=units.Distance(deg=27.9))

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
    temp_horizon_in_s=3600,
)

ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
ax = arena.garbage_source.plot_mask_from_xarray(
    xarray=arena.garbage_source.get_data_over_area(x_interval=lon_bnds, y_interval=lat_bnds),
    ax=ax,
)
problem.plot(ax=ax)
plt.show()


specific_settings_safety = {
    "filepath_distance_map": {
        "bathymetry": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/bathymetry_distance_res_0.083_0.083_max_elevation_0.nc",
        "garbage": "ocean_navigation_simulator/package_data/bathymetry_and_garbage/garbage_patch_distance_res_0.083_0.083_max.nc",
    }
}
planner = NaiveSafetyController(problem, specific_settings_safety)
#%% Run controller
observation = arena.reset(platform_state=x_0)
#%% Let controller run close-loop within the arena
for i in tqdm(range(int(3600 * 24 * 2 / 600))):  # 5 days
    action = planner.get_action(observation=observation, area_type="garbage")
    observation = arena.step(action)

#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, background="garbage")
arena.plot_all_on_map(problem=problem)

#%% Animate the trajectory
arena.animate_trajectory(
    add_ax_func_ext=arena.add_ax_func_ext_overlay, problem=problem, temporal_resolution=7200
)
