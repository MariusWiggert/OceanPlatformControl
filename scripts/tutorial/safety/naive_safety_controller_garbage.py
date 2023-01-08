import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from ocean_navigation_simulator.controllers.NaiveSafetyController import NaiveSafetyController

# from ocean_navigation_simulator.controllers.PassiveFloatingController import PassiveFloatController
from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.NavigationProblem import (
    NavigationProblem,
)
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.utils import units


# Initialize the Arena (holds all data sources and the platform, everything except controller)
arena = ArenaFactory.create(
    scenario_name="safety_region1_Copernicus_forecast_Copernicus_hindcast_local"
)

# -87.5, 29, should go directly down, did it and lead to more or less canceling out the currents
# -83, 28.5, near florida west coast, should go to middle, with lots of actuation it does, with 0.1m/s it is not moving much
# -83.11, 29.0 current directly onshore, goes away from shore and saves the day. Alternative is to use PassiveFloating Controller to see stranding.
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

# Not necessary, but good to visualize
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
    var_to_plot="garbage",
    contour=True,
    hatches="///",
    overlay=False,
    ax=ax,
)
problem.plot(ax=ax)
plt.show()
##


specific_settings = {
    # "d_land_min": 20,
    "filepath_distance_map": {
        "garbage": "data/garbage_patch/garbage_patch_distance_res_0.083_0.083_max.nc",
        "bathymetry": "data/bathymetry/bathymetry_distance_res_0.083_0.083_max.nc",
    },
}
planner = NaiveSafetyController(problem, specific_settings)
# % Run reachability planner
observation = arena.reset(platform_state=x_0)
#%% Let controller run close-loop within the arena
for i in tqdm(range(int(3600 * 24 * 3 / 600))):  # 5 days
    action = planner.get_action(observation=observation, area_type="garbage")
    observation = arena.step(action)

#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, background="garbage")
arena.plot_all_on_map(problem=problem)

#%% Animate the trajectory
larena.animate_trajectory(problem=problem, temporal_resolution=7200)
