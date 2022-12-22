import datetime
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
arena = ArenaFactory.create(scenario_name="safety_gulf_of_mexico_HYCOM_hindcast_local")

# -87.5, 29, should go directly down, did it and lead to more or less canceling out the currents
# -83, 28.5, near florida west coast, should go to middle, with lots of actuation it does, with 0.1m/s it is not moving much
# -83.11, 29.0 current directly onshore, goes away from shore and saves the day. Alternative is to use PassiveFloating Controller to see stranding.
x_0 = PlatformState(
    lon=units.Distance(deg=-83),
    lat=units.Distance(deg=28.95),
    date_time=datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-80.3), lat=units.Distance(deg=24.6))

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

specific_settings = {
    "d_land_min": 10,
    "filepath_distance_map": "data/bathymetry/bathymetry_distance_res_0.083_0.083_max.nc",
}
planner = NaiveSafetyController(problem, specific_settings)
# % Run reachability planner
observation = arena.reset(platform_state=x_0)
action = planner.get_action(observation=observation)
#%% Let controller run close-loop within the arena
for i in tqdm(range(int(3600 * 24 * 3 / 600))):  # 5 days
    action = planner.get_action(observation=observation)
    observation = arena.step(action)

#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, background="bathymetry")
#%% Animate the trajectory
arena.animate_trajectory(problem=problem, temporal_resolution=7200)
