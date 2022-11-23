# %% Run imports

import datetime
from tqdm import tqdm

from ocean_navigation_simulator.environment.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.environment.Platform import PlatformState
from ocean_navigation_simulator.environment.PlatformState import SpatialPoint
from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
from ocean_navigation_simulator.controllers.hj_planners.HJBSeaweed2DPlanner import (
    HJBSeaweed2DPlanner,
)
from ocean_navigation_simulator.utils import units
import matplotlib.pyplot as plt
import os

# %% Initialize
os.chdir(
    "/Users/matthiaskiller/Library/Mobile Documents/com~apple~CloudDocs/Studium/Master RCI/Masters Thesis/Code/OceanPlatformControl"
)
# Initialize the Arena (holds all data sources and the platform, everything except controller)
arena = ArenaFactory.create(scenario_name="gulf_of_mexico_HYCOM_hindcast_local_solar_seaweed")
# we can also download the respective files directly to a temp folder, then t_interval needs to be set
# % Specify Navigation Problem
x_0 = PlatformState(
    lon=units.Distance(deg=-150),
    lat=units.Distance(deg=30),
    date_time=datetime.datetime(2022, 7, 24, 12, 0, tzinfo=datetime.timezone.utc),
)
x_T = SpatialPoint(lon=units.Distance(deg=-120), lat=units.Distance(deg=60))

problem = NavigationProblem(
    start_state=x_0,
    end_region=x_T,
    target_radius=0.1,
    timeout=datetime.timedelta(days=40),
    platform_dict=arena.platform.platform_dict,
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


ax = arena.seaweed_field.hindcast_data_source.plot_data_at_time_over_area(
    time=x_0.date_time, x_interval=lon_bnds, y_interval=lat_bnds, return_ax=True
)
problem.plot(ax=ax)
plt.show()

# %% Instantiate the HJ Planner
specific_settings = {
    "replan_on_new_fmrc": True,
    "replan_every_X_seconds": False,
    "direction": "forward",
    "n_time_vector": 200,
    # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    "deg_around_xt_xT_box": 1.0,  # area over which to run HJ_reachability
    "accuracy": "high",
    "artificial_dissipation_scheme": "local_local",
    "T_goal_in_seconds": 3600 * 24 * 40,
    "use_geographic_coordinate_system": True,
    "progress_bar": True,
    "initial_set_radii": [
        0.1,
        0.1,
    ],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    "grid_res": 0.04,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    "d_max": 0.0,
    # 'EVM_threshold': 0.3 # in m/s error when floating in forecasted vs sensed currents
    # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
    "platform_dict": arena.platform.platform_dict,
}
planner = HJBSeaweed2DPlanner(arena=arena, problem=problem, specific_settings=specific_settings)


# %% Run reachability planner
observation = arena.reset(platform_state=x_0)
action = planner.get_action(observation=observation)
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
print(arena.platform.state.seaweed_mass.kg)


for i in tqdm(range(int(3600 * 24 * 40 / 600))):  # 3 days
    action = planner.get_action(observation=observation)
    observation = arena.step(action)


print(arena.platform.state.seaweed_mass.kg)
arena.plot_seaweed_trajectory_on_timeaxis()

#%% Plot the arena trajectory on the map
arena.plot_all_on_map(problem=problem, background="seaweed")
#%% Animate the trajectory
arena.animate_trajectory(
    problem=problem, temporal_resolution=7200, background="seaweed", output="trajectory_seaweed.mp4"
)

# %%
arena.animate_trajectory(
    problem=problem,
    temporal_resolution=7200,
    background="current",
    output="trajectory_currents.mp4",
)

# %%
arena.plot_seaweed_trajectory_on_timeaxis()
# %%
