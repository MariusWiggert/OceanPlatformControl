import datetime
import numpy as np
from tqdm import tqdm

from ocean_navigation_simulator.env.ArenaFactory import ArenaFactory
from ocean_navigation_simulator.env.Arena import Arena, ArenaObservation
from ocean_navigation_simulator.env.Platform import PlatformState
from ocean_navigation_simulator.env.PlatformState import SpatialPoint
from ocean_navigation_simulator.env.controllers.NaiveToTarget import NaiveToTargetController
from ocean_navigation_simulator.env.Problem import Problem
from ocean_navigation_simulator.env.utils import units
import matplotlib.pyplot as plt
import time
arena = ArenaFactory.create(scenario_name='gulf_of_mexico_files')
#
#% Plot to check if loading worked
t_0 = datetime.datetime(2021, 11, 24, 12, 0, tzinfo=datetime.timezone.utc)
t_interval = [t_0, t_0 + datetime.timedelta(days=4)]
x_interval = [-82, -80]
y_interval = [24, 26]
# x_0 = PlatformState(lon=units.Distance(deg=-81.5), lat=units.Distance(deg=23.5), date_time=t_0)
# x_T = SpatialPoint(lon=units.Distance(deg=-80), lat=units.Distance(deg=24.2))
# Plot Hindcast
# arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(time=t_0 + datetime.timedelta(days=2),
#                                                                    x_interval=x_interval, y_interval=y_interval)
# # Plot Forecast at same time
# xarray_out = arena.ocean_field.forecast_data_source.get_data_over_area(t_interval=t_interval, x_interval=x_interval, y_interval=y_interval)
# ax = arena.ocean_field.forecast_data_source.plot_data_from_xarray(time_idx=49, xarray=xarray_out)
# plt.show()
# forecast_data_source = arena.ocean_field.forecast_data_source
#% Specify Problem
x_0 = PlatformState(lon=units.Distance(deg=-82.5), lat=units.Distance(deg=23.7),
                    date_time=datetime.datetime(2021, 11, 22, 12, 0, tzinfo=datetime.timezone.utc))
x_T = SpatialPoint(lon=units.Distance(deg=-80.3), lat=units.Distance(deg=24.6))
problem = Problem(start_state=x_0, end_region=x_T, target_radius=0.1)
#% Plot the problem function -> To create

#%Instantiate the HJ Planner
from ocean_navigation_simulator.env.controllers.HjPlanners.HJReach2DPlanner import HJReach2DPlanner
specific_settings = {
    'replan_on_new_fmrc': True,
    'replan_every_X_seconds': False,
    'direction': 'multi-time-reach-back',
    'n_time_vector': 100,  # Note that this is the number of time-intervals, the vector is +1 longer because of init_time
    'deg_around_xt_xT_box': 1.,  # area over which to run HJ_reachability
    'accuracy': 'high',
    'artificial_dissipation_scheme': 'local_local',
    'T_goal_in_seconds': 3600*24*4,
    'use_geographic_coordinate_system': True,
    'progress_bar': True,
    'initial_set_radii': [0.1, 0.1],  # this is in deg lat, lon. Note: for Backwards-Reachability this should be bigger.
    # Note: grid_res should always be bigger than initial_set_radii, otherwise reachability behaves weirdly.
    'grid_res': 0.02,  # Note: this is in deg lat, lon (HYCOM Global is 0.083 and Mexico 0.04)
    'd_max': 0.0,
    # 'EVM_threshold': 0.3 # in m/s error when floating in forecasted vs sensed currents
    # 'fwd_back_buffer_in_seconds': 0.5,  # this is the time added to the earliest_to_reach as buffer for forward-backward
    'platform_dict': arena.platform.platform_dict
}
planner = HJReach2DPlanner(problem=problem, specific_settings=specific_settings)
#% Run forward reachability
observation = arena.reset(platform_state=x_0)
action = planner.get_action(observation=observation)
#%% Plot the problem on the map
x_int, y_int, t_interval = arena.get_lon_lat_time_interval(end_region=problem.end_region, margin=1)
ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=t_interval[0],
    x_interval = x_int,
    y_interval = y_int,
    return_ax=True)
problem.plot_on_currents(ax, None)
plt.show()
#%% Various plotting of the reachability computations
planner.plot_reachability_snapshot(rel_time_in_seconds=0, granularity_in_h=5,
                                   alpha_color=1, time_to_reach=True, fig_size_inches=(12, 12), plot_in_h=True)
# planner.plot_reachability_snapshot_over_currents(granularity_in_h = 5, time_to_reach=True)
# planner.plot_reachability_animation(time_to_reach=False)
#%% run closed-loop simulation
for i in tqdm(range(int(3600*24*4/600))):  # 720*10 min = 5 days
    action = planner.get_action(observation=observation)
    observation = arena.step(action)
#%%
# Next Priority: That full workflow should be easy & neat in a few functions.
# -> Let's create the evaluator class that does all of this including termination reasons
# as well as plotting (take stuff out of the arena so that the arena is more light-weight
# as it's running in the loop in RL.
# -> Also: we should build up the classes/structure for a control hierarchy with different re-planning levels.
#%% Plot the trajectory
x_int, y_int, t_interval = arena.get_lon_lat_time_interval(end_region=problem.end_region)
ax = arena.ocean_field.hindcast_data_source.plot_data_at_time_over_area(
    time=t_interval[0],
    x_interval = x_int,
    y_interval = y_int,
    return_ax=True
)
arena.plot_state_trajectory_on_map(ax=ax)
problem.plot_on_currents(ax=ax, data_source=None)
plt.show()