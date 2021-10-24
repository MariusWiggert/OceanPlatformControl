import sys
from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator import OceanNavSimulator
from datetime import datetime, timezone
from ocean_navigation_simulator.planners import HJReach2DPlanner
import numpy as np
from datetime import datetime, timezone
# sys.path.extend(['/Volumes/Data/2_Work/2_Graduate_Research/1_Seaweed/Reachability_Code/hj_reachability_c3'])
import hj_reachability as hj
import time
#%% Settings to feed into the planner
# Create the Problem
t_0 = datetime(2021, 6, 1, 12, 10, 10, tzinfo=timezone.utc)
x_0 = [-88.0, 25.0, 1]  # lon, lat, battery
x_T = [-88.2, 26.3]
hindcast_file = "data/hindcast_test/" + "2021_06_1-05_hourly.nc4"
forecast_folder = "data/forecast_test/"
forecast_delay_in_h = 0.
prob = Problem(x_0, x_T, t_0, hindcast_file, forecast_folder, forecast_delay_in_h=forecast_delay_in_h)
# # append to x_0 the posix time for the subsetting of the data
# x_0.append(t_0.timestamp())
#
# # Settings for the planner
# gen_settings = {'conv_m_to_deg': 111120., 'int_pol_type': 'linear', 'temporal_stride': 1}
# specific_settings = {'direction': 'forward', 'T_goal_in_h': 20, 'initial_set_radius': 0.1,
#                      'n_time_vector': 50, 'grid_res': (100, 100), 'deg_around_xt_xT_box': 0.8,
#                      'accuracy': 'high', 'artificial_dissipation_scheme': 'local_local'}
#%% Initialize planner
# planner = HJReach2DPlanner(prob, gen_settings, specific_settings)
# planner.cur_forecast_file = hindcast_file
# planner.plan(x_0)
#%% plot stuff
# planner.plot_reachability()
# planner.plot_2D_traj()
# planner.plot_ctrl_seq()
#%% test other functions
#%% Set stuff up
sim = OceanNavSimulator(sim_config="simulator.yaml", control_config='reach_controller.yaml', problem=prob)
sim.run(T_in_h=70)
#%%
# Step 5: plot from Simulator
# sim.plot_trajectory(name='reach_planner', plotting_type='battery')
sim.plot_trajectory(plotting_type='2D')
sim.plot_trajectory(plotting_type='ctrl')
#%% Plot from reachability planner
sim.high_level_planner.plot_2D_traj()
sim.high_level_planner.plot_ctrl_seq()
#%%
sim.high_level_planner.plot_reachability()