import sys
sys.path.append('.')
from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator import OceanNavSimulator
from datetime import datetime, timezone
from ocean_navigation_simulator.planners import HJReach2DPlanner
import numpy as np
from datetime import datetime, timezone
import os
# sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))) + 'hj_reachability_c3'])
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
#%%
prob.viz() # plots the current at t_0 with the start and goal position
# # create a video of the underlying currents rendered in Jupyter, Safari or as file
# prob.viz(video=True) # renders in Jupyter
# prob.viz(video=True, html_render='safari')    # renders in Safari
# prob.viz(video=True, filename='problem_animation.gif') # saves as gif file
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
#%% Step 5: plot from Simulator
# # plot Battery levels over time
# sim.plot_trajectory(plotting_type='battery')
# # plot 2D Trajectory without background currents
# sim.plot_trajectory(plotting_type='2D')
# # plot control over time
# sim.plot_trajectory(plotting_type='ctrl')
# # plot 2D Trajectory with currents at t_0
# sim.plot_trajectory(plotting_type='2D_w_currents')
# # plot 2D Trajectory with currents at t_0 and control_vec for each point
sim.plot_trajectory(plotting_type='2D_w_currents_w_controls')
#%% plot simulator animation
# TODO: can add also battery level to the animation =)
# # render in Jupyter
# # sim.plot_trajectory(plotting_type='video')
# render in Safari
# sim.plot_trajectory(plotting_type='video', html_render='safari')
# save as gif file
sim.plot_trajectory(plotting_type='video', vid_file_name='sim_anim.gif')
#%% Plot from reachability planner
sim.high_level_planner.plot_2D_traj()
sim.high_level_planner.plot_ctrl_seq()
#%% Plot 2D reachable set evolution
sim.high_level_planner.plot_reachability(render="html")