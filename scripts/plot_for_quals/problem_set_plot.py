import sys
sys.path.extend(['/Volumes/Data/2_Work/2_Graduate_Research/1_Seaweed/OceanPlatformControl'])
from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator import OceanNavSimulator
from ocean_navigation_simulator.planners import HJReach2DPlanner
import numpy as np
import datetime
import os
import hj_reachability as hj
import time
#%% Settings to feed into the planner
# Set the platform configurations
platform_config_dict = {'battery_cap': 500.0, 'u_max': 0.2, 'motor_efficiency': 1.0,
                        'solar_panel_size': 0.5, 'solar_efficiency': 0.2, 'drag_factor': 675}
#%%
# Create the navigation problem
# t_0 = datetime.datetime(2021, 6, 1, 12, 10, 10, tzinfo=datetime.timezone.utc)
# x_0 = [-87.0, 23.0, 1]  # lon, lat, battery
# x_T = [-87.0, 27.5]
# Large circle around the gulf stream
# t_0 = datetime.datetime(2021, 11, 21, 12, 10, 10, tzinfo=datetime.timezone.utc)
# x_0 = [-87.0, 23.5, 1]  # lon, lat, battery
# x_T = [-85.3, 25]
# # From oldest data
# t_0 = datetime.datetime(2021, 11, 21, 12, 10, 10, tzinfo=datetime.timezone.utc)
# x_0 = [-96.9, 22.5, 1]  # lon, lat, battery
# x_T = [-93.0, 21.3]
# # All through gulf of mexico
t_0 = datetime.datetime(2021, 11, 22, 10, 00, 10, tzinfo=datetime.timezone.utc)
x_0 = [-94.5, 25., 1]  # lon, lat, battery
# x_T = [-80.0, 24.0] # all the way in the end
x_T = [-83.0, 24.8] # upward eddy before florida
# # Scenarios for short term tests
# t_0 = datetime.datetime(2021, 11, 21, 12, 10, 10, tzinfo=datetime.timezone.utc)
# x_0 = [-87.5, 24.0, 1]  # lon, lat, battery
# x_T = [-87.0, 28]
# test long-term szenario
# t_0 = datetime.datetime(2021, 11, 25, 12, 10, 10, tzinfo=datetime.timezone.utc)
# x_0 = [-95, 22., 1]  # lon, lat, battery
# # x_T = [-80.0, 24.0] # all the way in the end
# # x_T = [-87, 27.5] # upwards of lower eddie
# x_T = [-86.0, 24.0] # Middle before Cuba
# x_0 = [-96.0, 24, 1]  # lon, lat, battery
# x_T = [-86.0, 24.0] # Middle before Cuba
hindcast_folder = "data/hindcast_test/"
forecast_folder = "data/forecast_test/"
forecast_delay_in_h = 0.
plan_on_gt=False
prob = Problem(x_0, x_T, t_0,
               platform_config_dict=platform_config_dict,
               hindcast_folder= hindcast_folder,
               forecast_folder=forecast_folder,
               plan_on_gt = plan_on_gt,
               x_T_radius = .5,
               forecast_delay_in_h=forecast_delay_in_h)
#%%
# prob.viz(cut_out_in_deg=4)
# #%%
# # import matplotlib.pyplot as plt
# # prob.viz(cut_out_in_deg=11)
# # ax = prob.viz(cut_out_in_deg=11, figsize=(17,12), spatial_shape=(85, 60), return_ax=False) # plots the current at t_0 with the start and goal position
# # # create a video of the underlying currents rendered in Jupyter, Safari or as file
# # prob.viz(video=True) # renders in Jupyter
# # prob.viz(video=True, html_render='safari')    # renders in Safari
# # prob.viz(video=True, filename='problem_animation.gif') # saves as gif file Debug HDF Error
# t_0 = datetime.datetime(2021, 12, 23, 12, 10, 10, tzinfo=datetime.timezone.utc)
# x_0 = [-95, 22., 1]  # lon, lat, battery
# x_T = [-86.0, 24.0] # Middle before Cuba
# hindcast_folder = "data/hindcast_test/"
# forecast_folder = "data/forecast_test/"
# forecast_delay_in_h = 0.
# plan_on_gt=True
# prob = Problem(x_0, x_T, t_0,
#                platform_config_dict=platform_config_dict,
#                hindcast_folder= hindcast_folder,
#                forecast_folder=forecast_folder,
#                plan_on_gt = plan_on_gt,
#                x_T_radius = .5,
#                forecast_delay_in_h=forecast_delay_in_h)
#%%
# # create list of points within a square area
# box_x_around_x_0 = 0.5
# box_y_around_x_0 = 0.5
# #%%
# xmin = x_0[0]-box_x_around_x_0
# xmax = x_0[0]+box_x_around_x_0
# ymin = x_0[1]-box_y_around_x_0
# ymax = x_0[1]+box_y_around_x_0
# X, Y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
# starting_points = np.vstack([X.ravel(), Y.ravel()]).T.tolist()
# #%%
# for start in starting_points:
#     ax.scatter(start[0], start[1], c='r', marker='o', s=100, label='start')
# plt.show()
#%% Set stuff up
sim = OceanNavSimulator(sim_config_dict="simulator.yaml", control_config_dict='reach_controller.yaml', problem=prob)
sim.check_feasibility(T_hours_forward=20, deg_around_xt_xT_box=1.)
# sim.run(T_in_h=5)
# #%% Step 5: plot from Simulator
# # # plot Battery levels over time
# sim.plot_trajectory(plotting_type='battery')
# # # # plot 2D Trajectory without background currents
# sim.plot_trajectory(plotting_type='2D')
# # # plot control over time
# sim.plot_trajectory(plotting_type='ctrl')
# # # plot 2D Trajectory with currents at t_0
# #%%
# sim.plot_trajectory(plotting_type='2D_w_currents', deg_around_x0_xT_box=5, temporal_stride=10)
# # # plot 2D Trajectory with currents at t_0 and control_vec for each point
# #%%
# sim.plot_trajectory(plotting_type='2D_w_currents_w_controls', temporal_stride=10)
# #%%
# # # render in Jupyter
# # # sim.plot_trajectory(plotting_type='video')
# # render in Safari
# # sim.plot_trajectory(plotting_type='video', html_render='safari')
# # save as gif file
# sim.plot_trajectory(plotting_type='video', vid_file_name='sim_animation_5.gif',
#                     deg_around_x0_xT_box=10, temporal_stride=5, linewidth=5, marker=None, linestyle='-')
# #%% Plot from reachability planner
# sim.high_level_planner.plot_2D_traj()
# sim.high_level_planner.plot_ctrl_seq()
# #%%
# import matplotlib.pyplot as plt
# plt.plot(np.arange(10), np.arange(10), marker=None)
# plt.show()
# #%% Plot 2D reachable set evolution
# sim.high_level_planner.plot_reachability(type='safari')
# #%% debug
# import matplotlib.pyplot as plt
# target_region_mask = sim.high_level_planner.get_initial_values(center=np.array(sim.high_level_planner.problem.x_T), direction="backward") <= 0
# #%%
# # plt.imshow(target_region_mask)
# plt.imshow(sim.high_level_planner.all_values[idx, ...] <= 0)
# plt.show()
# #%%
# idx = -1
# reached = np.logical_and(target_region_mask, sim.high_level_planner.all_values[idx, ...] <= 0).any()
# #%%
# sim.high_level_planner.all_values.shape