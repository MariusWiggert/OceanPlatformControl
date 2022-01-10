import sys
from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator import OceanNavSimulator
from datetime import datetime, timezone, timedelta
from ocean_navigation_simulator.planners import HJReach2DPlanner
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt

import hj_reachability as hj
import time

#%% Experiment Settings
from datetime import datetime, timezone, timedelta
t_0 = datetime(2021, 11, 25, 12, 10, 10, tzinfo=timezone.utc)
x_0 = [-96.0, 24, 1]  # lon, lat, battery
x_T = [-86.0, 24.0] # Middle before Cuba

x_T_radius = 0.5
plan_on_gt = True

T_max_sim = 720 # 30 day max it's fundamentally reachable after 544h!

# create list of points within a square area
box_x_around_x_0 = 1.
box_y_around_x_0 = 1.

xmin = x_0[0]-box_x_around_x_0
xmax = x_0[0]+box_x_around_x_0
ymin = x_0[1]-box_y_around_x_0
ymax = x_0[1]+box_y_around_x_0

X, Y = np.mgrid[xmin:xmax:10j, ymin:ymax:10j]

starting_points = np.vstack([X.ravel(), Y.ravel()]).T.tolist()
#%%
# Set the platform configurations
platform_config_dict = {'battery_cap': 500.0, 'u_max': 0.2, 'motor_efficiency': 1.0,
                        'solar_panel_size': 0.5, 'solar_efficiency': 0.2, 'drag_factor': 675}
hindcast_folder = "data/hindcast_test/"
forecast_folder = "data/forecast_test/"
forecast_delay_in_h = 0.
prob = Problem(x_0, x_T, t_0,
               platform_config_dict=platform_config_dict,
               hindcast_folder= hindcast_folder,
               forecast_folder=forecast_folder,
               plan_on_gt = plan_on_gt,
               x_T_radius = .5,
               forecast_delay_in_h=forecast_delay_in_h)
#%%
prob.viz()
#%%
ax = prob.viz(time=None, video=False, filename=None, cut_out_in_deg=10,
         html_render=None, temp_horizon_viz_in_h=None, return_ax=True)
#%%
for start in starting_points:
    ax.scatter(start[0], start[1], c='r', marker='o', s=10, label='start')
plt.show()