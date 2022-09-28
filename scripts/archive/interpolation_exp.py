import datetime
import os
import sys
import time

import hj_reachability as hj
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from ocean_navigation_simulator import OceanNavSimulator
from ocean_navigation_simulator.planners import HJReach2DPlanner
from ocean_navigation_simulator.problem import Problem
from ocean_navigation_simulator.utils import plotting_utils, simulation_utils

#%%
platform_config_dict = {
    "battery_cap": 400.0,
    "u_max": 0.1,
    "motor_efficiency": 1.0,
    "solar_panel_size": 0.5,
    "solar_efficiency": 0.2,
    "drag_factor": 675,
}

# Create the navigation problem
t_0 = datetime.datetime(2021, 6, 1, 12, 10, 10, tzinfo=datetime.timezone.utc)
x_0 = [-88.0, 25.0, 1]  # lon, lat, battery
# is on land so we can check the land-mask.
# x_0 = [-88.0, 20.0, 1]  # lon, lat, battery
x_T = [-88.2, 26.3]
hindcast_folder = "data/hindcast_test/"
forecast_folder = "data/forecast_test/"
forecast_delay_in_h = 0.0
plan_on_gt = True
prob = Problem(
    x_0,
    x_T,
    t_0,
    platform_config_dict=platform_config_dict,
    hindcast_source=hindcast_folder,
    forecast_source=forecast_folder,
    plan_on_gt=plan_on_gt,
    forecast_delay_in_h=forecast_delay_in_h,
)

#%% Let's plot the field normally
import datetime

grids_dict, u_data, v_data = simulation_utils.get_current_data_subset(
    t_interval=[
        datetime.datetime(2021, 6, 1, 12, 0, tzinfo=datetime.timezone.utc),
        datetime.datetime(2021, 6, 2, 11, 0, tzinfo=datetime.timezone.utc),
    ],
    lat_interval=[21, 22],
    lon_interval=[-87, -86],
    data_source=prob.hindcast_data_source,
)

# viz
plotting_utils.visualize_currents(
    time=datetime.datetime(2021, 6, 1, 12, 0, tzinfo=datetime.timezone.utc).timestamp(),
    grids_dict=grids_dict,
    u_data=u_data,
    v_data=v_data,
)
#%% Check spatial resolution
grid_dict_new, u_data_new, v_data_new = simulation_utils.spatio_temporal_interpolation(
    grids_dict, u_data, v_data, temp_res_in_h=0.5, spatial_shape=(50, 50), spatial_kind="cubic"
)

plotting_utils.visualize_currents(
    time=datetime.datetime(2021, 6, 1, 12, 0, tzinfo=datetime.timezone.utc).timestamp(),
    grids_dict=grid_dict_new,
    u_data=u_data_new,
    v_data=v_data_new,
)
