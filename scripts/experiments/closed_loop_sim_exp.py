import numpy as np
from src.problem import Problem
from src.simulation.closed_loop_simulator import ClosedLoopSimulator
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
#%%
# Test 1: roughly 150km
t_0 = datetime(2021, 5, 7, 12, 10, 10, tzinfo=timezone.utc)
x_0 = [-96.9, 22.5]     # lon, lat
x_T = [-96.7, 22.3]
hindcast_file = "data/hind_and_forecasts/hindcast/" + "2021_05_5-15.nc4"
forecast_folder = "data/hind_and_forecasts/forecast/"
forecast_delay_in_h = 0.
prob = Problem(x_0, x_T, t_0, hindcast_file, forecast_folder, forecast_delay_in_h=forecast_delay_in_h)
#%% Set stuff up
sim = ClosedLoopSimulator(sim_config="simulator.yaml", control_config='controller.yaml', problem=prob)
sim.run(T_in_h=100)
#%%
# Step 5: plot it
sim.plot_trajectory(name='ipopt_var_cur', plotting_type='battery')
sim.plot_trajectory(name='ipopt_var_cur', plotting_type='2D')
#%% get gif
sim.plot_trajectory(name='ipopt_var_cur', plotting_type='gif')
#%% plotting experiments
