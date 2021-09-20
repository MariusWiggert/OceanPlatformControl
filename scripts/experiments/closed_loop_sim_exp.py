import numpy as np
from src.problem import Problem
from src.simulation.closed_loop_simulator import ClosedLoopSimulator
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
#%%
# Test 1: roughly 110km
t_0 = datetime(2021, 6, 1, 12, 10, 10, tzinfo=timezone.utc)
x_0 = [-88.0, 25.0]    # lon, lat
x_T = [-88.0, 26.0]
hindcast_file = "data/hindcast_test/" + "2021_06_1-05_hourly.nc4"
forecast_folder = "data/forecast_test/"
forecast_delay_in_h = 0.
prob = Problem(x_0, x_T, t_0, hindcast_file, forecast_folder, forecast_delay_in_h=forecast_delay_in_h)
#%%
# prob.viz(time=t_0)
#%%
#%% Set stuff up
sim = ClosedLoopSimulator(sim_config="simulator.yaml", control_config='controller.yaml', problem=prob)
sim.run(T_in_h=50)
#%%
# Step 5: plot it
sim.plot_trajectory(name='ipopt_var_cur', plotting_type='battery')
sim.plot_trajectory(name='ipopt_var_cur', plotting_type='2D')
#%% get gif
sim.plot_trajectory(name='ipopt_var_cur', plotting_type='gif')
#%% plotting experiments
