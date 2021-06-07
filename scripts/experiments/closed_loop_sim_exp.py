from src.planners.ipopt_planner import *
import numpy as np
from src.problem import Problem
from src.simulation.closed_loop_simulator import ClosedLoopSimulator
from src.utils import simulation_utils

#%%
# Test 1: roughly 150km
x_0 = [-96.9, 22.5]     # lon, lat
x_T = [-96.3, 21.3]
t_0 = np.datetime64('2021-05-06T12:00')
hindcast_file = "data/hind_and_forecasts/hindcast/" + "2021_05_5-15.nc4"
forecast_folder = "data/hind_and_forecasts/forecast/"

prob = Problem(x_0, x_T, t_0, hindcast_file, forecast_folder)
#%% Set stuff up
sim = ClosedLoopSimulator(sim_config="simulator.yaml", control_config='controller.yaml', problem=prob)
#%%
sim.run(T_in_h=224.)
#%%
sim.high_level_planner.plot_opt_results()
#%%
dist = []
for i in range(sim.trajectory.shape[1]):
    d = np.square(sim.trajectory[:2, i] - sim.high_level_planner.x_solver[:, i])
    dist.append(d)
# i = 90
# print(sim.trajectory[:2, i])
# print(sim.high_level_planner.x_solver[:, i])
#%%
import matplotlib.pyplot as plt
plt.plot(np.arange(sim.trajectory.shape[1]), dist)
plt.show()
#%%
# Step 5: plot it
sim.plot_trajectory(name='ipopt_var_cur', plotting_type='battery')
sim.plot_trajectory(name='ipopt_var_cur', plotting_type='2D')
#%%
sim.plot_trajectory(name='ipopt_var_cur', plotting_type='gif')