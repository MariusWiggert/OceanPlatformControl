from src.utils import hycom_utils
from src.planners.ipopt_planner import IpoptPlannerFixCur
import os

from src.problem import Problem
from src.simulation.simulator import Simulator

project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

# # Test 3 long around the vortex
x_0 = [-96.9, 22.8]
x_T = [-96.9, 22.2]

# planner fixed time horizon
T_planner = 806764

# Step 1: set up problem
prob = Problem(fieldset, x_0, x_T, project_dir, config_yaml='platform.yaml', fixed_time_index=None)
prob.viz()
#%%
# Step 2: initialize planner
ipopt_planner = IpoptPlannerFixCur(problem=prob, t_init=T_planner)
ipopt_planner.plot_opt_results()
#%%
# Step 3: init the simulator
sim = Simulator(ipopt_planner, problem=prob, project_dir=project_dir, sim_config='simulator.yaml')
#%%
# Step 4: run the simulator
sim.run(T_planner)
#%%
# Step 5: plot it
sim.plot_trajectory(name='classes_test', plotting_type='battery')