from src.utils import hycom_utils
from src.planners.ipopt_planner import *
import os

from src.problem import Problem
from src.simulation.simulator import Simulator

project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-01-10_5h.nc4"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

# Test 1 easy follow currents
x_0 = [-96.9, 22.5]
x_T = [-96.3, 21.3]

# planner fixed time horizon
T_planner = 700000

# Step 1: set up problem
prob = Problem(fieldset, x_0, x_T, project_dir, config_yaml='platform.yaml', fixed_time_index=None)
prob.viz()
#%%
# Step 2: initialize planner
ipopt_planner = IpoptPlannerVarCur(problem=prob, t_init=T_planner)
ipopt_planner.plot_opt_results()
#%%
# Step 3: init the simulator
sim = Simulator(ipopt_planner, problem=prob, project_dir=project_dir, sim_config='simulator.yaml')
#%%
# Step 4: run the simulator
sim.run(T=T_planner)
#%%
# Step 5: plot it
sim.plot_trajectory(name='classes_test', plotting_type='battery')
sim.plot_trajectory(name='classes_test', plotting_type='2D')
sim.plot_trajectory(name='ARPA-E', plotting_type='gif')

#%% Compare with actuation vs. without Actuation
opt_traj = sim.trajectory

#%%
ipopt_planner.u_open_loop = np.zeros((2,100))
#%%
sim = Simulator(ipopt_planner, problem=prob, project_dir=project_dir, sim_config='simulator.yaml')
sim.run(T=T_planner)
passive_traj = sim.trajectory

#%%
passive_traj = sim.trajectory
import matplotlib.pyplot as plt
#%%
plt.figure(1)
plt.plot(passive_traj[0, :], passive_traj[1, :], '--', label='passive floatation')
plt.plot(opt_traj[0, :], opt_traj[1, :], '--', label='low-power targeted steering')
plt.title('Simulated Trajectory of Seaweed Platform')
plt.xlabel('lon in deg')
plt.ylabel('lat in deg')
plt.legend()
plt.show()

#%% calculate traj length
dist = 0
for i in range(1, opt_traj.shape[1]):
    dist += np.linalg.norm(opt_traj[:2,i] - opt_traj[:2,i-1])
st_dist = np.linalg.norm(np.array(x_T) - np.array(x_0))


#%% in m
dist = dist*111.120
st_dist = st_dist*111.120