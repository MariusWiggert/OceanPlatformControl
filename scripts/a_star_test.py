from src.planners.astar_planner import AStarPlanner
from src.problem import Problem
from src.simulation.simulator import Simulator
from src.utils import hycom_utils
import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

# Set starting positions
x_0 = [-96.7, 22.2]
x_T = [-97.2, 21.6]

# planner fixed time horizon
T_planner = 116500
#%% Step 1: set up problem
prob = Problem(fieldset, x_0, x_T, project_dir, config_yaml='platform.yaml', fixed_time_index=None)
# problem_set = ProblemSet(fieldset=fieldset)
# prob = problem_set.create_problem()
prob.viz()
#%% Step 2: initialize planner
planner = AStarPlanner(problem=prob, t_init=T_planner)

#%%
planner.show_planned_trajectory()

#%% Step 3: init the simulator
# print(planner.problem.dyn_dict['u_max'])
sim = Simulator(planner, problem=prob, project_dir=project_dir, sim_config='simulator.yaml')


#%% Step 4: run the simulator
# TODO: CURRENTLY DOES NOT WORK! still debugging why this is the case, but the trajectory looks good
sim.run()
#%% Step 5: plot it
sim.plot_trajectory(name='classes_test', plotting_type='2D')

#%%
planner.show_actual_trajectory()
# prob.viz()
# sim.plot_trajectory(name='classes_test')
# sim.plot_trajectory(name='classes_test', plotting_type='gif')
