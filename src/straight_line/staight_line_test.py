from src.straight_line.straight_line_planner import StraightLinePlanner
from src.utils.classes import *
from src.utils import hycom_utils
import os
project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

# Set starting positions
x_0 = [-96.6, 22.8]
x_T = [-97.5, 22.2]

# planner fixed time horizon
T_planner = 116500
#%% Step 1: set up problem
prob = Problem(fieldset, x_0, x_T, project_dir, config_yaml='platform.yaml', fixed_time_index=None)
# problem_set = ProblemSet(fieldset=fieldset)
# prob = problem_set.create_problem()
prob.viz()
#%% Step 2: initialize planner
planner = StraightLinePlanner(problem=prob, t_init=T_planner)
#%% Step 3: init the simulator
sim = Simulator(planner, problem=prob, project_dir=project_dir, sim_config='simulator.yaml')
#%% Step 4: run the simulator
sim.run(T_planner)
#%% Step 5: plot it
# sim.plot_trajectory(name='classes_test', plotting_type='2D')
sim.plot_trajectory(name='classes_test', plotting_type='battery')
# sim.plot_trajectory(name='classes_test', plotting_type='gif')