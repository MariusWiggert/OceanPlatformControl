from src.planners.straight_line_planner import StraightLinePlanner
from src.utils import hycom_utils
import os
from src.problem import Problem
from src.simulation.simulator import Simulator

project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
# nc_file = 'data/' + "gulf_of_mexico_2020-11-01-10_5h.nc4"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

# Set starting positions
x_0 = [-96.6, 22.8]
x_T = [-97.5, 22.2]

#%% Step 1: set up problem
prob = Problem(fieldset, x_0, x_T, project_dir, config_yaml='platform.yaml', fixed_time_index=0)
prob.viz()
#%% Step 2: initialize planner
planner = StraightLinePlanner(problem=prob)
#%% Step 3: init the simulator
sim = Simulator(planner, problem=prob, project_dir=project_dir, sim_config='simulator.yaml')
#%% Step 4: run the simulator
sim.run(T_in_h=24)
#%% Step 5: plot it
# sim.plot_trajectory(name='straight_line', plotting_type='2D')
# sim.plot_trajectory(name='straight_line', plotting_type='battery')
sim.plot_trajectory(name='straight_line_gif', plotting_type='gif')