from src.planners.astar_planner import AStarPlanner
from src.problem import Problem
from src.simulation.simulator import Simulator
from src.utils import hycom_utils
import os

project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)

# Set starting positions
# Prob 1: going with the flow
# x_0 = [-96.7, 22.2]
# x_T = [-97.2, 21.6]

# Prob 2: against the flow
x_0 = [-97.2, 21.6]
x_T = [-96.7, 22.2]

#%% Step 1: set up problem
prob = Problem(fieldset, x_0, x_T, project_dir, config_yaml='platform.yaml', fixed_time_index=None)
prob.viz()
#%% Step 2: initialize planner
planner = AStarPlanner(problem=prob)
planner.show_planned_trajectory(with_currents=True)
#%% Step 3: init the simulator
sim = Simulator(planner, problem=prob, project_dir=project_dir, sim_config='simulator.yaml')
#%% Step 4: run the simulator until reaching the goal or until max time
sim.run()
#%% Step 5: plot it
# sim.plot_trajectory(name='a_star', plotting_type='2D')
# sim.plot_trajectory(name='a_star', plotting_type='battery')
# sim.plot_trajectory(name='a_star_gif', plotting_type='gif')
planner.show_actual_trajectory(with_currents=True)