from src.straight_line.straight_line_planner import StraightLinePlanner
from src.utils.classes import *
from src.utils import hycom_utils
import os
project_dir = os.path.abspath(os.path.join(os.getcwd()))

#%% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)
u_max = .2

# Create the ProblemSet
filename = 'problems.txt'
old_problem_set = ProblemSet(fieldset=fieldset, num_problems=5)
old_problem_set.save_problems(filename)
new_problem_set = ProblemSet(fieldset=fieldset, filename=filename)
#%% Step 1: set up problem
print([(prob.x_0, prob.x_T) for prob in old_problem_set.problems])
print([(prob.x_0, prob.x_T) for prob in new_problem_set.problems])
#%% Step 2: initialize planner

planner = StraightLinePlanner(problem=prob)
#%% Step 3: init the simulator

settings = {'dt': planner.dt, 'conv_m_to_deg': 111120., 'int_pol_type': 'bspline', 'sim_integration': 'rk',
            'project_dir': project_dir}
sim = Simulator(planner, problem=prob, settings=settings)
#%% Step 4: run the simulator

#%% Step 5: plot it

sim.plot_trajectory(name='classes_test', plotting_type='2D')
