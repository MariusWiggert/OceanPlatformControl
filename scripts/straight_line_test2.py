from src.planners.straight_line_planner import StraightLinePlanner
from src.simulation.evaluate_planner import EvaluatePlanner
from src.problem_set import ProblemSet
from src.utils import hycom_utils
import os

project_dir = os.path.abspath(os.path.join(os.getcwd()))

# %% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)
u_max = .2

# Create the ProblemSet
filename = 'problems.txt'
problem_set = ProblemSet(fieldset=fieldset, num_problems=5, project_dir=project_dir)

# Checking to make sure saving works properly
problem_set.save_problems(filename)
new_problem_set = ProblemSet(fieldset=fieldset, filename=filename, project_dir=project_dir)
for problem_1, problem_2 in zip(problem_set.problems, new_problem_set.problems):
    assert problem_1.x_0 == problem_2.x_0 and problem_1.x_T == problem_2.x_T
# %% Step 2: init planner

prob = problem_set.problems[0]
planner = StraightLinePlanner(problem=prob)
# %% Step 3: init the simulator and evaluator

evaluator = EvaluatePlanner(planner=planner, problem_set=problem_set, project_dir=project_dir)
# %% Step 4: run the evaluator

evaluator.evaluate_planner()
