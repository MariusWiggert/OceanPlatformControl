from src.planners.astar_planner import AStarPlanner
from src.evaluation.evaluate_high_level_planner import EvaluatePlanner
from src.problem_set import ProblemSet
from src.simulation.simulator import Simulator
from src.utils import hycom_utils
import os

project_dir = os.path.abspath(os.path.join(os.getcwd()))

# %% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)
u_max = .2

# Create the ProblemSet
problem_set = ProblemSet(fieldset=fieldset, num_problems=20, project_dir=project_dir)

# %% Step 2: init planner
for prob in problem_set.problems[:]:
    prob.viz()
    planner = AStarPlanner(problem=prob)
    planner.show_planned_trajectory()
    sim = Simulator(planner, problem=prob, project_dir=project_dir, sim_config='simulator.yaml')
    sim.run()
    sim.plot_trajectory(gif_name='classes_test', plotting_type='2D')

# %% Step 3: init the simulator and evaluator

evaluator = EvaluatePlanner(planner=planner, problem_set=problem_set, project_dir=project_dir)
# %% Step 3: init the simulator and evaluator
prob.viz()

# %% Step 4: run the evaluator

evaluator.evaluate_planner()
