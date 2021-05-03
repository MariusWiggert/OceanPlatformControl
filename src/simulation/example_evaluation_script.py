""" An example script of how to test different planners, that are all using the same WTC """
from src.planners.astar_planner import AStarPlanner
from src.simulation.data_visualization import DataVisualization
from src.simulation.evaluate_high_level_planner import EvaluatePlanner
from src.problem_set import ProblemSet
from src.simulation.evalute_waypoint_controller import EvaluateWaypointController
from src.simulation.simulator import Simulator
from src.tracking_controllers.minimum_thrust_controller import MinimumThrustController
from src.tracking_controllers.simple_P_tracker import simple_P_tracker
from src.utils import hycom_utils
import os

project_dir = os.path.abspath(os.path.join(os.getcwd()))

# %% Set stuff up
nc_file = 'data/' + "gulf_of_mexico_2020-11-17_fixed_cur_small.nc"
fieldset = hycom_utils.get_hycom_fieldset(nc_file)
u_max = .2
planner = AStarPlanner()
filename = 'waypoint_tracking_problems.txt'

# Save the problems to a file
problem_set = ProblemSet(fieldset=fieldset, WTC=True, planner=planner, num_problems=12, project_dir=project_dir)
problem_set.save_problems(filename)

#%%
evaluate_wypt_contr = EvaluateWaypointController(planner=planner, project_dir=project_dir, filename=filename)
data_dict = evaluate_wypt_contr.compare_WTCs(wypt_contrs=[MinimumThrustController(None), simple_P_tracker(None)])
for WTC, data in data_dict.items():
    print("\nPresenting data for " + WTC)
    visualizer = DataVisualization()
    visualizer.visualize(evaluation_data=data)