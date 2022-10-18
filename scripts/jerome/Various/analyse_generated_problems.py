from ocean_navigation_simulator.reinforcement_learning.scripts.GenerationRunner import (
    GenerationRunner,
)

##### Analyse Specific Batch
# BATCH = 63
#
# from ocean_navigation_simulator.environment.NavigationProblem import NavigationProblem
# from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
#
# problem = NavigationProblem.from_dict(problems_df[problems_df['batch'] == BATCH].iloc[0])
# planner = HJReach2DPlanner.from_plan(f'{folder}group_{problem.extra_info["group"]}/batch_{problem.extra_info["batch"]}/', problem=problem)
#
# print('current_data_t_0:', planner.current_data_t_0)
# print('current_data_t_T:', planner.current_data_t_T)

GenerationRunner.plot_generation("/seaweed-storage/generation/increased_planner_area/", n=-1)
