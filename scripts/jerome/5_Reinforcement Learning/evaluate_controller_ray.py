import time

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import FileMissionProblemFactory
from ocean_navigation_simulator.scripts.EvaluationRunner import EvaluationRunner
from ocean_navigation_simulator.scripts.RayUtils import RayUtils

print('Script started ...')
script_start_time = time.time()

RayUtils.init_ray()


runner = EvaluationRunner(
    scenario_name='gulf_of_mexico_HYCOM_hindcast',
    controller_class=HJReach2DPlanner,
    problem_factory=FileMissionProblemFactory(limit=1, csv_file='missions/training/gulf_of_mexico_HYCOM_hindcast/missions.csv'),
    verbose=10,
)


script_time = time.time()-script_start_time
print(f"Script finished in {script_time/60:.0f}min {script_time%60:.0f}s.")