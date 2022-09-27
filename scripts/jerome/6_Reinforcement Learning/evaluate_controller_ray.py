import datetime
import time

import pytz

from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.problem_factories.FileMissionProblemFactory import (
    FileMissionProblemFactory,
)
from ocean_navigation_simulator.reinforcement_learning.scripts.EvaluationRunner import (
    EvaluationRunner,
)
from ocean_navigation_simulator.reinforcement_learning.scripts import cluster_utils

print(
    f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y_%m_%d_%H_%M_%S")}'
)
script_start_time = time.time()

cluster_utils.init_ray()


runner = EvaluationRunner(
    scenario_name="gulf_of_mexico_HYCOM_hindcast",
    controller_class=HJReach2DPlanner,
    problem_factory=FileMissionProblemFactory(
        limit=1, csv_file="missions/training/gulf_of_mexico_HYCOM_hindcast/missions.csv"
    ),
    verbose=10,
)


script_time = time.time() - script_start_time
print(
    f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s."
)
