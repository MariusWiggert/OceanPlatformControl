import datetime
import logging
import sys
import time
import os
import pytz

os.environ['RAY_DISABLE_MEMORY_MONITOR']='1'
sys.path.extend(['/home/ubuntu/OceanPlatformControl'])
print('Python %s on %s' % (sys.version, sys.platform))
print(sys.path)

from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.reinforcement_learning_scripts.EvaluationRunner import EvaluationRunner
from ocean_navigation_simulator.reinforcement_learning_scripts.Utils import Utils
from ocean_navigation_simulator.controllers.RLController import RLController

print(f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y_%m_%d_%H_%M_%S")}')
script_start_time = time.time()

Utils.ray_init(logging_level="warning")

eval_runner = EvaluationRunner(
    config={
        'scenario_name': 'gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast',
        'missions': {
            'folder': '/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/verification_1000_problems/',
            'limit': 10,
        },
        'controller': RLController,
        'experiment': '/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/test_evaluation_torch_2022_10_11_01_25_52/',
        'checkpoint': 1,
        'ray_options': {
            'max_retries': 10,
            'resources': {
                "CPU": 1.0,
                "GPU": 0.0,
                "RAM": 4000,
                # "Head CPU": 1.0,
                # "Worker CPU": 1.0,
            }
        },
    },
    verbose=2,
)

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")