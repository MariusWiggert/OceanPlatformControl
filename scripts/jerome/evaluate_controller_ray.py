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
from ocean_navigation_simulator.reinforcement_learning.scripts.EvaluationRunner import EvaluationRunner
from ocean_navigation_simulator.controllers.RLController import RLController
from ocean_navigation_simulator.utils import cluster_utils

print(f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y_%m_%d_%H_%M_%S")}')
script_start_time = time.time()

cluster_utils.ray_init()

eval_runner = EvaluationRunner(
    config={
        'scenario_file': 'config/reinforcement_learning/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast',
        'controller': RLController,
        'experiment': '/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/unique_training_data_2022_10_11_20_24_42/',
        'checkpoint': 255,
        'missions': {
            'folder': '/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/verification_1000_problems/',
            'limit': 1000,
            'exclude': [756]
        },
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