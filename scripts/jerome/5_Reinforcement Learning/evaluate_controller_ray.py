import datetime
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

Utils.ray_init(mode='local')

runner = EvaluationRunner(
    config={
        # 'scenario_name': 'gulf_of_mexico_HYCOM_hindcast',
        'scenario_name': 'gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast',
        'missions': {
            'folder': '/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/verification_1000_problems/',
            'limit': 1,
        },
        'controller': {
            'class': RLController,
            # 'experiment': '/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/bigger_custom_model_forecast_and_gp_error_2022_10_02_01_05_48/',
            'experiment': '/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/test_evaluation_2022_10_10_16_35_22/',
            'checkpoint': 'checkpoints/checkpoint_000001/checkpoint-1',
        },
        'ray_options': {
            'max_retries': 10,
            'resources': {
                "CPU": 1.0,
                "GPU": 1.0,
                "RAM": 6000,
                "Head CPU": 1.0,
                # "Worker CPU": 1.0,
            }
        },
        # 'result_folder': '/seaweed-storage/evaluation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/verification_1000_problems/RLController_bigger_custom_model_forecast_and_gp_error_100/',
        'result_folder': '/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/test_evaluation_2022_10_10_16_35_22/verification/',
        'wandb_run_id': '1i4q0g1q',
    },
    verbose=20,
)

script_time = time.time()-script_start_time
print(f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s.")