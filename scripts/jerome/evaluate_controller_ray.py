import datetime
import os
import sys
import time

import pytz


os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"
sys.path.extend(["/home/ubuntu/OceanPlatformControl"])
print("Python %s on %s" % (sys.version, sys.platform))
print(sys.path)

from ocean_navigation_simulator.reinforcement_learning.RLController import RLController
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import (
    HJReach2DPlanner,
)

# from ocean_navigation_simulator.reinforcement_learning.RLController import RLController
from ocean_navigation_simulator.reinforcement_learning.runners.EvaluationRunner import (
    EvaluationRunner,
)
from ocean_navigation_simulator.utils import cluster_utils

print(
    f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y_%m_%d_%H_%M_%S")}'
)
script_start_time = time.time()

cluster_utils.init_ray()

eval_runner = EvaluationRunner(
    config={
        "scenario_file": "config/reinforcement_learning/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
        # "controller": {
        #     "name": "CachedHJReach2DPlannerForecast",
        # "name": 'HJ Planner Forecast',
        # "type": HJReach2DPlanner,
        # "folder": "/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/",
        # },
        "controller": {
            "type": RLController,
            "name": "RLController",
            "experiment": "/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_2022_10_24_13_35_20/",
            "checkpoint": 85,
        },
        "missions": {
            "folder": "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/",
            "filter": {
                "no_random": True,
                "starts_per_target": 1,
                # "start": 0000,
                "limit": 1,
            },
            "seed": None,
        },
        "wandb": {
            "run_id": False,
            "fake_iterations": False,
            "upload_summary": True,
        },
        "ray_options": {
            "max_retries": 10,
            "resources": {
                "CPU": 1.0,
                "GPU": 0.0,
                "RAM": 4000,
                # "Head CPU": 1.0,
                "Unique CPU": 1.0,
            },
        },
    },
    verbose=2,
)

script_time = time.time() - script_start_time
print(
    f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s."
)
