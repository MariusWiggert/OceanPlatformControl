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
from ocean_navigation_simulator.controllers.NaiveController import NaiveController
from ocean_navigation_simulator.controllers.RandomController import RandomController
from ocean_navigation_simulator.controllers.hj_planners.HJReach2DPlanner import HJReach2DPlanner
from ocean_navigation_simulator.reinforcement_learning.runners.EvaluationRunner import (
    EvaluationRunner,
)
from ocean_navigation_simulator.utils import cluster_utils

print(
    f'Script started @ {datetime.datetime.now(tz=pytz.timezone("US/Pacific")).strftime("%Y_%m_%d_%H_%M_%S")}'
)
script_start_time = time.time()

cluster_utils.init_ray()

config = {
    "scenario_file": "config/reinforcement_learning/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast.yaml",
    # "controller": {
    #     # "name": "CachedHJReach2DPlannerHindcast",
    #     "name": "Random",
    #     # "name": 'HJ Planner Forecast',
    #     "type": RandomController,
    #     "kwargs": {'actions': 8},
    #     "folder": "/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/",
    # },
    "controller": {
        "type": RLController,
        "name": "RLController",
        # "experiment": "/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/increased_area_cnn_2022_11_08_18_34_46/",
        # "checkpoint": 200,
    },
    "missions": {
        "folder": "/seaweed-storage/generation/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/divers_training_improved_2022_10_23_05_10_12/",
        "filter": {
            "no_random": True,
            # "starts_per_target": 1,
            "start": 70204,
            "limit": 2000,
        },
        "seed": None,
    },
    "wandb": {
        "run_id": False,
        "fake_iterations": False,
        "upload_summary": False,
    },
    "ray_options": {
        "max_retries": 10,
        "resources": {
            "CPU": 1.0,
            "GPU": 0.0,
            "RAM": 4000,
            # "Head CPU": 6.0,
            # "Unique CPU": 1.0,
        },
    },
}

for experiment in ["grouped_cnn second_fc_2022_11_20_03_54_18"]:
    for checkpoint in [250, 200]:
        config["controller"]["checkpoint"] = checkpoint
        config["controller"]["experiment"] = (
            "/seaweed-storage/experiments/gulf_of_mexico_Copernicus_forecast_HYCOM_hindcast/"
            + experiment
            + "/"
        )
        eval_runner = EvaluationRunner(
            name=f"cp{checkpoint}",
            config=config,
            verbose=2,
        )

script_time = time.time() - script_start_time
print(
    f"Script finished in {script_time/3600:.0f}h {(script_time%3600)/60:.0f}min {script_time%60:.0f}s."
)
